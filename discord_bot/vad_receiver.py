"""VAD-based audio receiver for Discord with sample-based timing.

Processes audio with Silero VAD in the callback thread using sample-based timing
(not wall-clock) for accurate silence detection. Accumulates speech+silence and
triggers processing when silence threshold is exceeded.

Key features:
- Sample-based timing for accurate silence detection (avoids processing delays)
- Per-user audio buffers with independent VAD state
- LSTM state management for switching between users
- Configurable silence threshold and minimum speech duration
"""

import asyncio
import logging
import threading
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Discord audio format
DISCORD_SAMPLE_RATE = 48_000
TARGET_SAMPLE_RATE = 16_000
DOWNSAMPLE_FACTOR = DISCORD_SAMPLE_RATE // TARGET_SAMPLE_RATE

# Silero VAD expects 512 samples at 16 kHz
VAD_CHUNK_SAMPLES = 512


class UserAudioBuffer:
    """Per-user audio buffer with VAD state tracking."""

    def __init__(self, user_id: int, user_name: str):
        self.user_id = user_id
        self.user_name = user_name

        # Accumulated audio chunks (16kHz mono float32)
        self.audio_chunks: list[np.ndarray] = []

        # VAD buffer for incomplete chunks
        self.vad_buffer = np.empty(0, dtype=np.float32)

        # Speech state (using SAMPLE-BASED timing, not wall-clock!)
        self.is_speaking = False
        self.total_samples_processed = 0
        self.speech_start_sample = 0
        self.silence_start_sample: Optional[int] = None

    def reset(self) -> None:
        """Reset buffer state."""
        self.audio_chunks.clear()
        self.vad_buffer = np.empty(0, dtype=np.float32)
        self.is_speaking = False
        self.total_samples_processed = 0
        self.speech_start_sample = 0
        self.silence_start_sample = None

    def get_speech_audio(self) -> np.ndarray:
        """Get accumulated speech as single array."""
        if not self.audio_chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.audio_chunks)


class VADAudioReceiver:
    """
    VAD-based audio receiver for Discord.

    Processes audio in the callback thread using Silero VAD,
    accumulates complete utterances, and triggers callbacks.
    """

    def __init__(
        self,
        vad_model,
        vad_threshold: float = 0.5,
        silence_duration_ms: float = 300,
        min_speech_duration_s: float = 0.3,
        on_speech_complete: Optional[Callable] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize VAD audio receiver.

        Args:
            vad_model: Silero VAD model
            vad_threshold: VAD confidence threshold (0.0-1.0)
            silence_duration_ms: Silence duration to end speech (milliseconds)
            min_speech_duration_s: Minimum speech duration to process (seconds)
            on_speech_complete: Async callback(user_id, user_name, audio_array)
            loop: Event loop for running callbacks
        """
        self.vad_model = vad_model
        self.vad_model.eval()
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_s = min_speech_duration_s
        self.on_speech_complete = on_speech_complete
        self.loop = loop or asyncio.get_event_loop()

        # Per-user buffers
        self._buffers: dict[int, UserAudioBuffer] = {}
        self._lock = threading.Lock()

        # Track last user for VAD state reset
        self._last_vad_user: Optional[int] = None

        logger.info(
            f"VAD audio receiver initialized "
            f"(threshold={vad_threshold}, silence={silence_duration_ms}ms)"
        )

    def _get_buffer(self, user_id: int, user_name: str) -> UserAudioBuffer:
        """Get or create buffer for user."""
        if user_id not in self._buffers:
            self._buffers[user_id] = UserAudioBuffer(user_id, user_name)
            logger.debug(f"Created audio buffer for {user_name} ({user_id})")
        return self._buffers[user_id]

    def on_audio(self, user_id: int, user_name: str, pcm_data: bytes) -> None:
        """
        Process incoming audio from Discord.

        Called from Discord's audio thread - keep it fast!

        Args:
            user_id: Discord user ID
            user_name: User display name
            pcm_data: Raw PCM audio (48kHz stereo int16)
        """
        with self._lock:
            buf = self._get_buffer(user_id, user_name)

            # Convert Discord format to pipeline format
            # bytes → int16 stereo → float32 mono → downsample to 16kHz
            samples = np.frombuffer(pcm_data, dtype=np.int16)

            # Stereo → mono (average channels)
            if len(samples) % 2 == 0:
                stereo = samples.reshape(-1, 2)
                mono = stereo.mean(axis=1).astype(np.float32) / 32768.0
            else:
                mono = samples.astype(np.float32) / 32768.0

            # Downsample 48kHz → 16kHz (take every 3rd sample)
            downsampled = mono[::DOWNSAMPLE_FACTOR]

            # Append to VAD buffer
            buf.vad_buffer = np.concatenate([buf.vad_buffer, downsampled])

            # Reset VAD LSTM state when switching between users
            if self._last_vad_user != user_id:
                self.vad_model.reset_states()
                self._last_vad_user = user_id
                logger.debug(f"Reset VAD state for {user_name}")

            # Process VAD in chunks
            while len(buf.vad_buffer) >= VAD_CHUNK_SAMPLES:
                chunk = buf.vad_buffer[:VAD_CHUNK_SAMPLES]
                buf.vad_buffer = buf.vad_buffer[VAD_CHUNK_SAMPLES:]

                # Update sample counter (CRITICAL: use audio time, not wall-clock time!)
                buf.total_samples_processed += VAD_CHUNK_SAMPLES

                # Run VAD on chunk
                chunk_tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    speech_prob = self.vad_model(chunk_tensor, TARGET_SAMPLE_RATE).item()

                is_speech = speech_prob >= self.vad_threshold

                if is_speech:
                    # Speech detected
                    buf.silence_start_sample = None

                    if not buf.is_speaking:
                        # Speech start
                        buf.is_speaking = True
                        buf.speech_start_sample = buf.total_samples_processed
                        buf.audio_chunks.clear()
                        logger.info(f"Speech started: {user_name} (prob={speech_prob:.3f})")

                    # Accumulate audio during speech
                    buf.audio_chunks.append(chunk.copy())

                elif buf.is_speaking:
                    # Silence during speech - keep accumulating
                    buf.audio_chunks.append(chunk.copy())

                    if buf.silence_start_sample is None:
                        # First silence chunk after speech
                        buf.silence_start_sample = buf.total_samples_processed
                        logger.debug(f"Silence started for {user_name}")

                    else:
                        # Check if silence duration exceeded (using SAMPLE-BASED timing)
                        silence_samples = buf.total_samples_processed - buf.silence_start_sample
                        silence_duration_ms = (silence_samples / TARGET_SAMPLE_RATE) * 1000

                        if silence_duration_ms >= self.silence_duration_ms:
                            # Speech complete!
                            audio = buf.get_speech_audio()
                            duration_s = len(audio) / TARGET_SAMPLE_RATE

                            logger.info(
                                f"Speech complete: {user_name} "
                                f"({duration_s:.2f}s, "
                                f"silence: {silence_duration_ms:.0f}ms)"
                            )

                            # Reset buffer
                            buf.reset()

                            # Trigger callback if audio is long enough
                            if duration_s >= self.min_speech_duration_s:
                                if self.on_speech_complete:
                                    asyncio.run_coroutine_threadsafe(
                                        self.on_speech_complete(user_id, user_name, audio),
                                        self.loop,
                                    )
                            else:
                                logger.debug(
                                    f"Ignoring short speech: {user_name} ({duration_s:.2f}s)"
                                )

    def clear_user(self, user_id: int) -> None:
        """Clear buffer for user (when they leave)."""
        with self._lock:
            if user_id in self._buffers:
                user_name = self._buffers[user_id].user_name
                del self._buffers[user_id]
                logger.info(f"Cleared audio buffer for {user_name} ({user_id})")

    def clear_all(self) -> None:
        """Clear all user buffers."""
        with self._lock:
            self._buffers.clear()
            logger.info("Cleared all audio buffers")
