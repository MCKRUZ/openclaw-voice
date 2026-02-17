"""Voice Activity Detection using Silero VAD.

Detects speech start/end in audio streams for turn-taking and transcription.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch

from utils.logging import get_logger

logger = get_logger(__name__)


class SpeechState(Enum):
    """Current speech detection state."""

    SILENCE = "silence"
    SPEECH = "speech"
    UNKNOWN = "unknown"


@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""

    audio: np.ndarray  # Audio samples (float32)
    start_time: float  # Start time in seconds (relative to stream)
    end_time: float  # End time in seconds
    duration: float  # Duration in seconds
    user_id: int  # User ID who spoke

    @property
    def sample_count(self) -> int:
        """Get number of audio samples."""
        return len(self.audio)


class SileroVAD:
    """
    Silero VAD wrapper for speech detection.

    Silero VAD is a lightweight, fast voice activity detector that runs on CPU.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.3,
        speech_threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.3,
    ):
        """
        Initialize Silero VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000)
            silence_threshold: Silence threshold after speech (seconds)
            speech_threshold: VAD confidence threshold (0.0-1.0)
            min_speech_duration: Minimum speech duration to trigger (seconds)
            min_silence_duration: Minimum silence after speech to end segment
        """
        if sample_rate not in [8000, 16000]:
            raise ValueError(
                f"Silero VAD only supports 8000 or 16000 Hz, got {sample_rate}"
            )

        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.speech_threshold = speech_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        # Load Silero VAD model
        self.model = None
        self._load_model()

        # State tracking
        self.current_state = SpeechState.SILENCE
        self.speech_start_sample = 0
        self.last_speech_sample = 0
        self.accumulated_audio: list[np.ndarray] = []
        self.total_samples_processed = 0

    def _load_model(self) -> None:
        """Load Silero VAD model from torch hub."""
        try:
            logger.info("Loading Silero VAD model...")

            # Load model from torch hub
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            # Extract utility functions
            (get_speech_timestamps, _, read_audio, *_) = utils

            self.model.eval()

            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def process_chunk(self, audio: np.ndarray) -> tuple[SpeechState, Optional[float]]:
        """
        Process an audio chunk and detect speech.

        Args:
            audio: Audio chunk (float32, mono, 16kHz)

        Returns:
            Tuple of (current_state, speech_probability)
        """
        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio.dtype}")

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Run VAD
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        # Debug logging - log speech probability when it's above a minimal threshold
        if speech_prob > 0.1:
            logger.info(f"VAD: speech_prob={speech_prob:.3f}, threshold={self.speech_threshold:.3f}")

        # Determine state based on threshold
        if speech_prob >= self.speech_threshold:
            new_state = SpeechState.SPEECH
            logger.info(f"SPEECH DETECTED! probability={speech_prob:.3f}")
        else:
            new_state = SpeechState.SILENCE

        return new_state, speech_prob

    def process_stream(
        self, audio: np.ndarray
    ) -> tuple[SpeechState, Optional[SpeechSegment]]:
        """
        Process streaming audio and detect speech segments.

        Args:
            audio: Audio chunk to process (float32, mono)

        Returns:
            Tuple of (current_state, speech_segment_if_complete)
        """
        # Process chunk to get speech probability
        state, speech_prob = self.process_chunk(audio)

        # Update total samples
        self.total_samples_processed += len(audio)

        # State machine for speech detection
        if self.current_state == SpeechState.SILENCE:
            if state == SpeechState.SPEECH:
                # Speech started
                self.current_state = SpeechState.SPEECH
                self.speech_start_sample = self.total_samples_processed - len(audio)
                self.last_speech_sample = self.total_samples_processed
                self.accumulated_audio = [audio.copy()]

                logger.debug(
                    f"Speech started at sample {self.speech_start_sample} "
                    f"(prob: {speech_prob:.3f})"
                )

        elif self.current_state == SpeechState.SPEECH:
            # Accumulate audio
            self.accumulated_audio.append(audio.copy())

            if state == SpeechState.SPEECH:
                # Speech continuing
                self.last_speech_sample = self.total_samples_processed

            else:
                # Potential silence
                silence_duration = (
                    self.total_samples_processed - self.last_speech_sample
                ) / self.sample_rate

                if silence_duration >= self.min_silence_duration:
                    # Speech ended - create segment
                    segment = self._create_segment()

                    # Reset state
                    self.current_state = SpeechState.SILENCE
                    self.accumulated_audio = []

                    logger.debug(
                        f"Speech ended after {segment.duration:.2f}s "
                        f"(silence: {silence_duration:.2f}s)"
                    )

                    return self.current_state, segment

        return self.current_state, None

    def _create_segment(self) -> SpeechSegment:
        """
        Create a speech segment from accumulated audio.

        Returns:
            SpeechSegment
        """
        # Concatenate accumulated audio
        audio = np.concatenate(self.accumulated_audio)

        # Calculate times
        start_time = self.speech_start_sample / self.sample_rate
        end_time = self.last_speech_sample / self.sample_rate
        duration = end_time - start_time

        segment = SpeechSegment(
            audio=audio,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            user_id=0,  # Will be set by caller
        )

        return segment

    def reset(self) -> None:
        """Reset VAD state (for new stream or user)."""
        self.current_state = SpeechState.SILENCE
        self.speech_start_sample = 0
        self.last_speech_sample = 0
        self.accumulated_audio = []
        self.total_samples_processed = 0

        logger.debug("VAD state reset")

    def force_end_speech(self) -> Optional[SpeechSegment]:
        """
        Force end current speech segment (if any).

        Useful when user leaves or stream ends.

        Returns:
            SpeechSegment if speech was active, None otherwise
        """
        if self.current_state == SpeechState.SPEECH:
            segment = self._create_segment()
            self.current_state = SpeechState.SILENCE
            self.accumulated_audio = []

            logger.debug(f"Forced speech end after {segment.duration:.2f}s")

            return segment

        return None

    def get_state(self) -> SpeechState:
        """Get current speech detection state."""
        return self.current_state

    def is_speech_active(self) -> bool:
        """Check if speech is currently being detected."""
        return self.current_state == SpeechState.SPEECH


class PerUserVAD:
    """
    Manages VAD instances for multiple users.

    Maintains separate VAD state for each user in a voice channel.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.3,
        speech_threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        speech_callback: Optional[Callable[[int, SpeechSegment], None]] = None,
    ):
        """
        Initialize per-user VAD manager.

        Args:
            sample_rate: Audio sample rate
            silence_threshold: Silence duration threshold
            speech_threshold: VAD confidence threshold
            min_speech_duration: Minimum speech duration
            speech_callback: Async callback when speech segment detected
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.speech_threshold = speech_threshold
        self.min_speech_duration = min_speech_duration
        self.speech_callback = speech_callback

        self._vad_instances: dict[int, SileroVAD] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_vad(self, user_id: int) -> SileroVAD:
        """
        Get VAD instance for a user, creating if necessary.

        Args:
            user_id: User ID

        Returns:
            SileroVAD instance
        """
        async with self._lock:
            if user_id not in self._vad_instances:
                self._vad_instances[user_id] = SileroVAD(
                    sample_rate=self.sample_rate,
                    silence_threshold=self.silence_threshold,
                    speech_threshold=self.speech_threshold,
                    min_speech_duration=self.min_speech_duration,
                )
                logger.debug(f"Created VAD instance for user {user_id}")

            return self._vad_instances[user_id]

    async def process_audio(
        self, user_id: int, audio: np.ndarray
    ) -> Optional[SpeechSegment]:
        """
        Process audio for a user and detect speech.

        Args:
            user_id: User ID
            audio: Audio chunk (float32, mono)

        Returns:
            SpeechSegment if speech segment completed, None otherwise
        """
        vad = await self.get_or_create_vad(user_id)

        # Process audio
        state, segment = vad.process_stream(audio)

        # If segment completed, set user_id and invoke callback
        if segment is not None:
            segment.user_id = user_id

            if self.speech_callback:
                await self.speech_callback(user_id, segment)

        return segment

    async def reset_user(self, user_id: int) -> None:
        """
        Reset VAD state for a user.

        Args:
            user_id: User ID
        """
        async with self._lock:
            if user_id in self._vad_instances:
                self._vad_instances[user_id].reset()

    async def remove_user(self, user_id: int) -> None:
        """
        Remove VAD instance for a user.

        Args:
            user_id: User ID
        """
        async with self._lock:
            if user_id in self._vad_instances:
                # Force end any active speech
                vad = self._vad_instances[user_id]
                segment = vad.force_end_speech()

                if segment is not None:
                    segment.user_id = user_id
                    if self.speech_callback:
                        await self.speech_callback(user_id, segment)

                del self._vad_instances[user_id]
                logger.debug(f"Removed VAD instance for user {user_id}")

    async def get_active_users(self) -> list[int]:
        """
        Get list of users with active VAD instances.

        Returns:
            List of user IDs
        """
        async with self._lock:
            return list(self._vad_instances.keys())

    async def get_speaking_users(self) -> list[int]:
        """
        Get list of users currently speaking.

        Returns:
            List of user IDs
        """
        async with self._lock:
            return [
                user_id
                for user_id, vad in self._vad_instances.items()
                if vad.is_speech_active()
            ]

    async def remove_all(self) -> None:
        """Remove all VAD instances."""
        async with self._lock:
            self._vad_instances.clear()
            logger.debug("Removed all VAD instances")

    def __len__(self) -> int:
        """Get number of VAD instances."""
        return len(self._vad_instances)

    def __repr__(self) -> str:
        """String representation."""
        return f"PerUserVAD(users={len(self._vad_instances)})"
