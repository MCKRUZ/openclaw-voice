"""Audio bridge between Discord and processing pipeline.

Handles:
- Receiving per-user audio from Discord (placeholder for Phase 4+)
- Sending TTS audio back to Discord
"""

import asyncio
import threading
from typing import Callable, Optional

import discord
import numpy as np

from utils import audio
from utils.logging import get_logger

logger = get_logger(__name__)


class PipelineAudioSource(discord.AudioSource):
    """
    Audio source that sends TTS audio to Discord.

    Converts processing format (16kHz mono float32) to Discord format
    (48kHz stereo int16) and provides it as 20ms opus frames.
    """

    def __init__(self):
        """Initialize audio source."""
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._lock = threading.Lock()
        self._is_done = False

    def read(self) -> bytes:
        """
        Called by Discord to get next audio frame (runs on sync thread).

        Returns:
            20ms of PCM audio (48kHz stereo int16) or empty bytes if done
        """
        try:
            # Try to get from queue (non-blocking)
            try:
                data = self._queue.get_nowait()
                if data is None:
                    # Sentinel value means we're done
                    self._is_done = True
                    return b""
                return data
            except asyncio.QueueEmpty:
                # No data available, return silence
                silence_frame_size = 960 * 2 * 2  # 20ms @ 48kHz stereo int16
                return b"\x00" * silence_frame_size

        except Exception as e:
            logger.error(f"Error reading audio: {e}")
            return b""

    async def write_audio(self, audio_data: np.ndarray) -> None:
        """
        Write processing audio to be played in Discord.

        Args:
            audio_data: Processing format audio (16kHz mono float32)
        """
        try:
            # Convert to Discord format
            pcm_bytes = audio.processing_to_discord(audio_data)

            # Split into 20ms frames
            frames = audio.split_into_frames(pcm_bytes)

            # Queue all frames
            for frame in frames:
                await self._queue.put(frame)

        except Exception as e:
            logger.error(f"Error writing audio: {e}")

    async def finish(self) -> None:
        """Signal that no more audio will be written."""
        await self._queue.put(None)

    def is_opus(self) -> bool:
        """We provide PCM, not opus."""
        return False

    @property
    def is_done(self) -> bool:
        """Check if playback is complete."""
        return self._is_done


class AudioBridge:
    """
    Manages audio flow between Discord and processing pipeline.

    Handles:
    - Per-user audio reception from Discord (TODO: Phase 4+)
    - Audio callbacks to pipeline
    - TTS audio playback in Discord
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        """
        Initialize audio bridge.

        Args:
            loop: Asyncio event loop
        """
        self.loop = loop
        self._audio_sources: dict[int, PipelineAudioSource] = {}
        self._audio_callback: Optional[Callable[[int, int, bytes], None]] = None

    def set_audio_callback(
        self, callback: Callable[[int, int, bytes], None]
    ) -> None:
        """
        Set callback for received audio.

        Args:
            callback: Async function(guild_id, user_id, pcm_data)
        """
        self._audio_callback = callback

    async def start_receiving(
        self, guild_id: int, voice_client: discord.VoiceClient
    ) -> None:
        """
        Start receiving audio from Discord voice channel.

        NOTE: Audio receiving implementation pending Phase 4+.
        For now, this is a placeholder.

        Args:
            guild_id: Discord guild ID
            voice_client: Connected voice client
        """
        logger.info(
            f"Audio receiving for guild {guild_id}: TODO (Phase 4+)"
        )
        # TODO: Phase 4+ - Implement actual audio receiving
        # Will use voice_client.listen() or custom packet handler

    async def stop_receiving(self, guild_id: int) -> None:
        """
        Stop receiving audio from Discord voice channel.

        Args:
            guild_id: Discord guild ID
        """
        logger.debug(f"Stop receiving audio for guild {guild_id}")

    async def play_audio(
        self,
        guild_id: int,
        voice_client: discord.VoiceClient,
        audio_data: np.ndarray,
    ) -> None:
        """
        Play TTS audio in Discord voice channel.

        Args:
            guild_id: Discord guild ID
            voice_client: Connected voice client
            audio_data: Processing format audio (16kHz mono float32)
        """
        try:
            # Stop any currently playing audio
            if voice_client.is_playing():
                voice_client.stop()

            # Create audio source
            source = PipelineAudioSource()
            self._audio_sources[guild_id] = source

            # Write audio data
            await source.write_audio(audio_data)
            await source.finish()

            # Start playback
            voice_client.play(
                source,
                after=lambda error: self._playback_finished_callback(
                    guild_id, error
                ),
            )

            logger.info(
                f"Started playback for guild {guild_id} "
                f"({len(audio_data)} samples)"
            )

        except Exception as e:
            logger.error(f"Error playing audio for guild {guild_id}: {e}")

    async def stop_playback(
        self, guild_id: int, voice_client: discord.VoiceClient
    ) -> None:
        """
        Stop TTS playback (for barge-in).

        Args:
            guild_id: Discord guild ID
            voice_client: Connected voice client
        """
        if voice_client.is_playing():
            voice_client.stop()
            logger.info(f"Stopped playback for guild {guild_id} (barge-in)")

        # Clean up source
        self._audio_sources.pop(guild_id, None)

    def _playback_finished_callback(
        self, guild_id: int, error: Optional[Exception]
    ) -> None:
        """Called when playback finishes."""
        if error:
            logger.error(f"Playback error for guild {guild_id}: {error}")
        else:
            logger.debug(f"Playback finished for guild {guild_id}")

        # Clean up source
        self._audio_sources.pop(guild_id, None)

    async def cleanup(self) -> None:
        """Clean up all audio bridges."""
        logger.info("Cleaning up audio bridges")

        # Clear sources
        self._audio_sources.clear()
