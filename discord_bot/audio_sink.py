"""Discord audio sink for receiving per-user audio."""

import asyncio
from collections import defaultdict
from typing import Callable, Optional

import discord
import numpy as np

from utils import audio
from utils.logging import get_logger

logger = get_logger(__name__)


class VoiceAudioSink(discord.sinks.Sink):
    """
    Discord audio sink that receives per-user audio.

    Receives audio in Discord format (48kHz stereo int16 20ms frames)
    and forwards to callback for processing.
    """

    def __init__(
        self,
        guild_id: int,
        callback: Callable[[int, int, bytes], None],
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Initialize audio sink.

        Args:
            guild_id: Discord guild ID
            callback: Async callback function(guild_id, user_id, pcm_data)
            loop: Asyncio event loop
        """
        super().__init__()
        self.guild_id = guild_id
        self.callback = callback
        self.loop = loop
        self._user_buffers: dict[int, list[bytes]] = defaultdict(list)
        self._buffer_sizes: dict[int, int] = defaultdict(int)

        # Buffer thresholds (in bytes)
        # 48kHz stereo int16 = 192,000 bytes/sec
        # 500ms = 96,000 bytes
        self.MIN_BUFFER_SIZE = 96000  # 500ms
        self.MAX_BUFFER_SIZE = 960000  # 5 seconds

    def write(self, data: dict[int, discord.sinks.core.RawData], user: discord.User) -> None:
        """
        Called by Discord when audio data is available.

        Args:
            data: Dict mapping user_id to RawData containing PCM frames
            user: Discord user (deprecated parameter)
        """
        try:
            # Process each user's audio
            for user_id, raw_data in data.items():
                # raw_data.data is the PCM audio (48kHz stereo int16)
                if not raw_data.data:
                    continue

                # Add to buffer
                self._user_buffers[user_id].append(raw_data.data)
                self._buffer_sizes[user_id] += len(raw_data.data)

                # If buffer is large enough, process it
                if self._buffer_sizes[user_id] >= self.MIN_BUFFER_SIZE:
                    self._process_user_buffer(user_id)

        except Exception as e:
            logger.error(f"Error in audio sink write: {e}", exc_info=True)

    def _process_user_buffer(self, user_id: int) -> None:
        """
        Process buffered audio for a user.

        Args:
            user_id: Discord user ID
        """
        try:
            # Concatenate all buffered frames
            pcm_data = b"".join(self._user_buffers[user_id])

            # Clear buffer
            self._user_buffers[user_id].clear()
            self._buffer_sizes[user_id] = 0

            # Schedule callback on event loop
            asyncio.run_coroutine_threadsafe(
                self.callback(self.guild_id, user_id, pcm_data),
                self.loop
            )

        except Exception as e:
            logger.error(f"Error processing user buffer: {e}", exc_info=True)

    def cleanup(self) -> None:
        """Called when sink is being destroyed."""
        # Process any remaining buffered audio
        for user_id in list(self._user_buffers.keys()):
            if self._buffer_sizes[user_id] > 0:
                self._process_user_buffer(user_id)

        self._user_buffers.clear()
        self._buffer_sizes.clear()
