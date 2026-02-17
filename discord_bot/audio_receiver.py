"""Discord audio receiver using discord-ext-voice_recv."""

import asyncio
from collections import defaultdict
from typing import Callable

import discord

from utils.logging import get_logger

try:
    from discord.ext import voice_recv
    HAS_VOICE_RECV = True
except ImportError:
    voice_recv = None
    HAS_VOICE_RECV = False

logger = get_logger(__name__)


class AudioReceiver:
    """
    Receives audio from Discord voice channel using discord-ext-voice_recv.

    Buffers audio per user and calls callback when enough data is accumulated.
    """

    def __init__(
        self,
        guild_id: int,
        voice_client: discord.VoiceClient,
        callback: Callable[[int, int, bytes], None],
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Initialize audio receiver.

        Args:
            guild_id: Discord guild ID
            voice_client: Connected voice client
            callback: Async callback function(guild_id, user_id, pcm_data)
            loop: Asyncio event loop
        """
        self.guild_id = guild_id
        self.voice_client = voice_client
        self.callback = callback
        self.loop = loop
        self._user_buffers: dict[int, list[bytes]] = defaultdict(list)
        self._buffer_sizes: dict[int, int] = defaultdict(int)
        self._running = False
        self._packet_count = 0

        # Buffer thresholds (in bytes)
        # 48kHz stereo int16 = 192,000 bytes/sec
        # 500ms = 96,000 bytes
        self.MIN_BUFFER_SIZE = 96000  # 500ms
        self.MAX_BUFFER_SIZE = 960000  # 5 seconds

    def start(self) -> None:
        """Start receiving audio."""
        if self._running:
            return

        if not HAS_VOICE_RECV:
            logger.error(
                "voice_recv not available. Install discord-ext-voice-recv. "
                "Audio receive will NOT work."
            )
            return

        try:
            self._running = True

            # Create sink with callback
            sink = voice_recv.BasicSink(self._on_audio_packet)

            # Start listening
            self.voice_client.listen(sink)

            logger.info(f"Started audio receiving for guild {self.guild_id}")

        except Exception as e:
            logger.error(f"Failed to start audio receiving: {e}", exc_info=True)
            self._running = False

    def stop(self) -> None:
        """Stop receiving audio."""
        if not self._running:
            return

        self._running = False

        try:
            # Stop listening
            if self.voice_client:
                self.voice_client.stop_listening()

            # Process any remaining buffered audio
            for user_id in list(self._user_buffers.keys()):
                if self._buffer_sizes[user_id] > 0:
                    self._process_user_buffer(user_id)

            self._user_buffers.clear()
            self._buffer_sizes.clear()

            logger.info(f"Stopped audio receiving for guild {self.guild_id}")

        except Exception as e:
            logger.error(f"Error stopping audio receiving: {e}", exc_info=True)

    def _on_audio_packet(self, user, data) -> None:
        """
        Called by voice_recv for each audio packet (runs on audio thread).

        Args:
            user: Discord user who sent the packet (can be None)
            data: Audio data object with .pcm attribute
        """
        if not self._running:
            return

        # Ignore bot users and None
        if user is None or user.bot:
            return

        try:
            user_id = user.id
            pcm_data = data.pcm  # Raw PCM bytes (48kHz stereo int16)

            if not pcm_data:
                return

            self._packet_count += 1

            # Log occasionally
            if self._packet_count <= 3 or self._packet_count % 500 == 0:
                logger.info(
                    f"Audio packet #{self._packet_count} from {user.display_name}: {len(pcm_data)} bytes"
                )

            # Add to buffer
            self._user_buffers[user_id].append(pcm_data)
            self._buffer_sizes[user_id] += len(pcm_data)

            # If buffer is large enough, process it
            if self._buffer_sizes[user_id] >= self.MIN_BUFFER_SIZE:
                self._process_user_buffer(user_id)

        except Exception as e:
            logger.error(f"Error processing audio packet: {e}", exc_info=True)

    def _process_user_buffer(self, user_id: int) -> None:
        """
        Process buffered audio for a user.

        Args:
            user_id: Discord user ID
        """
        try:
            # Concatenate all buffered packets
            pcm_data = b"".join(self._user_buffers[user_id])

            # Clear buffer
            self._user_buffers[user_id].clear()
            self._buffer_sizes[user_id] = 0

            # Schedule callback on event loop (we're on audio thread)
            asyncio.run_coroutine_threadsafe(
                self.callback(self.guild_id, user_id, pcm_data), self.loop
            )

        except Exception as e:
            logger.error(f"Error processing user buffer: {e}", exc_info=True)
