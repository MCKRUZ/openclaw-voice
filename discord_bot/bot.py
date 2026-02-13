"""Main Discord bot implementation for Jarvis Voice Bot."""

import asyncio
from typing import Optional, Set

import discord
from discord.ext import tasks

from utils.config import Config
from utils.logging import get_logger

from .audio_bridge import AudioBridge
from .commands import setup_commands
from .voice_session import VoiceSessionManager

logger = get_logger(__name__)


class JarvisVoiceBot(discord.Client):
    """Discord bot for voice interaction with AI agents."""

    def __init__(self, config: Config):
        """
        Initialize the bot.

        Args:
            config: Application configuration
        """
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.voice_states = True
        intents.guild_messages = True

        super().__init__(intents=intents)

        self.config = config
        self.tree = discord.app_commands.CommandTree(self)
        self.session_manager = VoiceSessionManager()
        self.audio_bridge: Optional[AudioBridge] = None
        self._ready = False

    async def setup_hook(self) -> None:
        """Called when bot is starting up."""
        logger.info("Setting up bot...")

        # Initialize audio bridge
        self.audio_bridge = AudioBridge(asyncio.get_event_loop())
        self.audio_bridge.set_audio_callback(self.on_audio_received)

        # Register commands
        await setup_commands(self)

        # Start background tasks
        self.cleanup_task.start()

        logger.info("Bot setup complete")

    async def on_ready(self) -> None:
        """Called when bot is connected to Discord."""
        if self._ready:
            return

        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")

        # Sync slash commands
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=self.config.discord.status_message,
            )
        )

        self._ready = True
        logger.info("Bot is ready!")

    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Called when bot joins a new guild."""
        logger.info(f"Joined guild: {guild.name} (ID: {guild.id})")

        # Sync commands to this guild
        try:
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {guild.id}")
        except Exception as e:
            logger.error(f"Failed to sync commands to guild {guild.id}: {e}")

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Called when bot leaves a guild."""
        logger.info(f"Left guild: {guild.name} (ID: {guild.id})")

        # Clean up any sessions
        if self.session_manager.has_session(guild.id):
            await self.session_manager.remove_session(guild.id)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """
        Called when a user's voice state changes.

        Handles:
        - Users joining/leaving voice channels
        - Bot being disconnected
        - Channel movements
        """
        # Ignore bot's own state changes (handled separately)
        if member.id == self.user.id:
            return

        guild_id = member.guild.id
        session = self.session_manager.get_session(guild_id)

        if session is None:
            # No active session, ignore
            return

        # Check if user joined/left our channel
        before_in_channel = (
            before.channel and before.channel.id == session.channel_id
        )
        after_in_channel = (
            after.channel and after.channel.id == session.channel_id
        )

        if not before_in_channel and after_in_channel:
            # User joined our channel
            session.add_user(member.id)
            logger.info(
                f"User {member.name} joined voice channel in guild {guild_id}"
            )

        elif before_in_channel and not after_in_channel:
            # User left our channel
            session.remove_user(member.id)
            logger.info(
                f"User {member.name} left voice channel in guild {guild_id}"
            )

            # If channel is empty (except bot), consider leaving
            if session.is_empty():
                logger.info(
                    f"Channel empty in guild {guild_id}, will cleanup in background"
                )

    async def on_voice_join(
        self,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        voice_client: discord.VoiceClient,
    ) -> None:
        """
        Called when bot joins a voice channel.

        Args:
            guild: Discord guild
            channel: Voice channel joined
            voice_client: Voice client connection
        """
        logger.info(f"Joining voice channel {channel.name} in guild {guild.name}")

        # Get initial users in channel (excluding bot)
        initial_users: Set[int] = {
            member.id for member in channel.members if not member.bot
        }

        # Create session
        session = await self.session_manager.create_session(
            guild_id=guild.id,
            channel_id=channel.id,
            voice_client=voice_client,
            initial_users=initial_users,
        )

        # Set default agent and sensitivity from config
        session.current_agent = self.config.agents.default
        session.sensitivity = self.config.pipeline.relevance.default_sensitivity

        # Start receiving audio
        if self.audio_bridge:
            await self.audio_bridge.start_receiving(guild.id, voice_client)

        logger.info(
            f"Voice session started for guild {guild.id} with "
            f"{len(initial_users)} users"
        )

    async def on_voice_leave(self, guild: discord.Guild) -> None:
        """
        Called when bot leaves a voice channel.

        Args:
            guild: Discord guild
        """
        logger.info(f"Leaving voice channel in guild {guild.name}")

        # Stop receiving audio
        if self.audio_bridge:
            await self.audio_bridge.stop_receiving(guild.id)

        # Disconnect voice client
        if guild.voice_client:
            await guild.voice_client.disconnect()

        # Remove session
        await self.session_manager.remove_session(guild.id)

        logger.info(f"Voice session ended for guild {guild.id}")

    async def on_audio_received(
        self, guild_id: int, user_id: int, pcm_data: bytes
    ) -> None:
        """
        Called when audio is received from a user.

        Args:
            guild_id: Discord guild ID
            user_id: Discord user ID
            pcm_data: Raw PCM audio (48kHz stereo int16)
        """
        # TODO: Phase 4-11 - Send to pipeline for processing
        # For now, just log reception
        session = self.session_manager.get_session(guild_id)
        if session:
            # Audio received successfully
            pass
        else:
            logger.warning(
                f"Received audio for guild {guild_id} with no session"
            )

    @tasks.loop(minutes=5)
    async def cleanup_task(self) -> None:
        """Background task to cleanup empty sessions."""
        try:
            removed = await self.session_manager.cleanup_empty_sessions()
            if removed > 0:
                logger.info(f"Cleanup task removed {removed} empty sessions")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

    @cleanup_task.before_loop
    async def before_cleanup_task(self) -> None:
        """Wait for bot to be ready before starting cleanup task."""
        await self.wait_until_ready()

    async def close(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down bot...")

        # Stop background tasks
        if self.cleanup_task.is_running():
            self.cleanup_task.cancel()

        # Disconnect from all voice channels
        await self.session_manager.disconnect_all()

        # Cleanup audio bridge
        if self.audio_bridge:
            await self.audio_bridge.cleanup()

        await super().close()

        logger.info("Bot shutdown complete")


async def create_bot(config: Config) -> JarvisVoiceBot:
    """
    Create and initialize the Discord bot.

    Args:
        config: Application configuration

    Returns:
        Initialized bot instance
    """
    bot = JarvisVoiceBot(config)
    return bot


async def run_bot(config: Config) -> None:
    """
    Run the Discord bot.

    Args:
        config: Application configuration
    """
    bot = await create_bot(config)

    try:
        await bot.start(config.discord.token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        if not bot.is_closed():
            await bot.close()
