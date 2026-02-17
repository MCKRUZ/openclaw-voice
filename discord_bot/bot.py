"""Main Discord bot implementation for Jarvis Voice Bot."""

import asyncio
from typing import Optional, Set

import discord
from discord.ext import tasks
import numpy as np
import torch

from utils.config import Config
from utils.logging import get_logger
from openclaw_client import OpenClawConfig

from .audio_bridge import AudioBridge
from .commands import setup_commands
from .voice_session import VoiceSessionManager
from .vad_receiver import VADAudioReceiver

logger = get_logger(__name__)


class JarvisVoiceBot(discord.Client):
    """Discord bot for voice interaction with AI agents."""

    def __init__(
        self,
        config: Config,
        openclaw_config: Optional[OpenClawConfig] = None,
        tts_synthesizer=None,
        stt_transcriber=None,
        orchestrator=None,
        audio_output_callbacks=None,
    ):
        """
        Initialize the bot.

        Args:
            config: Application configuration
            openclaw_config: OpenClaw Gateway configuration
            tts_synthesizer: Shared TTS synthesizer instance
            stt_transcriber: Shared STT transcriber instance
            orchestrator: Pipeline orchestrator for voice processing
            audio_output_callbacks: Dict to register audio output callbacks
        """
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.voice_states = True
        intents.guild_messages = True

        super().__init__(intents=intents)

        self.config = config
        self.openclaw_config = openclaw_config
        self.tts_synthesizer = tts_synthesizer
        self.stt_transcriber = stt_transcriber
        self.orchestrator = orchestrator
        self.audio_output_callbacks = audio_output_callbacks or {}
        self.tree = discord.app_commands.CommandTree(self)
        self.session_manager = VoiceSessionManager()
        self.audio_bridge: Optional[AudioBridge] = None
        self.vad_receiver: Optional[VADAudioReceiver] = None
        self._ready = False

    async def setup_hook(self) -> None:
        """Called when bot is starting up."""
        logger.info("Setting up bot...")

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        vad_model.eval()
        logger.info("Silero VAD model loaded")

        # Create VAD receiver with callback
        # Use 800ms silence duration to match jarvis-voice-bridge (more reliable)
        self.vad_receiver = VADAudioReceiver(
            vad_model=vad_model,
            vad_threshold=0.5,
            silence_duration_ms=800,
            min_speech_duration_s=0.3,
            on_speech_complete=self.on_speech_complete,
            loop=asyncio.get_event_loop(),
        )

        # Initialize audio bridge with VAD receiver callback
        self.audio_bridge = AudioBridge(asyncio.get_event_loop())

        # Wire audio to VAD receiver instead of on_audio_received
        async def vad_audio_callback(guild_id: int, user_id: int, pcm_data: bytes):
            """Route audio from Discord to VAD receiver."""
            # Get user info
            guild = self.get_guild(guild_id)
            member = guild.get_member(user_id) if guild else None
            user_name = member.display_name if member else f"User{user_id}"

            # Pass to VAD receiver
            if self.vad_receiver:
                self.vad_receiver.on_audio(user_id, user_name, pcm_data)

        self.audio_bridge.set_audio_callback(vad_audio_callback)

        # Register commands
        await setup_commands(self)

        # Sync commands to specific guild immediately
        import os
        guild_id = os.getenv("DISCORD_GUILD_ID")
        if guild_id:
            try:
                guild = discord.Object(id=int(guild_id))

                # Copy global commands to guild for instant availability
                self.tree.copy_global_to(guild=guild)
                logger.info("Copied global commands to guild")

                # Sync to guild
                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} commands to guild {guild_id}")

                for cmd in synced:
                    logger.info(f"  - {cmd.name}")
            except Exception as e:
                logger.error(f"Failed to sync commands in setup_hook: {e}", exc_info=True)

        # Start background tasks
        self.cleanup_task.start()

        logger.info("Bot setup complete")

    async def on_ready(self) -> None:
        """Called when bot is connected to Discord."""
        if self._ready:
            return

        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")

        # Sync slash commands to specific guild for instant availability
        import os
        guild_id = os.getenv("DISCORD_GUILD_ID")

        try:
            if guild_id:
                # Sync to specific guild (instant)
                guild = discord.Object(id=int(guild_id))
                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} slash commands to guild {guild_id}")
            else:
                # Fallback to global sync (takes ~1 hour)
                synced = await self.tree.sync()
                logger.info(f"Synced {len(synced)} slash commands globally")
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
        # Use OpenClaw agent ID if configured, otherwise fall back to config default
        session.current_agent = self.openclaw_config.agent_id if self.openclaw_config else self.config.agents.default
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
        if self.audio_bridge and guild.voice_client:
            await self.audio_bridge.stop_receiving(guild.id, guild.voice_client)

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
        try:
            # Get session
            session = self.session_manager.get_session(guild_id)
            if not session:
                logger.warning(f"Received audio for guild {guild_id} with no session")
                return

            # Ignore if too short (< 200ms)
            duration_ms = len(pcm_data) / (48000 * 2 * 2) * 1000  # 48kHz stereo int16
            if duration_ms < 200:
                return

            # Get user info
            guild = self.get_guild(guild_id)
            member = guild.get_member(user_id) if guild else None
            user_name = member.display_name if member else f"User{user_id}"

            # Pass to VAD receiver (processes in audio thread)
            if self.vad_receiver:
                self.vad_receiver.on_audio(user_id, user_name, pcm_data)

        except Exception as e:
            logger.error(f"Error in on_audio_received: {e}", exc_info=True)

    async def on_speech_complete(
        self, user_id: int, user_name: str, audio: np.ndarray
    ) -> None:
        """
        Called when a complete speech segment is detected.

        Args:
            user_id: Discord user ID
            user_name: User display name
            audio: Complete speech audio (16kHz mono float32)
        """
        try:
            # Find guild for this user
            guild_id = None
            session = None
            for gid, sess in self.session_manager._sessions.items():
                if user_id in sess.active_users:
                    guild_id = gid
                    session = sess
                    break

            if not session:
                logger.warning(f"No session found for user {user_id}")
                return

            duration_s = len(audio) / 16000
            logger.info(f"Processing complete speech from {user_name}: {duration_s:.2f}s")

            # Direct processing: STT → LLM → TTS
            # Transcribe
            if not self.stt_transcriber:
                logger.error("STT transcriber not available")
                return

            logger.info("Transcribing speech...")
            result = await self.stt_transcriber.transcribe(audio, user_id)
            text = result.text if hasattr(result, 'text') else str(result)

            if not text or not text.strip():
                logger.info("Empty transcription, ignoring")
                return

            logger.info(f"Transcribed: '{text}'")

            # Send to OpenClaw Gateway
            if not self.openclaw_config:
                logger.error("OpenClaw Gateway not configured")
                return

            from openclaw_client import OpenClawClient

            client = OpenClawClient(self.openclaw_config)

            agent_id = session.current_agent
            logger.info(f"Sending to Gateway (agent={agent_id})...")

            response = await client.send_message(
                agent=agent_id,
                message=text,
                speaker=f"discord_{user_id}",
            )

            if not response or not response.strip():
                logger.warning("Empty response from Gateway")
                return

            logger.info(f"Gateway response: '{response}'")

            # Synthesize TTS
            if not self.tts_synthesizer:
                logger.error("TTS synthesizer not available")
                return

            # Map agent ID to TTS voice
            # "main" agent uses jarvis voice, "sage" uses sage voice
            if agent_id in ["jarvis", "main"]:
                agent_name = "jarvis"
            else:
                agent_name = "sage"
            logger.info(f"Synthesizing TTS for agent '{agent_name}' (agent_id={agent_id})...")

            tts_audio = await self.tts_synthesizer.synthesize(agent=agent_name, text=response)

            if tts_audio is None or len(tts_audio) == 0:
                logger.warning("TTS synthesis failed or returned empty audio")
                return

            logger.info(f"TTS complete, playing audio ({len(tts_audio)/16000:.2f}s)")

            # Play in Discord
            if self.audio_bridge and session.voice_client:
                await self.audio_bridge.play_audio(
                    guild_id=guild_id,
                    voice_client=session.voice_client,
                    audio_data=tts_audio,
                )
                logger.info("Audio playback started")

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)

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


async def create_bot(
    config: Config,
    openclaw_config: Optional[OpenClawConfig] = None,
    tts_synthesizer=None,
    stt_transcriber=None,
    orchestrator=None,
    audio_output_callbacks=None,
) -> JarvisVoiceBot:
    """
    Create and initialize the Discord bot.

    Args:
        config: Application configuration
        openclaw_config: OpenClaw Gateway configuration
        tts_synthesizer: Shared TTS synthesizer instance
        stt_transcriber: Shared STT transcriber instance
        orchestrator: Pipeline orchestrator for voice processing
        audio_output_callbacks: Dict to register audio output callbacks

    Returns:
        Initialized bot instance
    """
    bot = JarvisVoiceBot(
        config=config,
        openclaw_config=openclaw_config,
        tts_synthesizer=tts_synthesizer,
        stt_transcriber=stt_transcriber,
        orchestrator=orchestrator,
        audio_output_callbacks=audio_output_callbacks,
    )
    return bot


async def run_bot(
    config: Config,
    openclaw_config: Optional[OpenClawConfig] = None,
    tts_synthesizer=None,
    stt_transcriber=None,
    orchestrator=None,
    audio_output_callbacks=None,
) -> None:
    """
    Run the Discord bot.

    Args:
        config: Application configuration
        openclaw_config: OpenClaw Gateway configuration
        tts_synthesizer: Shared TTS synthesizer instance
        stt_transcriber: Shared STT transcriber instance
        orchestrator: Pipeline orchestrator for voice processing
        audio_output_callbacks: Dict to register audio output callbacks
    """
    bot = await create_bot(
        config=config,
        openclaw_config=openclaw_config,
        tts_synthesizer=tts_synthesizer,
        stt_transcriber=stt_transcriber,
        orchestrator=orchestrator,
        audio_output_callbacks=audio_output_callbacks,
    )

    try:
        await bot.start(config.discord.token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        if not bot.is_closed():
            await bot.close()
