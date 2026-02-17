"""Discord slash commands for the Jarvis Voice Bot."""

from typing import Optional

import discord
from discord import app_commands

from utils.logging import get_logger

try:
    from discord.ext import voice_recv
    HAS_VOICE_RECV = True
except ImportError:
    voice_recv = None
    HAS_VOICE_RECV = False

logger = get_logger(__name__)


class VoiceBotCommands(app_commands.Group):
    """Slash command group for voice bot controls."""

    def __init__(self, bot):
        """Initialize command group."""
        super().__init__(name="jarvis", description="Jarvis Voice Bot commands")
        self.bot = bot
        self.agent_name = "jarvis"

    @app_commands.command(
        name="join",
        description="Join your voice channel as Jarvis",
    )
    @app_commands.describe(channel="Voice channel to join (optional)")
    async def join(
        self,
        interaction: discord.Interaction,
        channel: Optional[discord.VoiceChannel] = None,
    ):
        """Join a voice channel as Jarvis."""
        await self._join_with_agent(interaction, channel, self.agent_name)

    async def _join_with_agent(
        self,
        interaction: discord.Interaction,
        channel: Optional[discord.VoiceChannel],
        agent: str,
    ):
        """Join voice channel and set agent."""
        await interaction.response.defer(thinking=True)

        try:
            # Determine which channel to join
            target_channel = channel

            if target_channel is None:
                # Join user's current voice channel
                if interaction.user.voice is None:
                    await interaction.followup.send(
                        "❌ You're not in a voice channel! "
                        "Either join one or specify a channel.",
                        ephemeral=True,
                    )
                    return

                target_channel = interaction.user.voice.channel

            # Check if already connected
            if interaction.guild.voice_client is not None:
                if interaction.guild.voice_client.channel.id == target_channel.id:
                    # Already in the channel - update agent
                    await self.bot.session_manager.set_agent(interaction.guild.id, agent)
                    await interaction.followup.send(
                        f"✅ Switched to **{agent.title()}** in {target_channel.mention}",
                        ephemeral=True,
                    )
                    return
                else:
                    # Move to new channel
                    await interaction.guild.voice_client.move_to(target_channel)
                    # Create session in new channel
                    await self.bot.on_voice_join(
                        interaction.guild,
                        target_channel,
                        interaction.guild.voice_client
                    )
                    # Set agent after session created
                    await self.bot.session_manager.set_agent(interaction.guild.id, agent)
                    await interaction.followup.send(
                        f"✅ **{agent.title()}** joined {target_channel.mention}"
                    )
                    return

            # Connect to channel using VoiceRecvClient for audio receiving
            connect_cls = voice_recv.VoiceRecvClient if HAS_VOICE_RECV else discord.VoiceClient
            voice_client = await target_channel.connect(
                cls=connect_cls,
                self_deaf=False,
                timeout=60.0
            )

            # Create session via bot handler
            await self.bot.on_voice_join(interaction.guild, target_channel, voice_client)

            # Set agent after session created
            await self.bot.session_manager.set_agent(interaction.guild.id, agent)

            personalities = {
                "jarvis": "🎩 Intelligent, witty, and sophisticated",
                "sage": "🧘 Wise, calm, and philosophical",
            }

            await interaction.followup.send(
                f"✅ **{agent.title()}** joined {target_channel.mention} and listening...\n"
                f"{personalities.get(agent, '')}"
            )

        except discord.errors.ClientException as e:
            logger.error(f"Failed to join voice channel: {e}")
            await interaction.followup.send(
                f"❌ Failed to join channel: {e}",
                ephemeral=True,
            )

        except Exception as e:
            logger.exception(f"Unexpected error in join command: {e}")
            await interaction.followup.send(
                "❌ An unexpected error occurred",
                ephemeral=True,
            )

    @app_commands.command(
        name="leave",
        description="Leave the current voice channel",
    )
    async def leave(self, interaction: discord.Interaction):
        """Leave voice channel."""
        await interaction.response.defer(thinking=True)

        try:
            if interaction.guild.voice_client is None:
                await interaction.followup.send(
                    "❌ Not in a voice channel",
                    ephemeral=True,
                )
                return

            # Disconnect via bot handler
            await self.bot.on_voice_leave(interaction.guild)

            await interaction.followup.send("👋 Left voice channel")

        except Exception as e:
            logger.exception(f"Error in leave command: {e}")
            await interaction.followup.send(
                "❌ An error occurred while leaving",
                ephemeral=True,
            )

    @app_commands.command(
        name="agent",
        description="Switch active AI agent",
    )
    @app_commands.describe(name="Agent to use (jarvis or sage)")
    @app_commands.choices(
        name=[
            app_commands.Choice(name="Jarvis", value="jarvis"),
            app_commands.Choice(name="Sage", value="sage"),
        ]
    )
    async def agent(self, interaction: discord.Interaction, name: str):
        """Switch active agent."""
        await interaction.response.defer(thinking=True)

        try:
            # Get session manager
            session_manager = self.bot.session_manager

            # Update agent
            success = await session_manager.set_agent(interaction.guild.id, name)

            if not success:
                await interaction.followup.send(
                    "❌ Not in a voice channel. Use `/jarvis join` first.",
                    ephemeral=True,
                )
                return

            # Get personality description
            personalities = {
                "jarvis": "🎩 Intelligent, witty, and sophisticated",
                "sage": "🧘 Wise, calm, and philosophical",
            }

            await interaction.followup.send(
                f"✅ Switched to **{name.title()}**\n"
                f"{personalities.get(name, '')}"
            )

        except Exception as e:
            logger.exception(f"Error in agent command: {e}")
            await interaction.followup.send(
                "❌ An error occurred",
                ephemeral=True,
            )

    @app_commands.command(
        name="sensitivity",
        description="Adjust how often the bot responds",
    )
    @app_commands.describe(level="Sensitivity level")
    @app_commands.choices(
        level=[
            app_commands.Choice(
                name="Low - Only when mentioned by name",
                value="low",
            ),
            app_commands.Choice(
                name="Medium - Name + relevant questions (recommended)",
                value="medium",
            ),
            app_commands.Choice(
                name="High - Responds more proactively",
                value="high",
            ),
        ]
    )
    async def sensitivity(self, interaction: discord.Interaction, level: str):
        """Set relevance sensitivity."""
        await interaction.response.defer(thinking=True)

        try:
            # Get session manager
            session_manager = self.bot.session_manager

            # Update sensitivity
            success = await session_manager.set_sensitivity(
                interaction.guild.id, level
            )

            if not success:
                await interaction.followup.send(
                    "❌ Not in a voice channel. Use `/jarvis join` first.",
                    ephemeral=True,
                )
                return

            descriptions = {
                "low": "Only responds when mentioned by name",
                "medium": "Responds to name mentions and relevant questions",
                "high": "Responds more proactively to conversations",
            }

            await interaction.followup.send(
                f"✅ Sensitivity set to **{level}**\n"
                f"{descriptions.get(level, '')}"
            )

        except Exception as e:
            logger.exception(f"Error in sensitivity command: {e}")
            await interaction.followup.send(
                "❌ An error occurred",
                ephemeral=True,
            )

    @app_commands.command(
        name="status",
        description="Show bot status and statistics",
    )
    async def status(self, interaction: discord.Interaction):
        """Show bot status."""
        await interaction.response.defer(thinking=True)

        try:
            session_manager = self.bot.session_manager
            session = session_manager.get_session(interaction.guild.id)

            if not session:
                await interaction.followup.send(
                    "❌ Not in a voice channel",
                    ephemeral=True,
                )
                return

            # Build status embed
            embed = discord.Embed(
                title="🤖 Jarvis Voice Bot Status",
                color=discord.Color.blue(),
            )

            # Session info
            embed.add_field(
                name="📊 Session",
                value=f"Channel: <#{session.channel_id}>\n"
                f"Duration: {session.duration:.0f}s\n"
                f"Active Users: {session.get_user_count()}",
                inline=True,
            )

            # Configuration
            embed.add_field(
                name="⚙️ Configuration",
                value=f"Agent: **{session.current_agent.title()}**\n"
                f"Sensitivity: **{session.sensitivity}**",
                inline=True,
            )

            # Global stats
            total_sessions = session_manager.get_session_count()
            embed.add_field(
                name="🌐 Global",
                value=f"Total Sessions: {total_sessions}",
                inline=True,
            )

            # TODO: Add latency stats when pipeline is implemented
            # embed.add_field(
            #     name="⚡ Performance",
            #     value=f"Avg Latency: X.XXs\n"
            #           f"Transcriptions: XX",
            #     inline=False,
            # )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.exception(f"Error in status command: {e}")
            await interaction.followup.send(
                "❌ An error occurred",
                ephemeral=True,
            )


class SageBotCommands(app_commands.Group):
    """Slash command group for Sage bot controls."""

    def __init__(self, bot):
        """Initialize command group."""
        super().__init__(name="sage", description="Sage Voice Bot commands")
        self.bot = bot
        self.agent_name = "sage"

    @app_commands.command(
        name="join",
        description="Join your voice channel as Sage",
    )
    @app_commands.describe(channel="Voice channel to join (optional)")
    async def join(
        self,
        interaction: discord.Interaction,
        channel: Optional[discord.VoiceChannel] = None,
    ):
        """Join a voice channel as Sage."""
        await self._join_with_agent(interaction, channel, self.agent_name)

    async def _join_with_agent(
        self,
        interaction: discord.Interaction,
        channel: Optional[discord.VoiceChannel],
        agent: str,
    ):
        """Join voice channel and set agent."""
        await interaction.response.defer(thinking=True)

        try:
            # Determine which channel to join
            target_channel = channel

            if target_channel is None:
                # Join user's current voice channel
                if interaction.user.voice is None:
                    await interaction.followup.send(
                        "❌ You're not in a voice channel! "
                        "Either join one or specify a channel.",
                        ephemeral=True,
                    )
                    return

                target_channel = interaction.user.voice.channel

            # Check if already connected
            if interaction.guild.voice_client is not None:
                if interaction.guild.voice_client.channel.id == target_channel.id:
                    # Already in the channel - update agent
                    await self.bot.session_manager.set_agent(interaction.guild.id, agent)
                    await interaction.followup.send(
                        f"✅ Switched to **{agent.title()}** in {target_channel.mention}",
                        ephemeral=True,
                    )
                    return
                else:
                    # Move to new channel
                    await interaction.guild.voice_client.move_to(target_channel)
                    # Create session in new channel with agent
                    await self.bot.on_voice_join(
                        interaction.guild,
                        target_channel,
                        interaction.guild.voice_client
                    )
                    # Set agent after session created
                    await self.bot.session_manager.set_agent(interaction.guild.id, agent)
                    await interaction.followup.send(
                        f"✅ **{agent.title()}** joined {target_channel.mention}"
                    )
                    return

            # Connect to channel using VoiceRecvClient for audio receiving
            connect_cls = voice_recv.VoiceRecvClient if HAS_VOICE_RECV else discord.VoiceClient
            voice_client = await target_channel.connect(
                cls=connect_cls,
                self_deaf=False,
                timeout=60.0
            )

            # Create session via bot handler
            await self.bot.on_voice_join(interaction.guild, target_channel, voice_client)

            # Set agent after session created
            await self.bot.session_manager.set_agent(interaction.guild.id, agent)

            personalities = {
                "jarvis": "🎩 Intelligent, witty, and sophisticated",
                "sage": "🧘 Wise, calm, and philosophical",
            }

            await interaction.followup.send(
                f"✅ **{agent.title()}** joined {target_channel.mention} and listening...\n"
                f"{personalities.get(agent, '')}"
            )

        except discord.errors.ClientException as e:
            logger.error(f"Failed to join voice channel: {e}")
            await interaction.followup.send(
                f"❌ Failed to join channel: {e}",
                ephemeral=True,
            )

        except Exception as e:
            logger.exception(f"Unexpected error in join command: {e}")
            await interaction.followup.send(
                "❌ An unexpected error occurred",
                ephemeral=True,
            )

    @app_commands.command(
        name="leave",
        description="Leave the current voice channel",
    )
    async def leave(self, interaction: discord.Interaction):
        """Leave voice channel."""
        await interaction.response.defer(thinking=True)

        try:
            if interaction.guild.voice_client is None:
                await interaction.followup.send(
                    "❌ Not in a voice channel",
                    ephemeral=True,
                )
                return

            # Disconnect via bot handler
            await self.bot.on_voice_leave(interaction.guild)

            await interaction.followup.send("👋 Sage left voice channel")

        except Exception as e:
            logger.exception(f"Error in leave command: {e}")
            await interaction.followup.send(
                "❌ An error occurred while leaving",
                ephemeral=True,
            )

    @app_commands.command(
        name="sensitivity",
        description="Adjust how often Sage responds",
    )
    @app_commands.describe(level="Sensitivity level")
    @app_commands.choices(
        level=[
            app_commands.Choice(
                name="Low - Only when mentioned by name",
                value="low",
            ),
            app_commands.Choice(
                name="Medium - Name + relevant questions (recommended)",
                value="medium",
            ),
            app_commands.Choice(
                name="High - Responds more proactively",
                value="high",
            ),
        ]
    )
    async def sensitivity(self, interaction: discord.Interaction, level: str):
        """Set relevance sensitivity."""
        await interaction.response.defer(thinking=True)

        try:
            # Get session manager
            session_manager = self.bot.session_manager

            # Update sensitivity
            success = await session_manager.set_sensitivity(
                interaction.guild.id, level
            )

            if not success:
                await interaction.followup.send(
                    "❌ Not in a voice channel. Use `/sage join` first.",
                    ephemeral=True,
                )
                return

            descriptions = {
                "low": "Only responds when mentioned by name",
                "medium": "Responds to name mentions and relevant questions",
                "high": "Responds more proactively to conversations",
            }

            await interaction.followup.send(
                f"✅ Sensitivity set to **{level}**\n"
                f"{descriptions.get(level, '')}"
            )

        except Exception as e:
            logger.exception(f"Error in sensitivity command: {e}")
            await interaction.followup.send(
                "❌ An error occurred",
                ephemeral=True,
            )

    @app_commands.command(
        name="status",
        description="Show Sage bot status and statistics",
    )
    async def status(self, interaction: discord.Interaction):
        """Show bot status."""
        await interaction.response.defer(thinking=True)

        try:
            session_manager = self.bot.session_manager
            session = session_manager.get_session(interaction.guild.id)

            if not session:
                await interaction.followup.send(
                    "❌ Not in a voice channel",
                    ephemeral=True,
                )
                return

            # Build status embed
            embed = discord.Embed(
                title="🧘 Sage Voice Bot Status",
                color=discord.Color.purple(),
            )

            # Session info
            embed.add_field(
                name="📊 Session",
                value=f"Channel: <#{session.channel_id}>\n"
                f"Duration: {session.duration:.0f}s\n"
                f"Active Users: {session.get_user_count()}",
                inline=True,
            )

            # Configuration
            embed.add_field(
                name="⚙️ Configuration",
                value=f"Agent: **{session.current_agent.title()}**\n"
                f"Sensitivity: **{session.sensitivity}**",
                inline=True,
            )

            # Global stats
            total_sessions = session_manager.get_session_count()
            embed.add_field(
                name="🌐 Global",
                value=f"Total Sessions: {total_sessions}",
                inline=True,
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.exception(f"Error in status command: {e}")
            await interaction.followup.send(
                "❌ An error occurred",
                ephemeral=True,
            )


async def setup_commands(bot):
    """
    Set up and register slash commands.

    Args:
        bot: Discord bot instance

    Returns:
        Tuple of command groups (jarvis, sage)
    """
    jarvis_commands = VoiceBotCommands(bot)
    sage_commands = SageBotCommands(bot)

    bot.tree.add_command(jarvis_commands)
    bot.tree.add_command(sage_commands)

    logger.info("Slash commands registered (jarvis, sage)")

    return jarvis_commands, sage_commands
