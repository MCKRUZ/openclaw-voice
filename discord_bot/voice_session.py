"""Voice session manager for Discord guilds.

Manages per-guild voice connections and tracks active users.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set

import discord

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VoiceSession:
    """Represents an active voice session in a Discord guild."""

    guild_id: int
    channel_id: int
    voice_client: discord.VoiceClient
    active_users: Set[int] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    current_agent: str = "jarvis"
    sensitivity: str = "medium"

    def add_user(self, user_id: int) -> None:
        """Add a user to the active users set."""
        self.active_users.add(user_id)
        logger.info(
            f"User {user_id} joined voice session in guild {self.guild_id}. "
            f"Active users: {len(self.active_users)}"
        )

    def remove_user(self, user_id: int) -> None:
        """Remove a user from the active users set."""
        self.active_users.discard(user_id)
        logger.info(
            f"User {user_id} left voice session in guild {self.guild_id}. "
            f"Active users: {len(self.active_users)}"
        )

    def is_empty(self) -> bool:
        """Check if no users are in the voice channel."""
        return len(self.active_users) == 0

    def get_user_count(self) -> int:
        """Get the number of active users."""
        return len(self.active_users)

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


class VoiceSessionManager:
    """Manages voice sessions across multiple Discord guilds."""

    def __init__(self):
        self._sessions: Dict[int, VoiceSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        guild_id: int,
        channel_id: int,
        voice_client: discord.VoiceClient,
        initial_users: Optional[Set[int]] = None,
    ) -> VoiceSession:
        """
        Create a new voice session.

        Args:
            guild_id: Discord guild ID
            channel_id: Voice channel ID
            voice_client: Connected voice client
            initial_users: Set of user IDs already in channel

        Returns:
            Created VoiceSession
        """
        async with self._lock:
            if guild_id in self._sessions:
                logger.warning(
                    f"Session already exists for guild {guild_id}, replacing"
                )
                await self.remove_session(guild_id)

            session = VoiceSession(
                guild_id=guild_id,
                channel_id=channel_id,
                voice_client=voice_client,
                active_users=initial_users or set(),
            )

            self._sessions[guild_id] = session

            logger.info(
                f"Created voice session for guild {guild_id}, "
                f"channel {channel_id} with {len(session.active_users)} users"
            )

            return session

    async def remove_session(self, guild_id: int) -> None:
        """
        Remove and cleanup a voice session.

        Args:
            guild_id: Discord guild ID
        """
        async with self._lock:
            session = self._sessions.pop(guild_id, None)

            if session:
                # Disconnect voice client if still connected
                if session.voice_client and session.voice_client.is_connected():
                    try:
                        await session.voice_client.disconnect(force=False)
                    except Exception as e:
                        logger.error(f"Error disconnecting voice client: {e}")

                logger.info(
                    f"Removed voice session for guild {guild_id} "
                    f"(duration: {session.duration:.1f}s)"
                )

    def get_session(self, guild_id: int) -> Optional[VoiceSession]:
        """
        Get voice session for a guild.

        Args:
            guild_id: Discord guild ID

        Returns:
            VoiceSession if exists, None otherwise
        """
        return self._sessions.get(guild_id)

    def has_session(self, guild_id: int) -> bool:
        """Check if guild has an active session."""
        return guild_id in self._sessions

    def get_all_sessions(self) -> list[VoiceSession]:
        """Get all active sessions."""
        return list(self._sessions.values())

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    async def update_users(
        self, guild_id: int, current_users: Set[int]
    ) -> tuple[Set[int], Set[int]]:
        """
        Update users in a session and return changes.

        Args:
            guild_id: Discord guild ID
            current_users: Current set of user IDs in channel

        Returns:
            Tuple of (joined_users, left_users)
        """
        session = self.get_session(guild_id)
        if not session:
            logger.warning(f"No session found for guild {guild_id}")
            return set(), set()

        # Calculate changes
        joined_users = current_users - session.active_users
        left_users = session.active_users - current_users

        # Update session
        for user_id in joined_users:
            session.add_user(user_id)

        for user_id in left_users:
            session.remove_user(user_id)

        return joined_users, left_users

    async def set_agent(self, guild_id: int, agent: str) -> bool:
        """
        Set the active agent for a guild session.

        Args:
            guild_id: Discord guild ID
            agent: Agent name (jarvis or sage)

        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(guild_id)
        if not session:
            return False

        old_agent = session.current_agent
        session.current_agent = agent

        logger.info(
            f"Guild {guild_id} switched agent from {old_agent} to {agent}"
        )

        return True

    async def set_sensitivity(self, guild_id: int, sensitivity: str) -> bool:
        """
        Set the relevance sensitivity for a guild session.

        Args:
            guild_id: Discord guild ID
            sensitivity: Sensitivity level (low, medium, high)

        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(guild_id)
        if not session:
            return False

        old_sensitivity = session.sensitivity
        session.sensitivity = sensitivity

        logger.info(
            f"Guild {guild_id} changed sensitivity from "
            f"{old_sensitivity} to {sensitivity}"
        )

        return True

    async def cleanup_empty_sessions(self) -> int:
        """
        Remove sessions with no active users.

        Returns:
            Number of sessions removed
        """
        to_remove = []

        for guild_id, session in self._sessions.items():
            if session.is_empty():
                to_remove.append(guild_id)

        for guild_id in to_remove:
            await self.remove_session(guild_id)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} empty sessions")

        return len(to_remove)

    async def disconnect_all(self) -> None:
        """Disconnect all voice sessions (for shutdown)."""
        logger.info(f"Disconnecting all {self.get_session_count()} sessions")

        guild_ids = list(self._sessions.keys())
        for guild_id in guild_ids:
            await self.remove_session(guild_id)

    def get_status_summary(self) -> str:
        """
        Get a summary of all active sessions.

        Returns:
            Formatted status string
        """
        if not self._sessions:
            return "No active voice sessions"

        lines = [f"Active Sessions: {self.get_session_count()}"]

        for session in self._sessions.values():
            lines.append(
                f"  Guild {session.guild_id}: "
                f"{session.get_user_count()} users, "
                f"agent={session.current_agent}, "
                f"sensitivity={session.sensitivity}, "
                f"duration={session.duration:.0f}s"
            )

        return "\n".join(lines)
