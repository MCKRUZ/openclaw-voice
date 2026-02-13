"""Unit tests for Discord bot components."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discord_bot.voice_session import VoiceSession, VoiceSessionManager
from utils.config import load_config


class TestVoiceSession:
    """Test VoiceSession class."""

    def test_create_session(self):
        """Test creating a voice session."""
        session = VoiceSession(
            guild_id=123456789,
            channel_id=987654321,
            voice_client=MagicMock(),
        )

        assert session.guild_id == 123456789
        assert session.channel_id == 987654321
        assert session.get_user_count() == 0
        assert session.current_agent == "jarvis"
        assert session.sensitivity == "medium"

    def test_add_remove_user(self):
        """Test adding and removing users."""
        session = VoiceSession(
            guild_id=123,
            channel_id=456,
            voice_client=MagicMock(),
        )

        # Add users
        session.add_user(111)
        assert session.get_user_count() == 1
        assert 111 in session.active_users

        session.add_user(222)
        assert session.get_user_count() == 2

        # Remove user
        session.remove_user(111)
        assert session.get_user_count() == 1
        assert 111 not in session.active_users
        assert 222 in session.active_users

    def test_is_empty(self):
        """Test empty check."""
        session = VoiceSession(
            guild_id=123,
            channel_id=456,
            voice_client=MagicMock(),
        )

        assert session.is_empty() is True

        session.add_user(111)
        assert session.is_empty() is False

        session.remove_user(111)
        assert session.is_empty() is True

    def test_duration(self):
        """Test session duration calculation."""
        import time

        session = VoiceSession(
            guild_id=123,
            channel_id=456,
            voice_client=MagicMock(),
        )

        time.sleep(0.1)
        assert session.duration >= 0.1


class TestVoiceSessionManager:
    """Test VoiceSessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session."""
        manager = VoiceSessionManager()

        voice_client = MagicMock()
        session = await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client,
            initial_users={111, 222},
        )

        assert session.guild_id == 123
        assert session.channel_id == 456
        assert session.get_user_count() == 2
        assert manager.has_session(123)
        assert manager.get_session_count() == 1

    @pytest.mark.asyncio
    async def test_remove_session(self):
        """Test removing a session."""
        manager = VoiceSessionManager()

        # Create mock voice client with async disconnect
        voice_client = MagicMock()
        voice_client.is_connected = MagicMock(return_value=True)
        voice_client.disconnect = AsyncMock()

        session = await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client,
        )

        await manager.remove_session(123)

        assert not manager.has_session(123)
        assert manager.get_session_count() == 0
        voice_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_users(self):
        """Test updating users in a session."""
        manager = VoiceSessionManager()

        voice_client = MagicMock()
        await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client,
            initial_users={111, 222},
        )

        # User 333 joins, user 111 leaves
        joined, left = await manager.update_users(123, {222, 333})

        assert joined == {333}
        assert left == {111}

        session = manager.get_session(123)
        assert session.active_users == {222, 333}

    @pytest.mark.asyncio
    async def test_set_agent(self):
        """Test setting agent for a session."""
        manager = VoiceSessionManager()

        voice_client = MagicMock()
        await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client,
        )

        success = await manager.set_agent(123, "sage")

        assert success is True

        session = manager.get_session(123)
        assert session.current_agent == "sage"

    @pytest.mark.asyncio
    async def test_set_sensitivity(self):
        """Test setting sensitivity for a session."""
        manager = VoiceSessionManager()

        voice_client = MagicMock()
        await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client,
        )

        success = await manager.set_sensitivity(123, "high")

        assert success is True

        session = manager.get_session(123)
        assert session.sensitivity == "high"

    @pytest.mark.asyncio
    async def test_cleanup_empty_sessions(self):
        """Test cleaning up empty sessions."""
        manager = VoiceSessionManager()

        # Create two sessions
        voice_client1 = MagicMock()
        voice_client1.is_connected = MagicMock(return_value=True)
        voice_client1.disconnect = AsyncMock()

        voice_client2 = MagicMock()
        voice_client2.is_connected = MagicMock(return_value=True)
        voice_client2.disconnect = AsyncMock()

        await manager.create_session(
            guild_id=123,
            channel_id=456,
            voice_client=voice_client1,
            initial_users=set(),  # Empty
        )

        await manager.create_session(
            guild_id=789,
            channel_id=456,
            voice_client=voice_client2,
            initial_users={111},  # Has user
        )

        # Cleanup should remove only the empty session
        removed = await manager.cleanup_empty_sessions()

        assert removed == 1
        assert not manager.has_session(123)
        assert manager.has_session(789)

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting all sessions."""
        manager = VoiceSessionManager()

        # Create multiple sessions
        for guild_id in [123, 456, 789]:
            voice_client = MagicMock()
            voice_client.is_connected = MagicMock(return_value=True)
            voice_client.disconnect = AsyncMock()

            await manager.create_session(
                guild_id=guild_id,
                channel_id=111,
                voice_client=voice_client,
            )

        assert manager.get_session_count() == 3

        await manager.disconnect_all()

        assert manager.get_session_count() == 0

    def test_get_status_summary(self):
        """Test getting status summary."""
        manager = VoiceSessionManager()

        # No sessions
        summary = manager.get_status_summary()
        assert "No active voice sessions" in summary


class TestBotInitialization:
    """Test bot initialization (without actually connecting)."""

    def test_create_bot(self):
        """Test creating bot instance."""
        config = load_config()

        # Import here to avoid issues
        from discord_bot.bot import JarvisVoiceBot

        bot = JarvisVoiceBot(config)

        assert bot.config == config
        assert bot.session_manager is not None
        assert bot.audio_bridge is None  # Not initialized until setup_hook

    @pytest.mark.asyncio
    async def test_bot_setup_hook(self):
        """Test bot setup hook."""
        config = load_config()

        from discord_bot.bot import JarvisVoiceBot

        bot = JarvisVoiceBot(config)

        # Mock the cleanup task
        with patch.object(bot.cleanup_task, "start") as mock_start:
            await bot.setup_hook()

            # Audio bridge should be initialized
            assert bot.audio_bridge is not None

            # Cleanup task should be started
            mock_start.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
