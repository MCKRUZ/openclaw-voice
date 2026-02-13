"""Unit tests for Transcript Manager."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.transcript_manager import (
    PerGuildTranscriptManager,
    TranscriptEntry,
    TranscriptManager,
    create_transcript_manager,
)


class TestTranscriptEntry:
    """Test TranscriptEntry dataclass."""

    def test_create_entry(self):
        """Test creating a transcript entry."""
        timestamp = datetime.now(timezone.utc)

        entry = TranscriptEntry(
            speaker="Matt",
            text="Hello world",
            timestamp=timestamp,
            user_id=123,
        )

        assert entry.speaker == "Matt"
        assert entry.text == "Hello world"
        assert entry.timestamp == timestamp
        assert entry.user_id == 123

    def test_create_entry_without_user_id(self):
        """Test creating bot entry (no user ID)."""
        entry = TranscriptEntry(
            speaker="Jarvis",
            text="Hello",
            timestamp=datetime.now(timezone.utc),
        )

        assert entry.speaker == "Jarvis"
        assert entry.user_id is None

    def test_age_seconds(self):
        """Test age calculation."""
        # Create entry 5 seconds ago
        timestamp = datetime.now(timezone.utc) - timedelta(seconds=5)

        entry = TranscriptEntry(
            speaker="Test",
            text="Test",
            timestamp=timestamp,
        )

        # Age should be approximately 5 seconds
        assert 4.5 <= entry.age_seconds <= 5.5

    def test_format_time(self):
        """Test time formatting."""
        timestamp = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)

        entry = TranscriptEntry(
            speaker="Test",
            text="Test",
            timestamp=timestamp,
        )

        # Default format (12-hour with AM/PM)
        formatted = entry.format_time()
        assert "02:30:45 PM" in formatted

        # Custom format (24-hour)
        formatted = entry.format_time("%H:%M:%S")
        assert formatted == "14:30:45"

    def test_format_compact(self):
        """Test compact formatting."""
        timestamp = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)

        entry = TranscriptEntry(
            speaker="Matt",
            text="Hello world",
            timestamp=timestamp,
        )

        formatted = entry.format_compact()

        assert "[14:30:45]" in formatted
        assert "Matt:" in formatted
        assert "Hello world" in formatted

    def test_format_readable(self):
        """Test readable formatting."""
        timestamp = datetime(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)

        entry = TranscriptEntry(
            speaker="Jake",
            text="How are you?",
            timestamp=timestamp,
        )

        formatted = entry.format_readable()

        assert "02:30:45 PM" in formatted
        assert "Jake:" in formatted
        assert "How are you?" in formatted


class TestTranscriptManager:
    """Test TranscriptManager class."""

    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return TranscriptManager(
            max_age_seconds=10.0,  # Short for testing
            max_entries=5,
        )

    def test_create_manager(self, manager):
        """Test creating manager."""
        assert manager.max_age_seconds == 10.0
        assert manager.max_entries == 5
        assert manager.total_entries_added == 0
        assert manager.total_entries_pruned == 0

    def test_add_entry(self, manager):
        """Test adding an entry."""
        entry = manager.add_entry(
            speaker="Matt",
            text="Hello",
            user_id=123,
        )

        assert isinstance(entry, TranscriptEntry)
        assert entry.speaker == "Matt"
        assert entry.text == "Hello"
        assert entry.user_id == 123
        assert manager.total_entries_added == 1

    def test_add_user_message(self, manager):
        """Test adding user message."""
        entry = manager.add_user_message(
            user_id=456,
            display_name="Jake",
            text="How are you?",
        )

        assert entry.speaker == "Jake"
        assert entry.text == "How are you?"
        assert entry.user_id == 456

    def test_add_bot_response(self, manager):
        """Test adding bot response."""
        entry = manager.add_bot_response(
            agent_name="Jarvis",
            text="I'm doing well, thank you!",
        )

        assert entry.speaker == "Jarvis"
        assert entry.text == "I'm doing well, thank you!"
        assert entry.user_id is None

    def test_get_entries(self, manager):
        """Test getting entries."""
        # Add some entries
        manager.add_entry("Matt", "First", 1)
        manager.add_entry("Jake", "Second", 2)
        manager.add_entry("Jarvis", "Third", None)

        entries = manager.get_entries()

        assert len(entries) == 3
        assert entries[0].speaker == "Matt"
        assert entries[1].speaker == "Jake"
        assert entries[2].speaker == "Jarvis"

    def test_max_entries_limit(self, manager):
        """Test max entries limit."""
        # Add more than max_entries
        for i in range(10):
            manager.add_entry(f"User{i}", f"Message {i}", i)

        entries = manager.get_entries()

        # Should only keep last 5 (max_entries)
        assert len(entries) == 5
        assert entries[-1].text == "Message 9"

    def test_age_based_pruning(self, manager):
        """Test age-based pruning."""
        # Add entry with old timestamp
        old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=15)
        manager.add_entry("Old", "Old message", 1, timestamp=old_timestamp)

        # Add recent entry
        manager.add_entry("Recent", "Recent message", 2)

        # Get entries (should prune old one)
        entries = manager.get_entries()

        assert len(entries) == 1
        assert entries[0].speaker == "Recent"

    def test_get_entries_with_max_age_override(self, manager):
        """Test getting entries with age override."""
        # Add entries at different times
        old_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        manager.add_entry("Old", "Old", 1, timestamp=old_time)
        manager.add_entry("Recent", "Recent", 2)

        # Get with very short max age
        entries = manager.get_entries(max_age_seconds=3.0)

        # Should only return recent one
        assert len(entries) == 1
        assert entries[0].speaker == "Recent"

    def test_get_entries_with_max_entries_override(self, manager):
        """Test getting entries with count override."""
        # Add 5 entries
        for i in range(5):
            manager.add_entry(f"User{i}", f"Msg {i}", i)

        # Get only last 2
        entries = manager.get_entries(max_entries=2)

        assert len(entries) == 2
        assert entries[0].text == "Msg 3"
        assert entries[1].text == "Msg 4"

    def test_get_context_readable(self, manager):
        """Test readable context formatting."""
        manager.add_entry("Matt", "Hey there", 1)
        manager.add_entry("Jarvis", "Hello Matt", None)

        context = manager.get_context(format="readable")

        assert "Matt: Hey there" in context
        assert "Jarvis: Hello Matt" in context
        assert "PM" in context or "AM" in context  # Has time

    def test_get_context_compact(self, manager):
        """Test compact context formatting."""
        manager.add_entry("Jake", "Test message", 2)

        context = manager.get_context(format="compact")

        assert "Jake: Test message" in context
        assert "[" in context  # Has timestamp

    def test_get_context_plain(self, manager):
        """Test plain context formatting."""
        manager.add_entry("User", "Plain text", 1)

        # With timestamps
        context = manager.get_context(format="plain", include_timestamps=True)
        assert "Plain text" in context
        assert "[" in context

        # Without timestamps
        context = manager.get_context(format="plain", include_timestamps=False)
        assert context == "Plain text"

    def test_get_context_empty(self, manager):
        """Test getting context when empty."""
        context = manager.get_context()
        assert context == ""

    def test_get_context_invalid_format(self, manager):
        """Test getting context with invalid format."""
        manager.add_entry("Test", "Test", 1)

        with pytest.raises(ValueError) as exc:
            manager.get_context(format="invalid")

        assert "Unknown format" in str(exc.value)

    def test_get_recent_speakers(self, manager):
        """Test getting recent speakers."""
        manager.add_entry("Matt", "First", 1)
        manager.add_entry("Jake", "Second", 2)
        manager.add_entry("Matt", "Third", 1)  # Matt again
        manager.add_entry("Jarvis", "Fourth", None)

        speakers = manager.get_recent_speakers(max_entries=5)

        # Should be unique, most recent first
        assert speakers == ["Jarvis", "Matt", "Jake"]

    def test_get_recent_speakers_limited(self, manager):
        """Test getting recent speakers with limit."""
        for i in range(5):
            manager.add_entry(f"User{i}", "Msg", i)

        speakers = manager.get_recent_speakers(max_entries=3)

        # Should only consider last 3 entries
        assert len(speakers) == 3
        assert speakers[0] == "User4"  # Most recent

    def test_get_last_speaker(self, manager):
        """Test getting last speaker."""
        manager.add_entry("Matt", "First", 1)
        manager.add_entry("Jake", "Second", 2)

        assert manager.get_last_speaker() == "Jake"

    def test_get_last_speaker_empty(self, manager):
        """Test getting last speaker when empty."""
        assert manager.get_last_speaker() is None

    def test_get_user_message_count(self, manager):
        """Test counting user messages."""
        manager.add_entry("Matt", "First", 123)
        manager.add_entry("Jake", "Second", 456)
        manager.add_entry("Matt", "Third", 123)
        manager.add_entry("Jarvis", "Bot", None)

        count = manager.get_user_message_count(123)
        assert count == 2

        count = manager.get_user_message_count(456)
        assert count == 1

        count = manager.get_user_message_count(999)
        assert count == 0

    def test_clear(self, manager):
        """Test clearing transcript."""
        # Add entries
        manager.add_entry("Matt", "Test 1", 1)
        manager.add_entry("Jake", "Test 2", 2)

        assert len(manager.get_entries()) == 2

        # Clear
        manager.clear()

        assert len(manager.get_entries()) == 0

    def test_get_stats(self, manager):
        """Test getting statistics."""
        # Add some entries
        manager.add_entry("User1", "Msg1", 1)
        manager.add_entry("User2", "Msg2", 2)

        stats = manager.get_stats()

        assert stats["current_entries"] == 2
        assert stats["max_entries"] == 5
        assert stats["max_age_seconds"] == 10.0
        assert stats["total_added"] == 2
        assert stats["oldest_entry_age"] >= 0

    def test_get_stats_empty(self, manager):
        """Test stats when empty."""
        stats = manager.get_stats()

        assert stats["current_entries"] == 0
        assert stats["oldest_entry_age"] == 0.0

    def test_timestamp_timezone_naive(self, manager):
        """Test that naive timestamps are converted to UTC."""
        # Create naive timestamp
        naive_time = datetime(2024, 1, 15, 12, 0, 0)

        entry = manager.add_entry("Test", "Test", 1, timestamp=naive_time)

        # Should have timezone set to UTC
        assert entry.timestamp.tzinfo == timezone.utc


class TestPerGuildTranscriptManager:
    """Test PerGuildTranscriptManager class."""

    @pytest.fixture
    def manager(self):
        """Create per-guild manager."""
        return PerGuildTranscriptManager(
            max_age_seconds=10.0,
            max_entries=5,
        )

    def test_create_manager(self, manager):
        """Test creating per-guild manager."""
        assert manager.max_age_seconds == 10.0
        assert manager.max_entries == 5

    def test_get_or_create(self, manager):
        """Test getting or creating guild manager."""
        guild_manager = manager.get_or_create(guild_id=123)

        assert isinstance(guild_manager, TranscriptManager)
        assert guild_manager.max_age_seconds == 10.0
        assert guild_manager.max_entries == 5

        # Getting again should return same instance
        guild_manager2 = manager.get_or_create(guild_id=123)
        assert guild_manager is guild_manager2

    def test_multiple_guilds(self, manager):
        """Test managing multiple guilds."""
        guild1 = manager.get_or_create(guild_id=111)
        guild2 = manager.get_or_create(guild_id=222)

        # Should be different instances
        assert guild1 is not guild2

        # Add entries to each
        guild1.add_entry("User1", "Guild 1 message", 1)
        guild2.add_entry("User2", "Guild 2 message", 2)

        # Should be independent
        assert len(guild1.get_entries()) == 1
        assert len(guild2.get_entries()) == 1
        assert guild1.get_entries()[0].text == "Guild 1 message"
        assert guild2.get_entries()[0].text == "Guild 2 message"

    def test_add_entry(self, manager):
        """Test adding entry via per-guild manager."""
        entry = manager.add_entry(
            guild_id=123,
            speaker="Matt",
            text="Test message",
            user_id=456,
        )

        assert entry.speaker == "Matt"
        assert entry.text == "Test message"

        # Verify it was added to correct guild
        guild_manager = manager.get_or_create(guild_id=123)
        entries = guild_manager.get_entries()
        assert len(entries) == 1

    def test_get_context(self, manager):
        """Test getting context for a guild."""
        manager.add_entry(123, "Matt", "Hello", 1)
        manager.add_entry(123, "Jarvis", "Hi Matt", None)

        context = manager.get_context(guild_id=123, format="readable")

        assert "Matt: Hello" in context
        assert "Jarvis: Hi Matt" in context

    def test_clear_guild(self, manager):
        """Test clearing a guild's transcript."""
        # Add to two guilds
        manager.add_entry(111, "User1", "Guild 1", 1)
        manager.add_entry(222, "User2", "Guild 2", 2)

        # Clear guild 111
        manager.clear_guild(guild_id=111)

        # Guild 111 should be empty
        guild1 = manager.get_or_create(guild_id=111)
        assert len(guild1.get_entries()) == 0

        # Guild 222 should still have entry
        guild2 = manager.get_or_create(guild_id=222)
        assert len(guild2.get_entries()) == 1

    def test_remove_guild(self, manager):
        """Test removing a guild's manager."""
        # Create guild manager
        manager.get_or_create(guild_id=123)
        assert 123 in manager._managers

        # Remove it
        manager.remove_guild(guild_id=123)
        assert 123 not in manager._managers

    def test_remove_nonexistent_guild(self, manager):
        """Test removing guild that doesn't exist."""
        # Should not raise error
        manager.remove_guild(guild_id=999)

    def test_get_all_stats(self, manager):
        """Test getting stats for all guilds."""
        # Add entries to two guilds
        manager.add_entry(111, "User1", "Msg1", 1)
        manager.add_entry(222, "User2", "Msg2", 2)
        manager.add_entry(222, "User3", "Msg3", 3)

        all_stats = manager.get_all_stats()

        assert 111 in all_stats
        assert 222 in all_stats
        assert all_stats[111]["current_entries"] == 1
        assert all_stats[222]["current_entries"] == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_transcript_manager(self):
        """Test creating manager with convenience function."""
        manager = create_transcript_manager(
            max_age_seconds=60.0,
            max_entries=10,
        )

        assert isinstance(manager, TranscriptManager)
        assert manager.max_age_seconds == 60.0
        assert manager.max_entries == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
