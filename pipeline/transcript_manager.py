"""Transcript management for rolling conversation context.

Maintains a sliding window of recent conversation for context in
relevance filtering and response generation.
"""

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptEntry:
    """A single entry in the conversation transcript."""

    speaker: str  # Display name (e.g., "Matt", "Jarvis")
    text: str  # What was said
    timestamp: datetime  # When it was said (UTC)
    user_id: Optional[int] = None  # Discord user ID (None for bot)

    @property
    def age_seconds(self) -> float:
        """Get age of this entry in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def format_time(self, format_str: str = "%I:%M:%S %p") -> str:
        """
        Format timestamp for display.

        Args:
            format_str: strftime format string

        Returns:
            Formatted time string
        """
        return self.timestamp.strftime(format_str)

    def format_compact(self) -> str:
        """
        Format entry in compact form for logging.

        Returns:
            Compact string: "[HH:MM:SS] Speaker: text"
        """
        return f"[{self.format_time('%H:%M:%S')}] {self.speaker}: {self.text}"

    def format_readable(self) -> str:
        """
        Format entry in human-readable form for LLM.

        Returns:
            Readable string: "[HH:MM:SS AM/PM] Speaker: text"
        """
        return f"[{self.format_time()}] {self.speaker}: {self.text}"


class TranscriptManager:
    """
    Manages rolling conversation transcript.

    Maintains a sliding window of recent conversation entries, automatically
    pruning old entries based on time and count limits.
    """

    def __init__(
        self,
        max_age_seconds: float = 90.0,
        max_entries: int = 20,
        timezone_offset: int = 0,
    ):
        """
        Initialize transcript manager.

        Args:
            max_age_seconds: Maximum age of entries (seconds)
            max_entries: Maximum number of entries to keep
            timezone_offset: Timezone offset from UTC (hours, for display)
        """
        self.max_age_seconds = max_age_seconds
        self.max_entries = max_entries
        self.timezone_offset = timezone_offset

        # Thread-safe deque for entries
        self._entries: deque[TranscriptEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

        # Stats
        self.total_entries_added = 0
        self.total_entries_pruned = 0

    def add_entry(
        self,
        speaker: str,
        text: str,
        user_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> TranscriptEntry:
        """
        Add an entry to the transcript.

        Args:
            speaker: Display name of speaker
            text: What was said
            user_id: Discord user ID (None for bot)
            timestamp: When it was said (defaults to now)

        Returns:
            The created TranscriptEntry
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Ensure timestamp is timezone-aware (UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            timestamp=timestamp,
            user_id=user_id,
        )

        with self._lock:
            self._entries.append(entry)
            self.total_entries_added += 1

            # Prune old entries
            self._prune_old_entries()

        logger.debug(f"Added transcript entry: {entry.format_compact()}")

        return entry

    def add_user_message(
        self, user_id: int, display_name: str, text: str
    ) -> TranscriptEntry:
        """
        Add a user message to the transcript.

        Args:
            user_id: Discord user ID
            display_name: User's display name
            text: Message text

        Returns:
            The created TranscriptEntry
        """
        return self.add_entry(
            speaker=display_name,
            text=text,
            user_id=user_id,
        )

    def add_bot_response(self, agent_name: str, text: str) -> TranscriptEntry:
        """
        Add a bot response to the transcript.

        Args:
            agent_name: Name of agent (e.g., "Jarvis", "Sage")
            text: Response text

        Returns:
            The created TranscriptEntry
        """
        return self.add_entry(
            speaker=agent_name,
            text=text,
            user_id=None,  # Bot has no user ID
        )

    def _prune_old_entries(self) -> int:
        """
        Remove entries that exceed age limit.

        Must be called with lock held.

        Returns:
            Number of entries pruned
        """
        pruned = 0
        current_time = datetime.now(timezone.utc)

        # Remove entries older than max_age_seconds
        while self._entries:
            oldest = self._entries[0]
            age = (current_time - oldest.timestamp).total_seconds()

            if age > self.max_age_seconds:
                self._entries.popleft()
                pruned += 1
                self.total_entries_pruned += 1
            else:
                break  # Entries are ordered, so we can stop

        if pruned > 0:
            logger.debug(f"Pruned {pruned} old transcript entries")

        return pruned

    def get_entries(
        self,
        max_age_seconds: Optional[float] = None,
        max_entries: Optional[int] = None,
    ) -> List[TranscriptEntry]:
        """
        Get transcript entries.

        Args:
            max_age_seconds: Override max age (None = use instance default)
            max_entries: Override max count (None = use instance default)

        Returns:
            List of transcript entries (oldest first)
        """
        with self._lock:
            # Prune first
            self._prune_old_entries()

            # Get all entries
            entries = list(self._entries)

        # Apply age filter if specified
        if max_age_seconds is not None:
            current_time = datetime.now(timezone.utc)
            entries = [
                e
                for e in entries
                if (current_time - e.timestamp).total_seconds() <= max_age_seconds
            ]

        # Apply count limit if specified
        if max_entries is not None and len(entries) > max_entries:
            entries = entries[-max_entries:]

        return entries

    def get_context(
        self,
        format: str = "readable",
        max_age_seconds: Optional[float] = None,
        max_entries: Optional[int] = None,
        include_timestamps: bool = True,
    ) -> str:
        """
        Get formatted transcript context.

        Args:
            format: Format type ("readable", "compact", "plain")
            max_age_seconds: Override max age
            max_entries: Override max count
            include_timestamps: Include timestamps in output

        Returns:
            Formatted transcript string
        """
        entries = self.get_entries(max_age_seconds, max_entries)

        if not entries:
            return ""

        # Format entries
        if format == "readable":
            lines = [e.format_readable() for e in entries]
        elif format == "compact":
            lines = [e.format_compact() for e in entries]
        elif format == "plain":
            if include_timestamps:
                lines = [f"[{e.format_time('%H:%M:%S')}] {e.text}" for e in entries]
            else:
                lines = [e.text for e in entries]
        else:
            raise ValueError(f"Unknown format: {format}")

        return "\n".join(lines)

    def get_recent_speakers(self, max_entries: int = 5) -> List[str]:
        """
        Get list of recent speakers (for context).

        Args:
            max_entries: How many recent entries to consider

        Returns:
            List of unique speaker names (most recent first)
        """
        entries = self.get_entries(max_entries=max_entries)

        # Get unique speakers in reverse order (most recent first)
        speakers = []
        seen = set()

        for entry in reversed(entries):
            if entry.speaker not in seen:
                speakers.append(entry.speaker)
                seen.add(entry.speaker)

        return speakers

    def get_last_speaker(self) -> Optional[str]:
        """
        Get the last speaker.

        Returns:
            Speaker name, or None if no entries
        """
        entries = self.get_entries(max_entries=1)
        return entries[0].speaker if entries else None

    def get_user_message_count(self, user_id: int) -> int:
        """
        Count messages from a specific user.

        Args:
            user_id: Discord user ID

        Returns:
            Number of messages from this user
        """
        entries = self.get_entries()
        return sum(1 for e in entries if e.user_id == user_id)

    def clear(self) -> None:
        """Clear all transcript entries."""
        with self._lock:
            pruned = len(self._entries)
            self._entries.clear()
            self.total_entries_pruned += pruned

        logger.info("Cleared all transcript entries")

    def get_stats(self) -> dict:
        """
        Get transcript statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            current_count = len(self._entries)
            oldest_age = (
                self._entries[0].age_seconds if self._entries else 0.0
            )

        return {
            "current_entries": current_count,
            "max_entries": self.max_entries,
            "max_age_seconds": self.max_age_seconds,
            "oldest_entry_age": oldest_age,
            "total_added": self.total_entries_added,
            "total_pruned": self.total_entries_pruned,
        }


class PerGuildTranscriptManager:
    """
    Manages separate transcripts for multiple Discord guilds.

    Each guild gets its own TranscriptManager instance.
    """

    def __init__(
        self,
        max_age_seconds: float = 90.0,
        max_entries: int = 20,
    ):
        """
        Initialize per-guild manager.

        Args:
            max_age_seconds: Default max age for all guilds
            max_entries: Default max entries for all guilds
        """
        self.max_age_seconds = max_age_seconds
        self.max_entries = max_entries

        # Per-guild managers
        self._managers: Dict[int, TranscriptManager] = {}
        self._lock = threading.Lock()

    def get_or_create(self, guild_id: int) -> TranscriptManager:
        """
        Get or create transcript manager for a guild.

        Args:
            guild_id: Discord guild ID

        Returns:
            TranscriptManager for this guild
        """
        with self._lock:
            if guild_id not in self._managers:
                self._managers[guild_id] = TranscriptManager(
                    max_age_seconds=self.max_age_seconds,
                    max_entries=self.max_entries,
                )
                logger.info(f"Created transcript manager for guild {guild_id}")

            return self._managers[guild_id]

    def add_entry(
        self,
        guild_id: int,
        speaker: str,
        text: str,
        user_id: Optional[int] = None,
    ) -> TranscriptEntry:
        """
        Add entry to a guild's transcript.

        Args:
            guild_id: Discord guild ID
            speaker: Display name
            text: Message text
            user_id: Discord user ID

        Returns:
            Created TranscriptEntry
        """
        manager = self.get_or_create(guild_id)
        return manager.add_entry(speaker, text, user_id)

    def get_context(
        self, guild_id: int, format: str = "readable"
    ) -> str:
        """
        Get formatted context for a guild.

        Args:
            guild_id: Discord guild ID
            format: Format type

        Returns:
            Formatted transcript
        """
        manager = self.get_or_create(guild_id)
        return manager.get_context(format=format)

    def clear_guild(self, guild_id: int) -> None:
        """
        Clear transcript for a guild.

        Args:
            guild_id: Discord guild ID
        """
        with self._lock:
            if guild_id in self._managers:
                self._managers[guild_id].clear()

    def remove_guild(self, guild_id: int) -> None:
        """
        Remove transcript manager for a guild.

        Args:
            guild_id: Discord guild ID
        """
        with self._lock:
            if guild_id in self._managers:
                del self._managers[guild_id]
                logger.info(f"Removed transcript manager for guild {guild_id}")

    def get_all_stats(self) -> Dict[int, dict]:
        """
        Get stats for all guilds.

        Returns:
            Dictionary mapping guild_id -> stats
        """
        with self._lock:
            return {
                guild_id: manager.get_stats()
                for guild_id, manager in self._managers.items()
            }


# Convenience function
def create_transcript_manager(
    max_age_seconds: float = 90.0,
    max_entries: int = 20,
) -> TranscriptManager:
    """
    Create a transcript manager with default settings.

    Args:
        max_age_seconds: Maximum age of entries
        max_entries: Maximum number of entries

    Returns:
        TranscriptManager instance
    """
    return TranscriptManager(
        max_age_seconds=max_age_seconds,
        max_entries=max_entries,
    )
