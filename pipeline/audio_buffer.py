"""Thread-safe ring buffer for per-user audio storage.

Stores recent audio for each user to support VAD and turn detection.
"""

import threading
from collections import deque
from typing import Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class AudioRingBuffer:
    """
    Thread-safe ring buffer for storing recent audio samples.

    Stores a fixed duration of audio (e.g., 10 seconds) and automatically
    discards older samples when the buffer is full.
    """

    def __init__(
        self,
        duration_seconds: float = 10.0,
        sample_rate: int = 16000,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize ring buffer.

        Args:
            duration_seconds: Maximum duration to store
            sample_rate: Audio sample rate (Hz)
            dtype: Data type of audio samples
        """
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.max_samples = int(duration_seconds * sample_rate)

        self._buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
        self._total_samples_written = 0

    def write(self, samples: np.ndarray) -> None:
        """
        Write audio samples to the buffer.

        Args:
            samples: Audio samples to write (1D array)
        """
        if samples.dtype != self.dtype:
            raise ValueError(
                f"Sample dtype {samples.dtype} doesn't match buffer dtype {self.dtype}"
            )

        if len(samples.shape) != 1:
            raise ValueError(f"Expected 1D array, got shape {samples.shape}")

        with self._lock:
            # Extend buffer (deque automatically removes old samples)
            self._buffer.extend(samples)
            self._total_samples_written += len(samples)

    def read(
        self, num_samples: Optional[int] = None, consume: bool = False
    ) -> np.ndarray:
        """
        Read audio samples from the buffer.

        Args:
            num_samples: Number of samples to read (None = all available)
            consume: If True, remove read samples from buffer

        Returns:
            Array of audio samples
        """
        with self._lock:
            if num_samples is None:
                num_samples = len(self._buffer)

            # Clamp to available samples
            num_samples = min(num_samples, len(self._buffer))

            if num_samples == 0:
                return np.array([], dtype=self.dtype)

            # Read samples
            if num_samples == len(self._buffer):
                # Read all
                samples = np.array(list(self._buffer), dtype=self.dtype)
            else:
                # Read last N samples
                samples = np.array(
                    list(self._buffer)[-num_samples:], dtype=self.dtype
                )

            # Optionally consume
            if consume:
                for _ in range(num_samples):
                    self._buffer.pop()

            return samples

    def read_time_range(
        self, start_seconds: float, end_seconds: float
    ) -> np.ndarray:
        """
        Read audio from a time range (relative to most recent sample).

        Args:
            start_seconds: Start time in seconds (0 = most recent)
            end_seconds: End time in seconds (positive = older audio)

        Returns:
            Array of audio samples in the time range

        Example:
            # Get last 2 seconds of audio
            audio = buffer.read_time_range(0, 2.0)

            # Get audio from 2-4 seconds ago
            audio = buffer.read_time_range(2.0, 4.0)
        """
        if start_seconds < 0 or end_seconds < start_seconds:
            raise ValueError("Invalid time range")

        start_samples = int(start_seconds * self.sample_rate)
        end_samples = int(end_seconds * self.sample_rate)

        with self._lock:
            total_available = len(self._buffer)

            # Clamp to available range
            start_idx = max(0, total_available - end_samples)
            end_idx = max(0, total_available - start_samples)

            if start_idx >= end_idx:
                return np.array([], dtype=self.dtype)

            # Extract range
            samples = np.array(
                list(self._buffer)[start_idx:end_idx], dtype=self.dtype
            )

            return samples

    def get_duration(self) -> float:
        """
        Get current duration of audio in buffer (seconds).

        Returns:
            Duration in seconds
        """
        with self._lock:
            return len(self._buffer) / self.sample_rate

    def get_sample_count(self) -> int:
        """
        Get number of samples currently in buffer.

        Returns:
            Sample count
        """
        with self._lock:
            return len(self._buffer)

    def get_total_written(self) -> int:
        """
        Get total number of samples written since creation.

        Returns:
            Total samples written
        """
        with self._lock:
            return self._total_samples_written

    def clear(self) -> None:
        """Clear all audio from the buffer."""
        with self._lock:
            self._buffer.clear()

    def is_full(self) -> bool:
        """
        Check if buffer is at maximum capacity.

        Returns:
            True if full, False otherwise
        """
        with self._lock:
            return len(self._buffer) >= self.max_samples

    def get_all(self) -> np.ndarray:
        """
        Get all audio currently in the buffer.

        Returns:
            Array of all audio samples
        """
        return self.read()

    def __len__(self) -> int:
        """Get number of samples in buffer."""
        return self.get_sample_count()

    def __repr__(self) -> str:
        """String representation."""
        duration = self.get_duration()
        return (
            f"AudioRingBuffer(duration={duration:.2f}s, "
            f"samples={self.get_sample_count()}, "
            f"max={self.max_samples})"
        )


class PerUserAudioBuffer:
    """
    Manages audio buffers for multiple users.

    Maintains separate ring buffers for each user in a voice channel.
    """

    def __init__(
        self,
        duration_seconds: float = 10.0,
        sample_rate: int = 16000,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize per-user buffer manager.

        Args:
            duration_seconds: Buffer duration per user
            sample_rate: Audio sample rate
            dtype: Audio data type
        """
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.dtype = dtype

        self._buffers: dict[int, AudioRingBuffer] = {}
        self._lock = threading.Lock()

    def get_or_create_buffer(self, user_id: int) -> AudioRingBuffer:
        """
        Get buffer for a user, creating if necessary.

        Args:
            user_id: User ID (Discord snowflake)

        Returns:
            AudioRingBuffer for the user
        """
        with self._lock:
            if user_id not in self._buffers:
                self._buffers[user_id] = AudioRingBuffer(
                    duration_seconds=self.duration_seconds,
                    sample_rate=self.sample_rate,
                    dtype=self.dtype,
                )
                logger.debug(f"Created audio buffer for user {user_id}")

            return self._buffers[user_id]

    def write(self, user_id: int, samples: np.ndarray) -> None:
        """
        Write audio samples for a user.

        Args:
            user_id: User ID
            samples: Audio samples
        """
        buffer = self.get_or_create_buffer(user_id)
        buffer.write(samples)

    def read(
        self, user_id: int, num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Read audio samples for a user.

        Args:
            user_id: User ID
            num_samples: Number of samples to read (None = all)

        Returns:
            Audio samples (empty array if user has no buffer)
        """
        with self._lock:
            if user_id not in self._buffers:
                return np.array([], dtype=self.dtype)

        return self._buffers[user_id].read(num_samples)

    def clear_user(self, user_id: int) -> None:
        """
        Clear audio buffer for a user.

        Args:
            user_id: User ID
        """
        with self._lock:
            if user_id in self._buffers:
                self._buffers[user_id].clear()

    def remove_user(self, user_id: int) -> None:
        """
        Remove user's buffer entirely.

        Args:
            user_id: User ID
        """
        with self._lock:
            if user_id in self._buffers:
                del self._buffers[user_id]
                logger.debug(f"Removed audio buffer for user {user_id}")

    def get_active_users(self) -> list[int]:
        """
        Get list of users with active buffers.

        Returns:
            List of user IDs
        """
        with self._lock:
            return list(self._buffers.keys())

    def get_user_count(self) -> int:
        """
        Get number of users with buffers.

        Returns:
            User count
        """
        with self._lock:
            return len(self._buffers)

    def clear_all(self) -> None:
        """Clear all user buffers."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()

    def remove_all(self) -> None:
        """Remove all user buffers."""
        with self._lock:
            self._buffers.clear()
            logger.debug("Removed all audio buffers")

    def get_status(self) -> dict[int, dict]:
        """
        Get status of all user buffers.

        Returns:
            Dict mapping user_id to buffer status
        """
        with self._lock:
            status = {}
            for user_id, buffer in self._buffers.items():
                status[user_id] = {
                    "duration": buffer.get_duration(),
                    "samples": buffer.get_sample_count(),
                    "total_written": buffer.get_total_written(),
                    "is_full": buffer.is_full(),
                }
            return status

    def __len__(self) -> int:
        """Get number of user buffers."""
        return self.get_user_count()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerUserAudioBuffer(users={self.get_user_count()}, "
            f"duration={self.duration_seconds}s)"
        )
