"""Unit tests for audio buffer."""

import numpy as np
import pytest

from pipeline.audio_buffer import AudioRingBuffer, PerUserAudioBuffer


class TestAudioRingBuffer:
    """Test AudioRingBuffer class."""

    def test_create_buffer(self):
        """Test creating a buffer."""
        buffer = AudioRingBuffer(
            duration_seconds=2.0,
            sample_rate=16000,
            dtype=np.float32,
        )

        assert buffer.duration_seconds == 2.0
        assert buffer.sample_rate == 16000
        assert buffer.max_samples == 32000  # 2.0 * 16000
        assert buffer.get_sample_count() == 0
        assert buffer.get_duration() == 0.0

    def test_write_samples(self):
        """Test writing audio samples."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        samples = np.random.randn(1000).astype(np.float32)
        buffer.write(samples)

        assert buffer.get_sample_count() == 1000
        assert abs(buffer.get_duration() - 0.0625) < 0.001  # 1000/16000

    def test_write_exceeds_capacity(self):
        """Test writing more samples than buffer capacity."""
        buffer = AudioRingBuffer(duration_seconds=0.1, sample_rate=16000)

        # Write 0.2 seconds (should keep only last 0.1 seconds)
        samples = np.random.randn(3200).astype(np.float32)
        buffer.write(samples)

        # Should have discarded oldest samples
        assert buffer.get_sample_count() == 1600  # 0.1 * 16000
        assert buffer.is_full()

    def test_read_all_samples(self):
        """Test reading all samples."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        # Write known samples
        samples = np.arange(1000, dtype=np.float32)
        buffer.write(samples)

        # Read all
        read_samples = buffer.read()

        assert len(read_samples) == 1000
        assert np.array_equal(read_samples, samples)

    def test_read_partial_samples(self):
        """Test reading partial samples."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        samples = np.arange(1000, dtype=np.float32)
        buffer.write(samples)

        # Read last 100 samples
        read_samples = buffer.read(num_samples=100)

        assert len(read_samples) == 100
        assert np.array_equal(read_samples, samples[-100:])

    def test_read_consume(self):
        """Test reading with consume flag."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        samples = np.arange(1000, dtype=np.float32)
        buffer.write(samples)

        # Read and consume 500 samples
        read_samples = buffer.read(num_samples=500, consume=True)

        assert len(read_samples) == 500
        assert buffer.get_sample_count() == 500  # 500 consumed

    def test_read_time_range(self):
        """Test reading a time range."""
        buffer = AudioRingBuffer(duration_seconds=2.0, sample_rate=16000)

        # Write 2 seconds of audio
        samples = np.arange(32000, dtype=np.float32)
        buffer.write(samples)

        # Read last 0.5 seconds (0 to 0.5 seconds ago)
        time_range = buffer.read_time_range(0.0, 0.5)

        expected_samples = 8000  # 0.5 * 16000
        assert len(time_range) == expected_samples
        assert np.array_equal(time_range, samples[-expected_samples:])

    def test_read_time_range_middle(self):
        """Test reading middle time range."""
        buffer = AudioRingBuffer(duration_seconds=2.0, sample_rate=16000)

        samples = np.arange(32000, dtype=np.float32)
        buffer.write(samples)

        # Read 0.5-1.0 seconds ago
        time_range = buffer.read_time_range(0.5, 1.0)

        start_idx = 32000 - int(1.0 * 16000)  # 1 second ago
        end_idx = 32000 - int(0.5 * 16000)  # 0.5 seconds ago

        assert len(time_range) == 8000
        assert np.array_equal(time_range, samples[start_idx:end_idx])

    def test_clear(self):
        """Test clearing buffer."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        samples = np.random.randn(1000).astype(np.float32)
        buffer.write(samples)

        buffer.clear()

        assert buffer.get_sample_count() == 0
        assert buffer.get_duration() == 0.0

    def test_is_full(self):
        """Test full check."""
        buffer = AudioRingBuffer(duration_seconds=0.1, sample_rate=16000)

        assert not buffer.is_full()

        # Fill buffer
        samples = np.random.randn(1600).astype(np.float32)
        buffer.write(samples)

        assert buffer.is_full()

    def test_total_written_tracking(self):
        """Test tracking total samples written."""
        buffer = AudioRingBuffer(duration_seconds=0.1, sample_rate=16000)

        # Write 1000 samples
        buffer.write(np.random.randn(1000).astype(np.float32))
        assert buffer.get_total_written() == 1000

        # Write 1000 more
        buffer.write(np.random.randn(1000).astype(np.float32))
        assert buffer.get_total_written() == 2000

        # Clear doesn't reset total written
        buffer.clear()
        assert buffer.get_total_written() == 2000

    def test_wrong_dtype(self):
        """Test that wrong dtype raises error."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000, dtype=np.float32)

        with pytest.raises(ValueError):
            buffer.write(np.array([1, 2, 3], dtype=np.int16))

    def test_wrong_shape(self):
        """Test that 2D array raises error."""
        buffer = AudioRingBuffer(duration_seconds=1.0, sample_rate=16000)

        with pytest.raises(ValueError):
            buffer.write(np.random.randn(100, 2).astype(np.float32))


class TestPerUserAudioBuffer:
    """Test PerUserAudioBuffer class."""

    def test_create_manager(self):
        """Test creating buffer manager."""
        manager = PerUserAudioBuffer(
            duration_seconds=5.0,
            sample_rate=16000,
        )

        assert manager.duration_seconds == 5.0
        assert manager.sample_rate == 16000
        assert manager.get_user_count() == 0

    def test_get_or_create_buffer(self):
        """Test getting/creating user buffer."""
        manager = PerUserAudioBuffer()

        buffer = manager.get_or_create_buffer(user_id=123)

        assert isinstance(buffer, AudioRingBuffer)
        assert manager.get_user_count() == 1

        # Getting again returns same buffer
        buffer2 = manager.get_or_create_buffer(user_id=123)
        assert buffer is buffer2

    def test_write_for_user(self):
        """Test writing audio for a user."""
        manager = PerUserAudioBuffer()

        samples = np.random.randn(1000).astype(np.float32)
        manager.write(user_id=123, samples=samples)

        assert manager.get_user_count() == 1

        # Read back
        read_samples = manager.read(user_id=123)
        assert np.array_equal(read_samples, samples)

    def test_multiple_users(self):
        """Test managing multiple users."""
        manager = PerUserAudioBuffer()

        # Write for user 1
        samples1 = np.ones(500, dtype=np.float32)
        manager.write(user_id=1, samples=samples1)

        # Write for user 2
        samples2 = np.ones(500, dtype=np.float32) * 2
        manager.write(user_id=2, samples=samples2)

        assert manager.get_user_count() == 2
        assert 1 in manager.get_active_users()
        assert 2 in manager.get_active_users()

        # Read back (should be independent)
        assert np.array_equal(manager.read(user_id=1), samples1)
        assert np.array_equal(manager.read(user_id=2), samples2)

    def test_clear_user(self):
        """Test clearing user buffer."""
        manager = PerUserAudioBuffer()

        manager.write(user_id=123, samples=np.random.randn(1000).astype(np.float32))
        manager.clear_user(user_id=123)

        # Buffer still exists but is empty
        assert manager.get_user_count() == 1
        assert len(manager.read(user_id=123)) == 0

    def test_remove_user(self):
        """Test removing user buffer."""
        manager = PerUserAudioBuffer()

        manager.write(user_id=123, samples=np.random.randn(1000).astype(np.float32))
        manager.remove_user(user_id=123)

        # Buffer removed entirely
        assert manager.get_user_count() == 0
        assert 123 not in manager.get_active_users()

    def test_read_nonexistent_user(self):
        """Test reading from user with no buffer."""
        manager = PerUserAudioBuffer()

        # Should return empty array, not error
        samples = manager.read(user_id=999)

        assert len(samples) == 0
        assert samples.dtype == np.float32

    def test_clear_all(self):
        """Test clearing all buffers."""
        manager = PerUserAudioBuffer()

        # Create buffers for multiple users
        for user_id in [1, 2, 3]:
            manager.write(user_id=user_id, samples=np.random.randn(100).astype(np.float32))

        manager.clear_all()

        # Buffers still exist but are empty
        assert manager.get_user_count() == 3
        for user_id in [1, 2, 3]:
            assert len(manager.read(user_id=user_id)) == 0

    def test_remove_all(self):
        """Test removing all buffers."""
        manager = PerUserAudioBuffer()

        # Create buffers
        for user_id in [1, 2, 3]:
            manager.write(user_id=user_id, samples=np.random.randn(100).astype(np.float32))

        manager.remove_all()

        # All buffers removed
        assert manager.get_user_count() == 0

    def test_get_status(self):
        """Test getting status of all buffers."""
        manager = PerUserAudioBuffer(duration_seconds=1.0, sample_rate=16000)

        # Create some buffers
        manager.write(user_id=1, samples=np.random.randn(500).astype(np.float32))
        manager.write(user_id=2, samples=np.random.randn(1000).astype(np.float32))

        status = manager.get_status()

        assert 1 in status
        assert 2 in status
        assert status[1]["samples"] == 500
        assert status[2]["samples"] == 1000
        assert "duration" in status[1]
        assert "is_full" in status[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
