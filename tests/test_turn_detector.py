"""Unit tests for Smart Turn detector."""

import numpy as np
import pytest

from pipeline.turn_detector import SmartTurnDetector, TurnDetectionManager


class TestSmartTurnDetector:
    """Test SmartTurnDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance (downloads model on first run)."""
        return SmartTurnDetector(threshold=0.7)

    def test_create_detector(self, detector):
        """Test creating detector."""
        assert detector.threshold == 0.7
        assert detector.session is not None
        assert detector.MODEL_SAMPLES == 128000  # 8 seconds @ 16kHz

    def test_prepare_audio_exact_length(self, detector):
        """Test preparing audio of exact length."""
        audio = np.random.randn(128000).astype(np.float32)

        prepared = detector.prepare_audio(audio)

        assert len(prepared) == 128000
        assert np.array_equal(prepared, audio)

    def test_prepare_audio_too_short(self, detector):
        """Test preparing audio shorter than 8 seconds."""
        audio = np.random.randn(16000).astype(np.float32)  # 1 second

        prepared = detector.prepare_audio(audio)

        assert len(prepared) == 128000
        # Should be zero-padded at beginning
        assert np.all(prepared[:112000] == 0)  # First 7 seconds
        assert np.array_equal(prepared[112000:], audio)  # Last 1 second

    def test_prepare_audio_too_long(self, detector):
        """Test preparing audio longer than 8 seconds."""
        audio = np.random.randn(160000).astype(np.float32)  # 10 seconds

        prepared = detector.prepare_audio(audio)

        assert len(prepared) == 128000
        # Should keep most recent 8 seconds
        assert np.array_equal(prepared, audio[-128000:])

    def test_detect_silence(self, detector):
        """Test detecting on silence."""
        # Generate 2 seconds of silence (will be padded to 8s)
        silence = np.zeros(32000, dtype=np.float32)

        is_complete, confidence = detector.detect(silence)

        # Silence typically indicates turn completion
        assert isinstance(is_complete, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_detect_short_audio(self, detector):
        """Test detecting on short audio."""
        # Generate 1 second of audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        is_complete, confidence = detector.detect(audio)

        # Short audio with padding should have some prediction
        assert isinstance(is_complete, bool)
        assert 0.0 <= confidence <= 1.0

    def test_detect_full_audio(self, detector):
        """Test detecting on full 8 seconds."""
        # Generate 8 seconds of audio
        t = np.arange(128000, dtype=np.float32) / 16000
        # Sine wave that fades out (simulates speech ending)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        envelope = np.exp(-t / 2).astype(np.float32)  # Exponential decay
        audio = audio * envelope

        is_complete, confidence = detector.detect(audio)

        assert isinstance(is_complete, bool)
        assert 0.0 <= confidence <= 1.0

    def test_set_threshold(self, detector):
        """Test updating threshold."""
        detector.set_threshold(0.5)
        assert detector.threshold == 0.5

        detector.set_threshold(0.9)
        assert detector.threshold == 0.9

    def test_threshold_validation(self, detector):
        """Test threshold validation."""
        with pytest.raises(ValueError):
            detector.set_threshold(-0.1)

        with pytest.raises(ValueError):
            detector.set_threshold(1.1)

    def test_get_model_info(self, detector):
        """Test getting model info."""
        info = detector.get_model_info()

        assert info["loaded"] is True
        assert "path" in info
        assert info["threshold"] == 0.7
        assert info["sample_rate"] == 16000
        assert info["duration"] == 8.0
        assert info["samples"] == 128000

    @pytest.mark.asyncio
    async def test_detect_async(self, detector):
        """Test async detection."""
        audio = np.random.randn(32000).astype(np.float32) * 0.1

        is_complete, confidence = await detector.detect_async(audio)

        assert isinstance(is_complete, bool)
        assert 0.0 <= confidence <= 1.0


class TestTurnDetectionManager:
    """Test TurnDetectionManager class."""

    @pytest.fixture
    def detector(self):
        """Create detector for manager."""
        return SmartTurnDetector(threshold=0.7)

    @pytest.fixture
    def manager(self, detector):
        """Create manager instance."""
        return TurnDetectionManager(
            detector=detector,
            max_wait=1.0,  # Short for testing
            check_interval=0.1,
        )

    @pytest.mark.asyncio
    async def test_check_turn_complete_immediate(self, manager):
        """Test turn check when immediately complete."""
        # Generate audio that appears complete (silence at end)
        audio = np.zeros(32000, dtype=np.float32)

        is_complete, confidence, timed_out = await manager.check_turn_complete(
            user_id=123,
            audio=audio,
        )

        assert isinstance(is_complete, bool)
        assert 0.0 <= confidence <= 1.0
        # Should complete quickly (not timeout)

    @pytest.mark.asyncio
    async def test_check_turn_incomplete_no_callback(self, manager):
        """Test incomplete turn with no callback."""
        # Set very high threshold so it's unlikely to be complete
        manager.detector.set_threshold(0.99)

        # Generate short audio
        audio = np.random.randn(8000).astype(np.float32) * 0.5

        is_complete, confidence, timed_out = await manager.check_turn_complete(
            user_id=123,
            audio=audio,
            audio_callback=None,  # No callback
        )

        # Should return as complete since no callback available
        assert is_complete is True

    @pytest.mark.asyncio
    async def test_cancel_waiting(self, manager):
        """Test cancelling wait for user."""
        # This should complete without error
        manager.cancel_waiting(user_id=123)

        # Cancelling non-existent wait should be safe
        manager.cancel_waiting(user_id=999)

    @pytest.mark.asyncio
    async def test_cancel_all(self, manager):
        """Test cancelling all waits."""
        manager.cancel_all()

        # Should complete without error even with no active waits


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
