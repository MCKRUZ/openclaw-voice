"""Simple VAD test to verify Silero model loads and works."""

import numpy as np
import pytest

from pipeline.vad import SileroVAD, SpeechState


class TestSileroVADBasic:
    """Basic tests for Silero VAD (model loading may take time on first run)."""

    def test_create_vad(self):
        """Test creating VAD instance (downloads model on first run)."""
        vad = SileroVAD(
            sample_rate=16000,
            speech_threshold=0.5,
        )

        assert vad.sample_rate == 16000
        assert vad.model is not None
        assert vad.current_state == SpeechState.SILENCE

    def test_process_silence(self):
        """Test processing silence."""
        vad = SileroVAD(sample_rate=16000)

        # Generate silence (zeros)
        silence = np.zeros(512, dtype=np.float32)

        state, prob = vad.process_chunk(silence)

        assert state == SpeechState.SILENCE
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_process_noise(self):
        """Test processing random noise."""
        vad = SileroVAD(sample_rate=16000)

        # Generate low-level noise
        noise = np.random.randn(512).astype(np.float32) * 0.01

        state, prob = vad.process_chunk(noise)

        # Low noise should be detected as silence
        assert state == SpeechState.SILENCE

    def test_process_loud_signal(self):
        """Test processing loud signal (simulated speech)."""
        vad = SileroVAD(sample_rate=16000, speech_threshold=0.3)

        # Generate loud signal (simulates speech-like characteristics)
        # Silero VAD requires exactly 512 samples for 16kHz
        t = np.arange(512) / 16000
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone
        signal += np.random.randn(512).astype(np.float32) * 0.1  # Add noise

        state, prob = vad.process_chunk(signal)

        # Note: Silero VAD is trained on actual speech, so pure tones
        # may not be reliably detected. This test just ensures it runs.
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_reset(self):
        """Test resetting VAD state."""
        vad = SileroVAD(sample_rate=16000)

        # Process some audio (512 samples = valid chunk size for 16kHz)
        audio = np.random.randn(512).astype(np.float32)
        vad.process_stream(audio)

        # Reset
        vad.reset()

        assert vad.current_state == SpeechState.SILENCE
        assert vad.total_samples_processed == 0

    def test_streaming_with_silence(self):
        """Test streaming with silence (should not create segments)."""
        vad = SileroVAD(sample_rate=16000)

        # Process multiple chunks of silence
        for _ in range(10):
            silence = np.zeros(512, dtype=np.float32)
            state, segment = vad.process_stream(silence)

            assert state == SpeechState.SILENCE
            assert segment is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
