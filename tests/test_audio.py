"""Unit tests for audio utilities."""

import numpy as np
import pytest

from utils import audio


class TestPCMConversion:
    """Test PCM bytes ↔ numpy array conversion."""

    def test_pcm_to_numpy_int16(self):
        """Test converting PCM bytes to int16 numpy array."""
        # Create test data: 4 samples (8 bytes)
        pcm_data = b"\x00\x00\xFF\x7F\x00\x80\x01\x00"  # [0, 32767, -32768, 1]

        audio_array = audio.pcm_to_numpy(pcm_data, dtype=np.int16)

        assert audio_array.dtype == np.int16
        assert len(audio_array) == 4
        assert audio_array[0] == 0
        assert audio_array[1] == 32767
        assert audio_array[2] == -32768
        assert audio_array[3] == 1

    def test_pcm_to_numpy_float32(self):
        """Test converting PCM bytes to float32 numpy array."""
        # Max int16 value should become ~1.0
        pcm_data = b"\xFF\x7F"  # 32767

        audio_array = audio.pcm_to_numpy(pcm_data, dtype=np.float32)

        assert audio_array.dtype == np.float32
        assert len(audio_array) == 1
        assert abs(audio_array[0] - 1.0) < 0.001  # Should be very close to 1.0

    def test_numpy_to_pcm_int16(self):
        """Test converting int16 numpy array to PCM bytes."""
        audio_array = np.array([0, 32767, -32768, 1], dtype=np.int16)

        pcm_data = audio.numpy_to_pcm(audio_array, dtype=np.int16)

        assert len(pcm_data) == 8
        assert pcm_data == b"\x00\x00\xFF\x7F\x00\x80\x01\x00"

    def test_numpy_to_pcm_float32_conversion(self):
        """Test converting float32 to int16 PCM."""
        audio_array = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)

        pcm_data = audio.numpy_to_pcm(audio_array, dtype=np.int16)

        # Convert back to verify
        result = audio.pcm_to_numpy(pcm_data, dtype=np.int16)

        assert result[0] == 0
        assert result[1] == 32767  # 1.0 * 32768 clipped to 32767
        assert result[2] == -32768
        assert abs(result[3] - 16384) < 2  # 0.5 * 32768

    def test_round_trip_int16(self):
        """Test PCM → numpy → PCM round trip."""
        original = b"\x00\x00\xFF\x7F\x00\x80"

        audio_array = audio.pcm_to_numpy(original, dtype=np.int16)
        result = audio.numpy_to_pcm(audio_array, dtype=np.int16)

        assert result == original


class TestDataTypeConversion:
    """Test int16 ↔ float32 conversion."""

    def test_int16_to_float32(self):
        """Test converting int16 to float32."""
        audio_int16 = np.array([0, 32767, -32768, 16384], dtype=np.int16)

        audio_float32 = audio.int16_to_float32(audio_int16)

        assert audio_float32.dtype == np.float32
        assert audio_float32[0] == 0.0
        assert abs(audio_float32[1] - 1.0) < 0.001
        assert audio_float32[2] == -1.0
        assert abs(audio_float32[3] - 0.5) < 0.001

    def test_float32_to_int16(self):
        """Test converting float32 to int16."""
        audio_float32 = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)

        audio_int16 = audio.float32_to_int16(audio_float32)

        assert audio_int16.dtype == np.int16
        assert audio_int16[0] == 0
        assert audio_int16[1] == 32767  # Clipped from 32768
        assert audio_int16[2] == -32768
        assert abs(audio_int16[3] - 16384) < 2

    def test_float32_to_int16_clipping(self):
        """Test that values outside [-1, 1] are clipped."""
        audio_float32 = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)

        audio_int16 = audio.float32_to_int16(audio_float32)

        assert audio_int16[0] == 32767  # Clipped
        assert audio_int16[1] == -32768  # Clipped
        assert audio_int16[2] == 32767  # Clipped
        assert audio_int16[3] == -32768  # Clipped

    def test_round_trip_conversion(self):
        """Test int16 → float32 → int16 round trip."""
        original = np.array([0, 10000, -10000, 32767, -32768], dtype=np.int16)

        float32_version = audio.int16_to_float32(original)
        result = audio.float32_to_int16(float32_version)

        # Should be identical (or very close due to float precision)
        assert np.allclose(result, original, atol=1)


class TestChannelConversion:
    """Test stereo ↔ mono conversion."""

    def test_stereo_to_mono_interleaved(self):
        """Test converting interleaved stereo to mono."""
        # Stereo: L=100, R=200, L=300, R=400
        stereo = np.array([100, 200, 300, 400], dtype=np.int16)

        mono = audio.stereo_to_mono(stereo)

        assert len(mono) == 2
        assert mono[0] == 150  # (100 + 200) / 2
        assert mono[1] == 350  # (300 + 400) / 2

    def test_stereo_to_mono_shaped(self):
        """Test converting shaped [samples, 2] stereo to mono."""
        stereo = np.array([[100, 200], [300, 400]], dtype=np.int16)

        mono = audio.stereo_to_mono(stereo)

        assert len(mono) == 2
        assert mono[0] == 150
        assert mono[1] == 350

    def test_mono_to_stereo(self):
        """Test converting mono to stereo."""
        mono = np.array([100, 200, 300], dtype=np.int16)

        stereo = audio.mono_to_stereo(mono)

        assert len(stereo) == 6
        # Should be: L, R, L, R, L, R with L=R for each sample
        assert stereo[0] == 100  # L
        assert stereo[1] == 100  # R
        assert stereo[2] == 200  # L
        assert stereo[3] == 200  # R
        assert stereo[4] == 300  # L
        assert stereo[5] == 300  # R

    def test_stereo_mono_round_trip(self):
        """Test mono → stereo → mono round trip."""
        original = np.array([100, 200, 300], dtype=np.int16)

        stereo = audio.mono_to_stereo(original)
        result = audio.stereo_to_mono(stereo)

        assert np.array_equal(result, original)


class TestResampling:
    """Test audio resampling."""

    def test_resample_downsampling(self):
        """Test downsampling 48kHz → 16kHz."""
        # Create 48kHz audio (48 samples = 1ms)
        audio_48k = np.sin(
            2 * np.pi * 440 * np.arange(48000) / 48000
        ).astype(np.float32)

        audio_16k = audio.resample(audio_48k, 48000, 16000)

        # Should have 1/3 the samples
        expected_length = 16000
        assert abs(len(audio_16k) - expected_length) < 5

    def test_resample_upsampling(self):
        """Test upsampling 16kHz → 48kHz."""
        # Create 16kHz audio
        audio_16k = np.sin(
            2 * np.pi * 440 * np.arange(16000) / 16000
        ).astype(np.float32)

        audio_48k = audio.resample(audio_16k, 16000, 48000)

        # Should have 3x the samples
        expected_length = 48000
        assert abs(len(audio_48k) - expected_length) < 5

    def test_resample_no_change(self):
        """Test resampling with same rate returns original."""
        original = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        result = audio.resample(original, 16000, 16000)

        assert np.array_equal(result, original)

    def test_resample_preserves_dtype(self):
        """Test resampling preserves data type."""
        audio_int16 = np.array([1000, 2000, 3000, 4000], dtype=np.int16)

        result = audio.resample(audio_int16, 48000, 16000)

        assert result.dtype == np.int16

    def test_resample_linear_method(self):
        """Test linear interpolation resampling."""
        audio_48k = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)

        audio_16k = audio.resample(audio_48k, 48000, 16000, method="linear")

        assert len(audio_16k) == 2  # 1/3 of 6


class TestCompleteConversions:
    """Test complete format conversions."""

    def test_discord_to_processing(self):
        """Test Discord → processing conversion."""
        # Create 20ms of 48kHz stereo audio (960 samples per channel)
        duration_samples = 960
        stereo_samples = duration_samples * 2  # Interleaved L, R

        # Create test signal: 440Hz sine wave
        t = np.arange(duration_samples) / 48000
        signal_mono = np.sin(2 * np.pi * 440 * t)
        signal_stereo = np.repeat(signal_mono, 2)  # Duplicate for stereo

        # Convert to int16 PCM
        pcm_int16 = (signal_stereo * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        # Convert to processing format
        result = audio.discord_to_processing(pcm_bytes)

        # Should be 16kHz mono float32
        assert result.dtype == np.float32
        expected_length = int(duration_samples * 16000 / 48000)
        assert abs(len(result) - expected_length) < 5
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_processing_to_discord(self):
        """Test processing → Discord conversion."""
        # Create 20ms of 16kHz mono float32 audio
        duration_samples = 320  # 20ms @ 16kHz
        t = np.arange(duration_samples) / 16000
        audio_processing = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Convert to Discord format
        pcm_bytes = audio.processing_to_discord(audio_processing)

        # Should be 48kHz stereo int16
        expected_samples = int(duration_samples * 48000 / 16000) * 2  # Stereo
        expected_bytes = expected_samples * 2  # int16 = 2 bytes
        assert abs(len(pcm_bytes) - expected_bytes) < 20

    def test_round_trip_conversion(self):
        """Test Discord → processing → Discord round trip."""
        # Create simple test signal
        original = np.array([0, 10000, -10000, 20000] * 240, dtype=np.int16)
        pcm_bytes = original.tobytes()

        # Convert to processing and back
        processing = audio.discord_to_processing(pcm_bytes)
        result_bytes = audio.processing_to_discord(processing)

        # Won't be exact due to resampling, but should be similar length
        assert abs(len(result_bytes) - len(pcm_bytes)) < 100


class TestOpusFraming:
    """Test Opus frame handling."""

    def test_validate_opus_frame_size(self):
        """Test Opus frame size validation."""
        assert audio.validate_opus_frame_size(960, 48000) is True
        assert audio.validate_opus_frame_size(480, 48000) is True
        assert audio.validate_opus_frame_size(1000, 48000) is False

    def test_align_to_opus_frame_already_aligned(self):
        """Test alignment when already aligned."""
        # 960 samples * 2 channels * 2 bytes = 3840 bytes
        pcm_data = b"\x00" * 3840

        result = audio.align_to_opus_frame(pcm_data)

        assert result == pcm_data

    def test_align_to_opus_frame_needs_padding(self):
        """Test alignment with padding."""
        # 100 bytes (not aligned)
        pcm_data = b"\x00" * 100

        result = audio.align_to_opus_frame(pcm_data)

        # Should be padded to next frame boundary
        assert len(result) > len(pcm_data)
        assert len(result) % 3840 == 0

    def test_split_into_frames(self):
        """Test splitting PCM into frames."""
        # 2 complete frames worth of data
        frame_bytes = 960 * 2 * 2  # 960 samples, 2 channels, 2 bytes
        pcm_data = b"\x00" * (frame_bytes * 2)

        frames = audio.split_into_frames(pcm_data)

        assert len(frames) == 2
        assert len(frames[0]) == frame_bytes
        assert len(frames[1]) == frame_bytes

    def test_split_into_frames_incomplete(self):
        """Test splitting with incomplete last frame."""
        frame_bytes = 960 * 2 * 2
        pcm_data = b"\x00" * (frame_bytes + 100)  # One complete + incomplete

        frames = audio.split_into_frames(pcm_data)

        # Incomplete frame should be dropped
        assert len(frames) == 1


class TestAudioAnalysis:
    """Test audio analysis functions."""

    def test_compute_rms_silence(self):
        """Test RMS of silence."""
        silence = np.zeros(1000, dtype=np.float32)

        rms = audio.compute_rms(silence)

        assert rms == 0.0

    def test_compute_rms_full_scale(self):
        """Test RMS of full-scale signal."""
        full_scale = np.ones(1000, dtype=np.float32)

        rms = audio.compute_rms(full_scale)

        assert abs(rms - 1.0) < 0.001

    def test_compute_db_silence(self):
        """Test dB of silence."""
        silence = np.zeros(1000, dtype=np.float32)

        db = audio.compute_db(silence)

        assert db == -np.inf

    def test_compute_db_full_scale(self):
        """Test dB of full-scale signal."""
        full_scale = np.ones(1000, dtype=np.float32)

        db = audio.compute_db(full_scale)

        assert abs(db - 0.0) < 0.1  # Should be ~0 dB

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create quiet audio (RMS = 0.01, which is ~-40 dB)
        quiet = np.ones(1000, dtype=np.float32) * 0.01

        # Normalize to -20 dB (should make it louder)
        normalized = audio.normalize_audio(quiet, target_db=-20.0)

        # Should be louder now
        assert audio.compute_rms(normalized) > audio.compute_rms(quiet)

        # Target dB should be close to -20 dB
        target_db = audio.compute_db(normalized)
        assert abs(target_db - (-20.0)) < 1.0  # Within 1 dB

    def test_apply_gain(self):
        """Test applying gain."""
        original = np.ones(1000, dtype=np.float32) * 0.5

        # Apply +6dB gain (should approximately double)
        louder = audio.apply_gain(original, 6.0)

        assert audio.compute_rms(louder) > audio.compute_rms(original)

        # Apply -6dB gain (should approximately halve)
        quieter = audio.apply_gain(original, -6.0)

        assert audio.compute_rms(quieter) < audio.compute_rms(original)

    def test_detect_silence_true(self):
        """Test silence detection on quiet audio."""
        quiet = np.ones(1000, dtype=np.float32) * 0.001

        is_silence = audio.detect_silence(quiet, threshold_db=-40.0)

        assert is_silence is True

    def test_detect_silence_false(self):
        """Test silence detection on loud audio."""
        loud = np.ones(1000, dtype=np.float32) * 0.5

        is_silence = audio.detect_silence(loud, threshold_db=-40.0)

        assert is_silence is False


class TestValidation:
    """Test validation functions."""

    def test_validate_sample_rate_valid(self):
        """Test validating valid sample rates."""
        for rate in [16000, 48000, 44100]:
            audio.validate_sample_rate(rate)  # Should not raise

    def test_validate_sample_rate_invalid(self):
        """Test validating invalid sample rate."""
        with pytest.raises(ValueError):
            audio.validate_sample_rate(12345)

    def test_validate_channels_valid(self):
        """Test validating valid channel counts."""
        for channels in [1, 2]:
            audio.validate_channels(channels)  # Should not raise

    def test_validate_channels_invalid(self):
        """Test validating invalid channel count."""
        with pytest.raises(ValueError):
            audio.validate_channels(5)

    def test_validate_audio_format(self):
        """Test complete audio format validation."""
        # Create 20ms of 48kHz stereo audio
        duration_ms = 20
        sample_rate = 48000
        channels = 2
        num_samples = sample_rate * duration_ms // 1000
        pcm_data = b"\x00" * (num_samples * channels * 2)

        audio.validate_audio_format(pcm_data, sample_rate, channels, duration_ms)

    def test_validate_audio_format_wrong_duration(self):
        """Test validation fails with wrong duration."""
        pcm_data = b"\x00" * 100

        with pytest.raises(ValueError):
            audio.validate_audio_format(pcm_data, 48000, 2, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
