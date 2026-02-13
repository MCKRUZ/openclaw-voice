"""Audio format conversion and processing utilities.

Handles conversion between various audio formats used by Discord, VAD, STT, and TTS.

Typical conversions:
    Discord (48kHz stereo int16) → Processing (16kHz mono int16) → Numpy (float32)
    Numpy (float32) → Processing (16kHz mono int16) → Discord (48kHz stereo int16)
"""

import io
import struct
from typing import Optional, Tuple

import numpy as np
from scipy import signal


# Audio format constants
DISCORD_SAMPLE_RATE = 48000  # Hz
PROCESSING_SAMPLE_RATE = 16000  # Hz
DISCORD_CHANNELS = 2  # Stereo
PROCESSING_CHANNELS = 1  # Mono
DISCORD_FRAME_SIZE = 960  # Samples per channel per frame (20ms @ 48kHz)
DISCORD_FRAME_DURATION = 0.02  # 20ms

# Opus frame sizes (samples per channel)
OPUS_FRAME_SIZES = {
    DISCORD_SAMPLE_RATE: [120, 240, 480, 960, 1920, 2880],  # Valid at 48kHz
}


def pcm_to_numpy(pcm_data: bytes, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Convert PCM bytes to numpy array.

    Args:
        pcm_data: Raw PCM bytes
        dtype: Data type (np.int16 or np.float32)

    Returns:
        Numpy array of audio samples

    Example:
        >>> pcm_bytes = b'\\x00\\x00\\xFF\\x7F'  # 2 int16 samples
        >>> audio = pcm_to_numpy(pcm_bytes, np.int16)
        >>> audio.shape
        (2,)
    """
    if dtype == np.int16:
        return np.frombuffer(pcm_data, dtype=np.int16)
    elif dtype == np.float32:
        # Convert from int16 to float32 in range [-1.0, 1.0]
        int16_array = np.frombuffer(pcm_data, dtype=np.int16)
        return int16_array.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def numpy_to_pcm(audio: np.ndarray, dtype: np.dtype = np.int16) -> bytes:
    """
    Convert numpy array to PCM bytes.

    Args:
        audio: Numpy array of audio samples
        dtype: Target data type (np.int16 or np.float32)

    Returns:
        Raw PCM bytes

    Example:
        >>> audio = np.array([0, 32767], dtype=np.int16)
        >>> pcm_bytes = numpy_to_pcm(audio)
        >>> len(pcm_bytes)
        4
    """
    if dtype == np.int16:
        # Ensure input is int16
        if audio.dtype != np.int16:
            # Assume float32 in range [-1.0, 1.0]
            audio = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        return audio.tobytes()
    elif dtype == np.float32:
        # Ensure input is float32
        if audio.dtype != np.float32:
            # Assume int16
            audio = audio.astype(np.float32) / 32768.0
        return audio.tobytes()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert int16 audio to float32 in range [-1.0, 1.0].

    Args:
        audio: Int16 audio array

    Returns:
        Float32 audio array normalized to [-1.0, 1.0]
    """
    if audio.dtype != np.int16:
        raise ValueError(f"Expected int16, got {audio.dtype}")

    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio to int16.

    Args:
        audio: Float32 audio array (values should be in [-1.0, 1.0])

    Returns:
        Int16 audio array
    """
    if audio.dtype != np.float32:
        raise ValueError(f"Expected float32, got {audio.dtype}")

    # Clip to valid range and convert
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16)


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio: Stereo audio array (interleaved or shape [samples, 2])

    Returns:
        Mono audio array

    Example:
        >>> stereo = np.array([100, 200, 300, 400], dtype=np.int16)  # L, R, L, R
        >>> mono = stereo_to_mono(stereo)
        >>> mono
        array([150, 350], dtype=int16)
    """
    if len(audio.shape) == 1:
        # Interleaved stereo (L, R, L, R, ...)
        if len(audio) % 2 != 0:
            raise ValueError("Stereo audio must have even number of samples")

        # Reshape to [samples, 2] and average
        stereo_shaped = audio.reshape(-1, 2)
        return stereo_shaped.mean(axis=1).astype(audio.dtype)

    elif len(audio.shape) == 2 and audio.shape[1] == 2:
        # Already shaped [samples, 2]
        return audio.mean(axis=1).astype(audio.dtype)

    else:
        raise ValueError(f"Invalid stereo audio shape: {audio.shape}")


def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert mono audio to stereo by duplicating the channel.

    Args:
        audio: Mono audio array

    Returns:
        Stereo audio array (interleaved: L, R, L, R, ...)

    Example:
        >>> mono = np.array([100, 200], dtype=np.int16)
        >>> stereo = mono_to_stereo(mono)
        >>> stereo
        array([100, 100, 200, 200], dtype=int16)
    """
    if len(audio.shape) != 1:
        raise ValueError(f"Expected 1D mono audio, got shape {audio.shape}")

    # Stack and interleave
    stereo = np.repeat(audio, 2)
    return stereo


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    method: str = "scipy",
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        audio: Audio array (mono or stereo interleaved)
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)
        method: Resampling method ('scipy', 'linear')

    Returns:
        Resampled audio array

    Example:
        >>> audio_48k = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
        >>> audio_16k = resample(audio_48k, 48000, 16000)
        >>> len(audio_16k)
        2
    """
    if orig_sr == target_sr:
        return audio

    if method == "scipy":
        # High-quality resampling using scipy
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)

        # Preserve dtype
        if audio.dtype == np.int16:
            resampled = resampled.clip(-32768, 32767).astype(np.int16)
        elif audio.dtype == np.float32:
            resampled = resampled.astype(np.float32)

        return resampled

    elif method == "linear":
        # Fast linear interpolation
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio,
        )

        # Preserve dtype
        if audio.dtype == np.int16:
            resampled = resampled.clip(-32768, 32767).astype(np.int16)
        elif audio.dtype == np.float32:
            resampled = resampled.astype(np.float32)

        return resampled

    else:
        raise ValueError(f"Unknown resampling method: {method}")


def discord_to_processing(pcm_data: bytes) -> np.ndarray:
    """
    Convert Discord audio format to processing format.

    Discord: 48kHz stereo int16
    Processing: 16kHz mono float32

    Args:
        pcm_data: Raw PCM from Discord (48kHz stereo int16)

    Returns:
        Numpy array ready for VAD/STT (16kHz mono float32)
    """
    # Convert to numpy (int16)
    audio = pcm_to_numpy(pcm_data, dtype=np.int16)

    # Stereo to mono
    audio = stereo_to_mono(audio)

    # Resample 48kHz → 16kHz
    audio = resample(audio, DISCORD_SAMPLE_RATE, PROCESSING_SAMPLE_RATE)

    # Convert to float32
    audio = int16_to_float32(audio)

    return audio


def processing_to_discord(audio: np.ndarray) -> bytes:
    """
    Convert processing format to Discord audio format.

    Processing: 16kHz mono float32
    Discord: 48kHz stereo int16

    Args:
        audio: Processing audio (16kHz mono float32)

    Returns:
        Raw PCM for Discord (48kHz stereo int16)
    """
    # Convert to int16
    audio = float32_to_int16(audio)

    # Resample 16kHz → 48kHz
    audio = resample(audio, PROCESSING_SAMPLE_RATE, DISCORD_SAMPLE_RATE)

    # Mono to stereo
    audio = mono_to_stereo(audio)

    # Convert to bytes
    return numpy_to_pcm(audio, dtype=np.int16)


def validate_opus_frame_size(frame_size: int, sample_rate: int) -> bool:
    """
    Check if frame size is valid for Opus encoding.

    Args:
        frame_size: Number of samples per channel
        sample_rate: Sample rate in Hz

    Returns:
        True if valid, False otherwise
    """
    valid_sizes = OPUS_FRAME_SIZES.get(sample_rate, [])
    return frame_size in valid_sizes


def align_to_opus_frame(
    pcm_data: bytes,
    sample_rate: int = DISCORD_SAMPLE_RATE,
    channels: int = DISCORD_CHANNELS,
) -> bytes:
    """
    Align PCM data to Opus frame boundary by padding with silence if needed.

    Args:
        pcm_data: Raw PCM data
        sample_rate: Sample rate (Hz)
        channels: Number of channels

    Returns:
        PCM data aligned to frame boundary (may be padded)
    """
    bytes_per_sample = 2  # int16
    frame_size = DISCORD_FRAME_SIZE  # 960 samples per channel
    frame_bytes = frame_size * channels * bytes_per_sample

    remainder = len(pcm_data) % frame_bytes

    if remainder == 0:
        return pcm_data

    # Pad with silence
    padding_bytes = frame_bytes - remainder
    return pcm_data + (b"\x00" * padding_bytes)


def split_into_frames(
    pcm_data: bytes,
    frame_size: int = DISCORD_FRAME_SIZE,
    sample_rate: int = DISCORD_SAMPLE_RATE,
    channels: int = DISCORD_CHANNELS,
) -> list[bytes]:
    """
    Split PCM data into frames of specified size.

    Args:
        pcm_data: Raw PCM data
        frame_size: Samples per channel per frame
        sample_rate: Sample rate (Hz)
        channels: Number of channels

    Returns:
        List of frame bytes
    """
    bytes_per_sample = 2  # int16
    frame_bytes = frame_size * channels * bytes_per_sample

    frames = []
    for i in range(0, len(pcm_data), frame_bytes):
        frame = pcm_data[i : i + frame_bytes]
        if len(frame) == frame_bytes:
            frames.append(frame)

    return frames


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) of audio signal.

    Useful for measuring audio loudness.

    Args:
        audio: Audio array (int16 or float32)

    Returns:
        RMS value
    """
    if audio.dtype == np.int16:
        audio = int16_to_float32(audio)

    return float(np.sqrt(np.mean(audio**2)))


def compute_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """
    Compute decibel level of audio signal.

    Args:
        audio: Audio array (int16 or float32)
        ref: Reference value (default 1.0 for float32)

    Returns:
        Decibel level (dB)
    """
    rms = compute_rms(audio)
    if rms == 0:
        return -np.inf

    return float(20 * np.log10(rms / ref))


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target decibel level.

    Args:
        audio: Audio array (float32)
        target_db: Target RMS level in dB

    Returns:
        Normalized audio array
    """
    if audio.dtype != np.float32:
        raise ValueError("normalize_audio requires float32 input")

    current_db = compute_db(audio)
    if current_db == -np.inf:
        return audio  # Silent audio, no normalization needed

    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    normalized = audio * gain_linear

    # Clip to valid range
    return np.clip(normalized, -1.0, 1.0)


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain to audio signal.

    Args:
        audio: Audio array (float32)
        gain_db: Gain in decibels (positive = louder, negative = quieter)

    Returns:
        Audio with gain applied
    """
    if audio.dtype != np.float32:
        raise ValueError("apply_gain requires float32 input")

    gain_linear = 10 ** (gain_db / 20)
    return np.clip(audio * gain_linear, -1.0, 1.0)


def detect_silence(
    audio: np.ndarray,
    threshold_db: float = -40.0,
    frame_duration: float = 0.02,
    sample_rate: int = PROCESSING_SAMPLE_RATE,
) -> bool:
    """
    Detect if audio is predominantly silence.

    Args:
        audio: Audio array (float32)
        threshold_db: Silence threshold in dB
        frame_duration: Frame duration for analysis (seconds)
        sample_rate: Sample rate (Hz)

    Returns:
        True if audio is silence, False otherwise
    """
    if len(audio) == 0:
        return True

    # Compute RMS in dB
    db_level = compute_db(audio)

    return db_level < threshold_db


# Validation functions
def validate_sample_rate(sample_rate: int) -> None:
    """Validate sample rate is supported."""
    valid_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    if sample_rate not in valid_rates:
        raise ValueError(
            f"Sample rate {sample_rate} not in valid rates: {valid_rates}"
        )


def validate_channels(channels: int) -> None:
    """Validate number of channels is supported."""
    if channels not in [1, 2]:
        raise ValueError(f"Channels must be 1 (mono) or 2 (stereo), got {channels}")


def validate_audio_format(
    pcm_data: bytes,
    sample_rate: int,
    channels: int,
    duration_ms: Optional[int] = None,
) -> None:
    """
    Validate audio format is correct.

    Args:
        pcm_data: Raw PCM data
        sample_rate: Sample rate (Hz)
        channels: Number of channels
        duration_ms: Expected duration in milliseconds (optional)

    Raises:
        ValueError: If format is invalid
    """
    validate_sample_rate(sample_rate)
    validate_channels(channels)

    bytes_per_sample = 2  # int16
    expected_bytes_per_ms = sample_rate * channels * bytes_per_sample // 1000

    if duration_ms is not None:
        expected_bytes = expected_bytes_per_ms * duration_ms
        if len(pcm_data) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} bytes for {duration_ms}ms, "
                f"got {len(pcm_data)} bytes"
            )

    # Check byte alignment
    if len(pcm_data) % (channels * bytes_per_sample) != 0:
        raise ValueError(
            f"PCM data length ({len(pcm_data)}) not aligned to sample size "
            f"({channels * bytes_per_sample} bytes)"
        )
