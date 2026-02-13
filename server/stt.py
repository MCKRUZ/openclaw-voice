"""Speech-to-Text using faster-whisper.

GPU-accelerated transcription with support for multiple model sizes.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

from utils.logging import get_logger, log_latency

logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed speech."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float  # Average log probability (0.0-1.0 approximation)

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str  # Full transcript
    segments: List[TranscriptSegment]  # Individual segments
    language: str  # Detected/specified language
    duration: float  # Audio duration in seconds

    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.text.split())

    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self.segments)


class FasterWhisperSTT:
    """
    Faster-whisper STT engine.

    Much faster than OpenAI Whisper while maintaining similar accuracy.
    Uses CTranslate2 for efficient inference on CPU and GPU.
    """

    # Available model sizes (quality vs speed tradeoff)
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: Optional[str] = None,
        download_root: Optional[Path] = None,
    ):
        """
        Initialize faster-whisper STT engine.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu)
            compute_type: Compute precision (float16, float32, int8)
            beam_size: Beam search size (higher = more accurate but slower)
            language: Language code (None = auto-detect)
            download_root: Model download directory (None = default cache)
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(
                f"Invalid model size {model_size}. "
                f"Choose from: {self.MODEL_SIZES}"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.download_root = download_root

        # Model instance
        self.model: Optional[WhisperModel] = None

        # Load model
        self._load_model()

        # Stats
        self.transcription_count = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(
                f"Loading faster-whisper model: {self.model_size} "
                f"(device: {self.device}, compute: {self.compute_type})"
            )

            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )

            logger.info(f"Whisper model loaded successfully: {self.model_size}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio array (float32, mono, 16kHz)
            language: Language code (overrides instance setting)
            beam_size: Beam search size (overrides instance setting)
            vad_filter: Use VAD to filter out silence

        Returns:
            TranscriptionResult with text and segments
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Validate audio
        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio.dtype}")

        if len(audio.shape) != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

        # Use provided values or instance defaults
        language = language or self.language
        beam_size = beam_size or self.beam_size

        with log_latency(logger, f"transcribe_{self.model_size}"):
            # Run transcription
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=False,  # Disable for speed
            )

            # Convert generator to list and build result
            segment_list = []
            full_text = []

            for segment in segments:
                # Create segment object
                seg = TranscriptSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=float(np.exp(segment.avg_logprob)),  # Convert log prob
                )
                segment_list.append(seg)
                full_text.append(seg.text)

            # Build result
            result = TranscriptionResult(
                text=" ".join(full_text).strip(),
                segments=segment_list,
                language=info.language,
                duration=info.duration,
            )

            # Update stats
            self.transcription_count += 1
            self.total_audio_duration += result.duration

            logger.info(
                f"Transcribed {result.duration:.2f}s audio: "
                f'"{result.text[:50]}..." '
                f"({result.segment_count} segments, language: {result.language})"
            )

            return result

    async def transcribe_async(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: bool = False,
    ) -> TranscriptionResult:
        """
        Async wrapper for transcribe().

        Runs transcription in executor to avoid blocking event loop.

        Args:
            audio: Audio array
            language: Language code
            beam_size: Beam search size
            vad_filter: Use VAD filter

        Returns:
            TranscriptionResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.transcribe,
            audio,
            language,
            beam_size,
            vad_filter,
        )

    def get_stats(self) -> dict:
        """
        Get transcription statistics.

        Returns:
            Dictionary with stats
        """
        avg_duration = (
            self.total_audio_duration / self.transcription_count
            if self.transcription_count > 0
            else 0.0
        )

        avg_processing = (
            self.total_processing_time / self.transcription_count
            if self.transcription_count > 0
            else 0.0
        )

        rtf = (
            avg_processing / avg_duration
            if avg_duration > 0
            else 0.0
        )  # Real-time factor

        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "transcription_count": self.transcription_count,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "avg_audio_duration": avg_duration,
            "avg_processing_time": avg_processing,
            "real_time_factor": rtf,
        }

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "language": self.language or "auto-detect",
            "loaded": self.model is not None,
        }


class STTTranscriber:
    """
    Pipeline stage for speech-to-text transcription.

    Handles queueing and concurrent transcription requests.
    """

    def __init__(
        self,
        engine: FasterWhisperSTT,
        max_concurrent: int = 1,
    ):
        """
        Initialize transcriber.

        Args:
            engine: STT engine instance
            max_concurrent: Max concurrent transcriptions (default 1 for single GPU)
        """
        self.engine = engine
        self.max_concurrent = max_concurrent

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Queue for pending requests
        self._queue_size = 0

    async def transcribe(
        self,
        audio: np.ndarray,
        user_id: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio with queue management.

        Args:
            audio: Audio array (float32, mono, 16kHz)
            user_id: User ID for logging
            language: Language code (optional)

        Returns:
            TranscriptionResult
        """
        async with self._semaphore:
            self._queue_size = self.max_concurrent - self._semaphore._value

            logger.debug(
                f"Transcribing for user {user_id} "
                f"(queue size: {self._queue_size})"
            )

            try:
                result = await self.engine.transcribe_async(
                    audio=audio,
                    language=language,
                )

                logger.info(
                    f"User {user_id} transcription: "
                    f'"{result.text}" '
                    f"({result.duration:.2f}s, {result.word_count} words)"
                )

                return result

            except Exception as e:
                logger.error(f"Transcription error for user {user_id}: {e}")
                raise

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue_size

    def get_stats(self) -> dict:
        """Get transcriber statistics."""
        return {
            **self.engine.get_stats(),
            "max_concurrent": self.max_concurrent,
            "current_queue_size": self._queue_size,
        }


# Convenience function for creating transcriber
async def create_transcriber(
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
) -> STTTranscriber:
    """
    Create STT transcriber with default settings.

    Args:
        model_size: Whisper model size
        device: Device (cuda/cpu)
        compute_type: Compute precision
        language: Language code

    Returns:
        STTTranscriber instance
    """
    engine = FasterWhisperSTT(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    transcriber = STTTranscriber(
        engine=engine,
        max_concurrent=1,  # Single GPU, process one at a time
    )

    return transcriber
