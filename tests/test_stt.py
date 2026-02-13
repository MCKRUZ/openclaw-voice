"""Unit tests for Speech-to-Text engine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from server.stt import (
    FasterWhisperSTT,
    STTTranscriber,
    TranscriptSegment,
    TranscriptionResult,
    create_transcriber,
)
from pipeline.transcriber import PipelineTranscriber, create_pipeline_transcriber


class TestTranscriptSegment:
    """Test TranscriptSegment dataclass."""

    def test_create_segment(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            text="Hello world",
            start=0.0,
            end=1.5,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence == 0.95

    def test_segment_duration(self):
        """Test segment duration calculation."""
        segment = TranscriptSegment(
            text="Test",
            start=2.0,
            end=5.5,
            confidence=0.9,
        )

        assert segment.duration == 3.5

    def test_segment_duration_zero(self):
        """Test zero duration segment."""
        segment = TranscriptSegment(
            text="Quick",
            start=1.0,
            end=1.0,
            confidence=0.8,
        )

        assert segment.duration == 0.0


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_create_result(self):
        """Test creating a transcription result."""
        segments = [
            TranscriptSegment("Hello", 0.0, 1.0, 0.95),
            TranscriptSegment("world", 1.0, 2.0, 0.93),
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            duration=2.0,
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 2.0

    def test_word_count(self):
        """Test word count calculation."""
        result = TranscriptionResult(
            text="This is a test sentence",
            segments=[],
            language="en",
            duration=3.0,
        )

        assert result.word_count == 5

    def test_word_count_empty(self):
        """Test word count for empty text."""
        result = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            duration=0.0,
        )

        # Empty string split() gives []
        assert result.word_count == 0

    def test_segment_count(self):
        """Test segment count."""
        segments = [
            TranscriptSegment("First", 0.0, 1.0, 0.9),
            TranscriptSegment("second", 1.0, 2.0, 0.85),
            TranscriptSegment("third", 2.0, 3.0, 0.92),
        ]

        result = TranscriptionResult(
            text="First second third",
            segments=segments,
            language="en",
            duration=3.0,
        )

        assert result.segment_count == 3


class TestFasterWhisperSTT:
    """Test FasterWhisperSTT class."""

    @pytest.fixture
    def mock_whisper_model(self):
        """Create mock WhisperModel."""
        with patch("server.stt.WhisperModel") as mock:
            # Mock the model instance
            model_instance = MagicMock()

            # Mock transcription response
            segment1 = Mock()
            segment1.text = " Hello "
            segment1.start = 0.0
            segment1.end = 1.0
            segment1.avg_logprob = -0.1

            segment2 = Mock()
            segment2.text = " world "
            segment2.start = 1.0
            segment2.end = 2.0
            segment2.avg_logprob = -0.15

            # Mock info
            info = Mock()
            info.language = "en"
            info.duration = 2.0

            # Model returns (segments_generator, info)
            model_instance.transcribe.return_value = ([segment1, segment2], info)

            mock.return_value = model_instance
            yield mock

    def test_create_engine_valid_model(self, mock_whisper_model):
        """Test creating engine with valid model size."""
        engine = FasterWhisperSTT(
            model_size="tiny",
            device="cpu",
            compute_type="float32",
        )

        assert engine.model_size == "tiny"
        assert engine.device == "cpu"
        assert engine.compute_type == "float32"
        assert engine.beam_size == 5  # default
        assert engine.language is None
        assert engine.model is not None

    def test_create_engine_invalid_model(self):
        """Test creating engine with invalid model size."""
        with pytest.raises(ValueError) as exc:
            FasterWhisperSTT(model_size="invalid")

        assert "Invalid model size" in str(exc.value)
        assert "Choose from:" in str(exc.value)

    def test_create_engine_with_language(self, mock_whisper_model):
        """Test creating engine with language specified."""
        engine = FasterWhisperSTT(
            model_size="tiny",
            device="cpu",
            language="es",
        )

        assert engine.language == "es"

    def test_transcribe_valid_audio(self, mock_whisper_model):
        """Test transcribing valid audio."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        # Generate 2 seconds of audio @ 16kHz
        audio = np.random.randn(32000).astype(np.float32)

        result = engine.transcribe(audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.0
        assert result.segment_count == 2
        assert result.word_count == 2

        # Check segments
        assert result.segments[0].text == "Hello"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0
        assert 0.0 <= result.segments[0].confidence <= 1.0

        # Check stats updated
        assert engine.transcription_count == 1
        assert engine.total_audio_duration == 2.0

    def test_transcribe_invalid_dtype(self, mock_whisper_model):
        """Test transcribing audio with wrong dtype."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        # Wrong dtype (float64 instead of float32)
        audio = np.random.randn(16000).astype(np.float64)

        with pytest.raises(ValueError) as exc:
            engine.transcribe(audio)

        assert "Expected float32 audio" in str(exc.value)

    def test_transcribe_invalid_shape(self, mock_whisper_model):
        """Test transcribing audio with wrong shape."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        # Wrong shape (2D instead of 1D)
        audio = np.random.randn(16000, 2).astype(np.float32)

        with pytest.raises(ValueError) as exc:
            engine.transcribe(audio)

        assert "Expected 1D audio" in str(exc.value)

    def test_transcribe_with_language_override(self, mock_whisper_model):
        """Test transcribing with language override."""
        engine = FasterWhisperSTT(
            model_size="tiny",
            device="cpu",
            language="en",  # Instance default
        )

        audio = np.random.randn(16000).astype(np.float32)

        # Override with Spanish
        result = engine.transcribe(audio, language="es")

        # Check that model.transcribe was called with Spanish
        mock_whisper_model.return_value.transcribe.assert_called_once()
        call_kwargs = mock_whisper_model.return_value.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"

    def test_transcribe_with_beam_size_override(self, mock_whisper_model):
        """Test transcribing with beam size override."""
        engine = FasterWhisperSTT(
            model_size="tiny",
            device="cpu",
            beam_size=5,  # Instance default
        )

        audio = np.random.randn(16000).astype(np.float32)

        # Override with beam size 10
        result = engine.transcribe(audio, beam_size=10)

        # Check that model.transcribe was called with beam size 10
        call_kwargs = mock_whisper_model.return_value.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 10

    @pytest.mark.asyncio
    async def test_transcribe_async(self, mock_whisper_model):
        """Test async transcription."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        audio = np.random.randn(16000).astype(np.float32)

        result = await engine.transcribe_async(audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"

    def test_get_stats_no_transcriptions(self, mock_whisper_model):
        """Test getting stats with no transcriptions."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        stats = engine.get_stats()

        assert stats["model_size"] == "tiny"
        assert stats["device"] == "cpu"
        assert stats["transcription_count"] == 0
        assert stats["total_audio_duration"] == 0.0
        assert stats["avg_audio_duration"] == 0.0
        assert stats["real_time_factor"] == 0.0

    def test_get_stats_with_transcriptions(self, mock_whisper_model):
        """Test getting stats after transcriptions."""
        engine = FasterWhisperSTT(model_size="tiny", device="cpu")

        # Do two transcriptions
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(32000).astype(np.float32)

        engine.transcribe(audio1)
        engine.transcribe(audio2)

        stats = engine.get_stats()

        assert stats["transcription_count"] == 2
        assert stats["total_audio_duration"] == 4.0  # 2.0 + 2.0
        assert stats["avg_audio_duration"] == 2.0

    def test_get_model_info(self, mock_whisper_model):
        """Test getting model info."""
        engine = FasterWhisperSTT(
            model_size="small",
            device="cuda",
            compute_type="float16",
            beam_size=7,
            language="fr",
        )

        info = engine.get_model_info()

        assert info["model_size"] == "small"
        assert info["device"] == "cuda"
        assert info["compute_type"] == "float16"
        assert info["beam_size"] == 7
        assert info["language"] == "fr"
        assert info["loaded"] is True


class TestSTTTranscriber:
    """Test STTTranscriber class."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock STT engine."""
        engine = Mock(spec=FasterWhisperSTT)

        # Mock async transcription
        async def mock_transcribe_async(audio, language=None):
            return TranscriptionResult(
                text="Test transcription",
                segments=[TranscriptSegment("Test transcription", 0.0, 1.5, 0.95)],
                language=language or "en",
                duration=1.5,
            )

        engine.transcribe_async = mock_transcribe_async
        engine.get_stats.return_value = {
            "transcription_count": 0,
            "total_audio_duration": 0.0,
        }

        return engine

    def test_create_transcriber(self, mock_engine):
        """Test creating transcriber."""
        transcriber = STTTranscriber(engine=mock_engine, max_concurrent=2)

        assert transcriber.engine == mock_engine
        assert transcriber.max_concurrent == 2
        assert transcriber._queue_size == 0

    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_engine):
        """Test successful transcription."""
        transcriber = STTTranscriber(engine=mock_engine)

        audio = np.random.randn(16000).astype(np.float32)

        result = await transcriber.transcribe(audio, user_id=123)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, mock_engine):
        """Test transcription with language hint."""
        transcriber = STTTranscriber(engine=mock_engine)

        audio = np.random.randn(16000).astype(np.float32)

        result = await transcriber.transcribe(audio, user_id=123, language="es")

        assert result.language == "es"

    @pytest.mark.asyncio
    async def test_transcribe_error_handling(self):
        """Test transcription error handling."""
        # Create engine that raises error
        engine = Mock(spec=FasterWhisperSTT)

        async def mock_error(audio, language=None):
            raise RuntimeError("Transcription failed")

        engine.transcribe_async = mock_error

        transcriber = STTTranscriber(engine=engine)

        audio = np.random.randn(16000).astype(np.float32)

        with pytest.raises(RuntimeError) as exc:
            await transcriber.transcribe(audio, user_id=123)

        assert "Transcription failed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, mock_engine):
        """Test concurrent transcription limit."""
        # Create engine with delay to test queueing
        engine = Mock(spec=FasterWhisperSTT)

        async def mock_delayed_transcribe(audio, language=None):
            await asyncio.sleep(0.1)  # Simulate processing time
            return TranscriptionResult(
                text="Test", segments=[], language="en", duration=1.0
            )

        engine.transcribe_async = mock_delayed_transcribe
        engine.get_stats.return_value = {"transcription_count": 0}

        # Max concurrent = 1
        transcriber = STTTranscriber(engine=engine, max_concurrent=1)

        audio = np.random.randn(16000).astype(np.float32)

        # Start two transcriptions concurrently
        task1 = asyncio.create_task(transcriber.transcribe(audio, user_id=1))
        task2 = asyncio.create_task(transcriber.transcribe(audio, user_id=2))

        # Both should complete successfully (one queued)
        results = await asyncio.gather(task1, task2)

        assert len(results) == 2
        assert all(r.text == "Test" for r in results)

    def test_get_queue_size(self, mock_engine):
        """Test getting queue size."""
        transcriber = STTTranscriber(engine=mock_engine)

        assert transcriber.get_queue_size() == 0

    def test_get_stats(self, mock_engine):
        """Test getting transcriber stats."""
        transcriber = STTTranscriber(engine=mock_engine, max_concurrent=2)

        stats = transcriber.get_stats()

        assert "max_concurrent" in stats
        assert stats["max_concurrent"] == 2
        assert "current_queue_size" in stats

    @pytest.mark.asyncio
    async def test_create_transcriber_convenience(self):
        """Test convenience function for creating transcriber."""
        with patch("server.stt.FasterWhisperSTT") as mock_stt:
            mock_instance = Mock(spec=FasterWhisperSTT)
            mock_stt.return_value = mock_instance

            transcriber = await create_transcriber(
                model_size="tiny", device="cpu", language="en"
            )

            assert isinstance(transcriber, STTTranscriber)
            mock_stt.assert_called_once_with(
                model_size="tiny",
                device="cpu",
                compute_type="float16",
                language="en",
            )


class TestPipelineTranscriber:
    """Test PipelineTranscriber class."""

    @pytest.fixture
    def mock_transcriber(self):
        """Create mock STT transcriber."""
        transcriber = Mock(spec=STTTranscriber)

        # Mock async transcription
        async def mock_transcribe(audio, user_id, language=None):
            return TranscriptionResult(
                text="Pipeline test",
                segments=[TranscriptSegment("Pipeline test", 0.0, 2.0, 0.9)],
                language=language or "en",
                duration=2.0,
            )

        transcriber.transcribe = mock_transcribe
        transcriber.get_stats.return_value = {
            "transcription_count": 0,
            "max_concurrent": 1,
        }

        return transcriber

    def test_create_pipeline_transcriber(self, mock_transcriber):
        """Test creating pipeline transcriber."""
        pipeline = PipelineTranscriber(transcriber=mock_transcriber)

        assert pipeline.transcriber == mock_transcriber
        assert pipeline.transcription_callback is None
        assert pipeline.total_transcriptions == 0
        assert pipeline.total_failures == 0

    @pytest.mark.asyncio
    async def test_process_speech_success(self, mock_transcriber):
        """Test successful speech processing."""
        pipeline = PipelineTranscriber(transcriber=mock_transcriber)

        audio = np.random.randn(16000).astype(np.float32)

        result = await pipeline.process_speech(user_id=123, audio=audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Pipeline test"
        assert pipeline.total_transcriptions == 1
        assert pipeline.total_failures == 0

    @pytest.mark.asyncio
    async def test_process_speech_with_callback(self, mock_transcriber):
        """Test speech processing with callback."""
        callback_called = False
        callback_user_id = None
        callback_result = None

        async def callback(user_id: int, result: TranscriptionResult):
            nonlocal callback_called, callback_user_id, callback_result
            callback_called = True
            callback_user_id = user_id
            callback_result = result

        pipeline = PipelineTranscriber(
            transcriber=mock_transcriber, transcription_callback=callback
        )

        audio = np.random.randn(16000).astype(np.float32)

        result = await pipeline.process_speech(user_id=456, audio=audio)

        assert callback_called
        assert callback_user_id == 456
        assert callback_result.text == "Pipeline test"

    @pytest.mark.asyncio
    async def test_process_speech_error_handling(self):
        """Test error handling in speech processing."""
        # Create transcriber that raises error
        transcriber = Mock(spec=STTTranscriber)

        async def mock_error(audio, user_id, language=None):
            raise RuntimeError("Processing failed")

        transcriber.transcribe = mock_error

        pipeline = PipelineTranscriber(transcriber=transcriber)

        audio = np.random.randn(16000).astype(np.float32)

        # Should return None on error, not raise
        result = await pipeline.process_speech(user_id=123, audio=audio)

        assert result is None
        assert pipeline.total_failures == 1
        assert pipeline.total_transcriptions == 0

    @pytest.mark.asyncio
    async def test_process_speech_with_language(self, mock_transcriber):
        """Test processing with language hint."""
        pipeline = PipelineTranscriber(transcriber=mock_transcriber)

        audio = np.random.randn(16000).astype(np.float32)

        result = await pipeline.process_speech(
            user_id=123, audio=audio, language="fr"
        )

        assert result.language == "fr"

    def test_get_stats(self, mock_transcriber):
        """Test getting pipeline stats."""
        pipeline = PipelineTranscriber(transcriber=mock_transcriber)

        # Manually update stats for testing
        pipeline.total_transcriptions = 10
        pipeline.total_failures = 2

        stats = pipeline.get_stats()

        assert stats["total_transcriptions"] == 10
        assert stats["total_failures"] == 2
        assert stats["success_rate"] == 10 / 12  # 10 / (10 + 2)

    def test_get_stats_no_attempts(self, mock_transcriber):
        """Test stats with no transcription attempts."""
        pipeline = PipelineTranscriber(transcriber=mock_transcriber)

        stats = pipeline.get_stats()

        assert stats["total_transcriptions"] == 0
        assert stats["total_failures"] == 0
        assert stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_create_pipeline_transcriber_convenience(self, mock_transcriber):
        """Test convenience function for creating pipeline transcriber."""
        callback = Mock()

        pipeline = await create_pipeline_transcriber(
            transcriber=mock_transcriber, transcription_callback=callback
        )

        assert isinstance(pipeline, PipelineTranscriber)
        assert pipeline.transcriber == mock_transcriber
        assert pipeline.transcription_callback == callback


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
