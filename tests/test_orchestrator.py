"""Unit tests for Pipeline Orchestrator."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from pipeline.audio_buffer import AudioRingBuffer
from pipeline.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    PipelineState,
    UserPipeline,
)
from pipeline.relevance_filter import RelevanceClassifier
from pipeline.transcriber import STTTranscriber, TranscriptionResult
from pipeline.transcript_manager import TranscriptManager
from pipeline.turn_detector import SmartTurnDetector
from pipeline.vad import SileroVAD
from server.tts import TTSSynthesizer


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_create_config(self):
        """Test creating config with defaults."""
        config = PipelineConfig()

        assert config.vad_silence_duration == 0.3
        assert config.turn_wait_timeout == 3.0
        assert config.turn_completion_threshold == 0.7
        assert config.max_concurrent_users == 5

    def test_create_config_with_values(self):
        """Test creating config with custom values."""
        config = PipelineConfig(
            vad_silence_duration=0.5,
            turn_wait_timeout=2.0,
            max_concurrent_users=10,
        )

        assert config.vad_silence_duration == 0.5
        assert config.turn_wait_timeout == 2.0
        assert config.max_concurrent_users == 10


class TestUserPipeline:
    """Test UserPipeline dataclass."""

    def test_create_pipeline(self):
        """Test creating user pipeline."""
        pipeline = UserPipeline(user_id=123, user_name="TestUser")

        assert pipeline.user_id == 123
        assert pipeline.user_name == "TestUser"
        assert pipeline.state == PipelineState.IDLE
        assert isinstance(pipeline.audio_buffer, AudioRingBuffer)
        assert pipeline.total_utterances == 0


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PipelineConfig(
            vad_silence_duration=0.1,  # Short for testing
            turn_wait_timeout=1.0,
            stt_timeout=1.0,
            relevance_timeout=1.0,
            llm_timeout=1.0,
            tts_timeout=1.0,
        )

    @pytest.fixture
    def mock_vad(self):
        """Create mock VAD."""
        vad = Mock(spec=SileroVAD)
        vad.process_chunk = Mock(return_value=False)  # Default: silence
        return vad

    @pytest.fixture
    def mock_turn_detector(self):
        """Create mock turn detector."""
        detector = Mock(spec=SmartTurnDetector)
        detector.detect_async = AsyncMock(return_value=0.8)  # Complete
        return detector

    @pytest.fixture
    def mock_transcriber(self):
        """Create mock transcriber."""
        transcriber = Mock(spec=STTTranscriber)
        transcriber.transcribe_async = AsyncMock(
            return_value=TranscriptionResult(
                text="Test transcription",
                language="en",
                segments=[],
                duration=1.0,
                word_count=2,
            )
        )
        return transcriber

    @pytest.fixture
    def mock_transcript_manager(self):
        """Create mock transcript manager."""
        manager = Mock(spec=TranscriptManager)
        manager.add_entry = Mock()
        manager.get_context = Mock(
            return_value="[8:00:00 PM] TestUser: Previous message"
        )
        return manager

    @pytest.fixture
    def mock_relevance_classifier(self):
        """Create mock relevance classifier."""
        classifier = Mock(spec=RelevanceClassifier)
        classifier.classify = AsyncMock(return_value=True)  # Respond
        classifier.sensitivity = "medium"
        return classifier

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""

        async def llm_client(agent, message, context, speaker):
            return f"Mock response to: {message}"

        return llm_client

    @pytest.fixture
    def mock_tts_synthesizer(self):
        """Create mock TTS synthesizer."""
        synthesizer = Mock(spec=TTSSynthesizer)
        synthesizer.synthesize = AsyncMock(
            return_value=np.zeros(16000, dtype=np.float32)  # 1 second
        )
        return synthesizer

    @pytest.fixture
    def mock_audio_output(self):
        """Create mock audio output callback."""
        return Mock()

    @pytest.fixture
    def orchestrator(
        self,
        config,
        mock_vad,
        mock_turn_detector,
        mock_transcriber,
        mock_transcript_manager,
        mock_relevance_classifier,
        mock_llm_client,
        mock_tts_synthesizer,
        mock_audio_output,
    ):
        """Create orchestrator instance."""
        return PipelineOrchestrator(
            config=config,
            vad=mock_vad,
            turn_detector=mock_turn_detector,
            transcriber=mock_transcriber,
            transcript_manager=mock_transcript_manager,
            relevance_classifier=mock_relevance_classifier,
            llm_client=mock_llm_client,
            tts_synthesizer=mock_tts_synthesizer,
            audio_output_callback=mock_audio_output,
        )

    def test_create_orchestrator(self, orchestrator):
        """Test creating orchestrator."""
        assert orchestrator.current_agent == "jarvis"
        assert len(orchestrator.pipelines) == 0
        assert orchestrator.total_pipeline_runs == 0

    def test_get_or_create_pipeline(self, orchestrator):
        """Test getting or creating pipeline."""
        pipeline = orchestrator.get_or_create_pipeline(123, "TestUser")

        assert pipeline.user_id == 123
        assert pipeline.user_name == "TestUser"
        assert 123 in orchestrator.pipelines

        # Get again - should return same instance
        pipeline2 = orchestrator.get_or_create_pipeline(123, "TestUser")
        assert pipeline is pipeline2

    def test_remove_pipeline(self, orchestrator):
        """Test removing pipeline."""
        orchestrator.get_or_create_pipeline(123, "TestUser")
        assert 123 in orchestrator.pipelines

        orchestrator.remove_pipeline(123)
        assert 123 not in orchestrator.pipelines

    @pytest.mark.asyncio
    async def test_process_audio_frame_silence(
        self, orchestrator, mock_vad
    ):
        """Test processing audio frame with silence."""
        audio_frame = np.zeros(512, dtype=np.float32)

        mock_vad.process_chunk.return_value = False  # Silence

        await orchestrator.process_audio_frame(123, "TestUser", audio_frame)

        pipeline = orchestrator.pipelines[123]
        assert pipeline.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_process_audio_frame_speech_start(
        self, orchestrator, mock_vad
    ):
        """Test processing audio frame with speech start."""
        audio_frame = np.zeros(512, dtype=np.float32)

        mock_vad.process_chunk.return_value = True  # Speech

        await orchestrator.process_audio_frame(123, "TestUser", audio_frame)

        pipeline = orchestrator.pipelines[123]
        assert pipeline.state == PipelineState.LISTENING
        assert pipeline.speech_start_time is not None

    @pytest.mark.asyncio
    async def test_speech_end_triggers_processing(
        self, orchestrator, mock_vad, mock_turn_detector
    ):
        """Test that speech end triggers turn detection."""
        # First frame: speech
        mock_vad.process_chunk.return_value = True
        audio_frame = np.random.randn(512).astype(np.float32)

        await orchestrator.process_audio_frame(123, "TestUser", audio_frame)

        pipeline = orchestrator.pipelines[123]
        assert pipeline.state == PipelineState.LISTENING

        # Silence frames to trigger speech end
        mock_vad.process_chunk.return_value = False

        for _ in range(10):  # Enough frames for silence duration
            await orchestrator.process_audio_frame(
                123, "TestUser", np.zeros(512, dtype=np.float32)
            )
            await asyncio.sleep(0.01)  # Small delay

        # Wait for processing to start
        await asyncio.sleep(0.1)

        # Should have triggered turn detection
        assert pipeline.state in [
            PipelineState.TURN_WAIT,
            PipelineState.PROCESSING,
            PipelineState.IDLE,
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        orchestrator,
        mock_vad,
        mock_turn_detector,
        mock_transcriber,
        mock_relevance_classifier,
        mock_llm_client,
        mock_tts_synthesizer,
        mock_audio_output,
    ):
        """Test full successful pipeline run."""
        # Simulate speech
        mock_vad.process_chunk.side_effect = [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        audio_frames = [
            np.random.randn(512).astype(np.float32) for _ in range(10)
        ]

        for frame in audio_frames:
            await orchestrator.process_audio_frame(123, "TestUser", frame)
            await asyncio.sleep(0.01)

        # Wait for pipeline to complete
        await asyncio.sleep(0.5)

        # Check that all stages were called
        assert mock_turn_detector.detect_async.called
        assert mock_transcriber.transcribe_async.called
        assert mock_relevance_classifier.classify.called
        assert mock_tts_synthesizer.synthesize.called
        assert mock_audio_output.called

    @pytest.mark.asyncio
    async def test_relevance_filter_blocks_response(
        self,
        orchestrator,
        mock_vad,
        mock_relevance_classifier,
        mock_tts_synthesizer,
    ):
        """Test that relevance filter blocks response."""
        # Relevance filter says don't respond
        mock_relevance_classifier.classify.return_value = False

        # Simulate speech
        mock_vad.process_chunk.side_effect = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        audio_frames = [
            np.random.randn(512).astype(np.float32) for _ in range(6)
        ]

        for frame in audio_frames:
            await orchestrator.process_audio_frame(123, "TestUser", frame)
            await asyncio.sleep(0.01)

        # Wait for processing
        await asyncio.sleep(0.3)

        # TTS should NOT be called
        assert not mock_tts_synthesizer.synthesize.called

    @pytest.mark.asyncio
    async def test_barge_in_cancels_response(
        self, orchestrator, mock_vad
    ):
        """Test that user speaking during response cancels it."""
        # Create pipeline in RESPONDING state
        pipeline = orchestrator.get_or_create_pipeline(123, "TestUser")
        pipeline.state = PipelineState.RESPONDING

        # User speaks (barge-in)
        mock_vad.process_chunk.return_value = True
        audio_frame = np.random.randn(512).astype(np.float32)

        await orchestrator.process_audio_frame(123, "TestUser", audio_frame)

        # Should transition to LISTENING
        assert pipeline.state == PipelineState.LISTENING

    @pytest.mark.asyncio
    async def test_empty_transcription_returns_to_idle(
        self, orchestrator, mock_vad, mock_transcriber
    ):
        """Test that empty transcription returns to idle."""
        # Empty transcription
        mock_transcriber.transcribe_async.return_value = TranscriptionResult(
            text="",
            language="en",
            segments=[],
            duration=0.0,
            word_count=0,
        )

        # Simulate speech
        mock_vad.process_chunk.side_effect = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        audio_frames = [
            np.random.randn(512).astype(np.float32) for _ in range(6)
        ]

        for frame in audio_frames:
            await orchestrator.process_audio_frame(123, "TestUser", frame)
            await asyncio.sleep(0.01)

        # Wait for processing
        await asyncio.sleep(0.3)

        pipeline = orchestrator.pipelines[123]
        assert pipeline.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_stt_timeout_handled(
        self, orchestrator, mock_vad, mock_transcriber
    ):
        """Test STT timeout is handled gracefully."""

        # STT takes too long
        async def slow_transcribe(audio):
            await asyncio.sleep(5.0)  # Longer than timeout
            return TranscriptionResult(
                text="Too slow", language="en", segments=[], duration=1.0, word_count=2
            )

        mock_transcriber.transcribe_async.side_effect = slow_transcribe

        # Simulate speech
        mock_vad.process_chunk.side_effect = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        audio_frames = [
            np.random.randn(512).astype(np.float32) for _ in range(6)
        ]

        for frame in audio_frames:
            await orchestrator.process_audio_frame(123, "TestUser", frame)
            await asyncio.sleep(0.01)

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Should have returned to idle after timeout
        pipeline = orchestrator.pipelines[123]
        assert pipeline.state == PipelineState.IDLE
        assert orchestrator.total_errors > 0

    def test_set_agent(self, orchestrator):
        """Test setting active agent."""
        orchestrator.set_agent("Sage")
        assert orchestrator.current_agent == "sage"

    def test_set_sensitivity(self, orchestrator, mock_relevance_classifier):
        """Test setting relevance sensitivity."""
        orchestrator.set_sensitivity("High")
        assert mock_relevance_classifier.sensitivity == "high"

    def test_get_stats_initial(self, orchestrator):
        """Test getting stats initially."""
        stats = orchestrator.get_stats()

        assert stats["active_users"] == 0
        assert stats["current_agent"] == "jarvis"
        assert stats["total_utterances"] == 0
        assert stats["total_responses"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_processing(
        self, orchestrator, mock_vad
    ):
        """Test stats after processing."""
        # Create some activity
        orchestrator.get_or_create_pipeline(123, "User1")
        orchestrator.get_or_create_pipeline(456, "User2")

        pipeline1 = orchestrator.pipelines[123]
        pipeline1.total_utterances = 5
        pipeline1.total_responses = 3
        pipeline1.stage_latencies = {
            "stt": 0.3,
            "relevance": 0.1,
            "llm": 2.0,
            "tts": 0.5,
            "total": 3.0,
        }

        stats = orchestrator.get_stats()

        assert stats["active_users"] == 2
        assert stats["total_utterances"] == 5
        assert stats["total_responses"] == 3
        assert "avg_stt_latency" in stats

    def test_get_user_stats(self, orchestrator):
        """Test getting stats for specific user."""
        pipeline = orchestrator.get_or_create_pipeline(123, "TestUser")
        pipeline.total_utterances = 10
        pipeline.total_responses = 7

        stats = orchestrator.get_user_stats(123)

        assert stats is not None
        assert stats["user_id"] == 123
        assert stats["user_name"] == "TestUser"
        assert stats["total_utterances"] == 10
        assert stats["total_responses"] == 7

    def test_get_user_stats_not_found(self, orchestrator):
        """Test getting stats for non-existent user."""
        stats = orchestrator.get_user_stats(999)
        assert stats is None

    @pytest.mark.asyncio
    async def test_concurrent_users(
        self, orchestrator, mock_vad
    ):
        """Test handling multiple users concurrently."""
        # Simulate two users speaking simultaneously
        mock_vad.process_chunk.return_value = True

        users = [(123, "User1"), (456, "User2"), (789, "User3")]

        # Send audio from multiple users
        for user_id, user_name in users:
            audio_frame = np.random.randn(512).astype(np.float32)
            await orchestrator.process_audio_frame(
                user_id, user_name, audio_frame
            )

        assert len(orchestrator.pipelines) == 3

        # All should be in LISTENING state
        for user_id, _ in users:
            assert orchestrator.pipelines[user_id].state == PipelineState.LISTENING


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
