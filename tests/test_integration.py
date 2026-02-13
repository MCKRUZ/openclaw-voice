"""Integration tests for end-to-end voice processing flows."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from pipeline.audio_buffer import AudioRingBuffer
from pipeline.orchestrator import PipelineConfig, PipelineOrchestrator
from pipeline.relevance_filter import RelevanceClassifier
from pipeline.transcriber import STTTranscriber, TranscriptionResult
from pipeline.transcript_manager import TranscriptManager
from pipeline.turn_detector import SmartTurnDetector
from pipeline.vad import SileroVAD
from server.tts import TTSSynthesizer


class TestEndToEndFlow:
    """Test complete end-to-end voice processing flows."""

    @pytest.fixture
    def mock_components(self):
        """Create all mocked pipeline components."""
        # VAD
        vad = Mock(spec=SileroVAD)
        vad.process_chunk = Mock(return_value=False)  # Default: silence

        # Turn detector
        turn_detector = Mock(spec=SmartTurnDetector)
        turn_detector.detect_async = AsyncMock(return_value=0.8)

        # STT
        transcriber = Mock(spec=STTTranscriber)
        transcriber.transcribe_async = AsyncMock(
            return_value=TranscriptionResult(
                text="Hello Jarvis, what's the weather?",
                language="en",
                segments=[],
                duration=2.0,
                word_count=5,
            )
        )
        transcriber.get_stats = Mock(return_value={})

        # Transcript manager
        transcript_manager = TranscriptManager()

        # Relevance classifier
        relevance_classifier = Mock(spec=RelevanceClassifier)
        relevance_classifier.classify = AsyncMock(return_value=True)
        relevance_classifier.sensitivity = "medium"

        # LLM client
        async def mock_llm(agent, message, context, speaker):
            return f"The weather is sunny today, {speaker}!"

        # TTS
        tts_synthesizer = Mock(spec=TTSSynthesizer)
        tts_synthesizer.synthesize = AsyncMock(
            return_value=np.random.randn(24000).astype(np.float32)
        )
        tts_synthesizer.get_stats = Mock(return_value={})

        # Audio output callback
        audio_output = Mock()

        return {
            "vad": vad,
            "turn_detector": turn_detector,
            "transcriber": transcriber,
            "transcript_manager": transcript_manager,
            "relevance_classifier": relevance_classifier,
            "llm_client": mock_llm,
            "tts_synthesizer": tts_synthesizer,
            "audio_output": audio_output,
        }

    @pytest.fixture
    def orchestrator(self, mock_components):
        """Create orchestrator with mocked components."""
        config = PipelineConfig(
            vad_silence_duration=0.1,
            turn_wait_timeout=0.5,
            stt_timeout=1.0,
            relevance_timeout=1.0,
            llm_timeout=1.0,
            tts_timeout=1.0,
        )

        return PipelineOrchestrator(
            config=config,
            vad=mock_components["vad"],
            turn_detector=mock_components["turn_detector"],
            transcriber=mock_components["transcriber"],
            transcript_manager=mock_components["transcript_manager"],
            relevance_classifier=mock_components["relevance_classifier"],
            llm_client=mock_components["llm_client"],
            tts_synthesizer=mock_components["tts_synthesizer"],
            audio_output_callback=mock_components["audio_output"],
        )

    @pytest.mark.asyncio
    async def test_single_user_full_conversation(
        self, orchestrator, mock_components
    ):
        """Test complete flow: user speaks → bot responds."""
        # Simulate user speaking
        vad = mock_components["vad"]
        vad.process_chunk.side_effect = [
            True,
            True,
            True,  # Speech
            False,
            False,
            False,
            False,
            False,  # Silence
        ]

        # Send audio frames
        for i in range(8):
            audio_frame = np.random.randn(512).astype(np.float32)
            await orchestrator.process_audio_frame(123, "TestUser", audio_frame)
            await asyncio.sleep(0.02)

        # Wait for processing
        await asyncio.sleep(0.8)

        # Verify all stages were called
        assert mock_components["turn_detector"].detect_async.called
        assert mock_components["transcriber"].transcribe_async.called
        assert mock_components["relevance_classifier"].classify.called
        assert mock_components["tts_synthesizer"].synthesize.called
        assert mock_components["audio_output"].called

        # Verify transcript was updated
        context = mock_components["transcript_manager"].get_context()
        assert "TestUser" in context
        assert "Jarvis" in context or len(context) > 0

    @pytest.mark.asyncio
    async def test_multi_user_concurrent_speech(
        self, orchestrator, mock_components
    ):
        """Test multiple users speaking concurrently."""
        vad = mock_components["vad"]
        vad.process_chunk.return_value = True

        # Two users speak simultaneously
        users = [(123, "User1"), (456, "User2")]

        for user_id, user_name in users:
            for _ in range(5):
                audio_frame = np.random.randn(512).astype(np.float32)
                await orchestrator.process_audio_frame(
                    user_id, user_name, audio_frame
                )

        # Both users should have pipelines
        assert len(orchestrator.pipelines) == 2
        assert 123 in orchestrator.pipelines
        assert 456 in orchestrator.pipelines

    @pytest.mark.asyncio
    async def test_barge_in_during_tts(self, orchestrator, mock_components):
        """Test user interrupting bot during TTS playback."""
        # Set up pipeline in RESPONDING state
        from pipeline.orchestrator import PipelineState

        pipeline = orchestrator.get_or_create_pipeline(123, "TestUser")
        pipeline.state = PipelineState.RESPONDING

        # User speaks (barge-in)
        vad = mock_components["vad"]
        vad.process_chunk.return_value = True

        audio_frame = np.random.randn(512).astype(np.float32)
        await orchestrator.process_audio_frame(123, "TestUser", audio_frame)

        # Should transition to LISTENING
        assert pipeline.state == PipelineState.LISTENING
        assert pipeline.total_cancellations == 0  # State change, not task cancel

    @pytest.mark.asyncio
    async def test_relevance_filter_blocks_response(
        self, orchestrator, mock_components
    ):
        """Test that relevance filter prevents unnecessary responses."""
        # Set relevance to always return False
        mock_components["relevance_classifier"].classify.return_value = False

        # Simulate speech
        vad = mock_components["vad"]
        vad.process_chunk.side_effect = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        for i in range(6):
            audio_frame = np.random.randn(512).astype(np.float32)
            await orchestrator.process_audio_frame(123, "TestUser", audio_frame)
            await asyncio.sleep(0.02)

        # Wait for processing
        await asyncio.sleep(0.5)

        # TTS should NOT be called
        assert not mock_components["tts_synthesizer"].synthesize.called

    @pytest.mark.asyncio
    async def test_long_conversation_transcript_window(
        self, orchestrator, mock_components
    ):
        """Test transcript maintains sliding window over long conversation."""
        transcript_manager = mock_components["transcript_manager"]

        # Add many entries (more than max_entries)
        for i in range(30):
            transcript_manager.add_entry(
                speaker=f"User{i % 2}",
                text=f"Message {i}",
            )

        # Should only keep last 20 (default max_entries)
        entries = transcript_manager._entries
        assert len(entries) <= 20

    @pytest.mark.asyncio
    async def test_agent_switching(self, orchestrator):
        """Test switching between agents."""
        assert orchestrator.current_agent == "jarvis"

        orchestrator.set_agent("Sage")
        assert orchestrator.current_agent == "sage"

        orchestrator.set_agent("JARVIS")  # Case insensitive
        assert orchestrator.current_agent == "jarvis"

    @pytest.mark.asyncio
    async def test_sensitivity_adjustment(
        self, orchestrator, mock_components
    ):
        """Test adjusting relevance sensitivity."""
        relevance = mock_components["relevance_classifier"]

        orchestrator.set_sensitivity("low")
        assert relevance.sensitivity == "low"

        orchestrator.set_sensitivity("HIGH")  # Case insensitive
        assert relevance.sensitivity == "high"

    @pytest.mark.asyncio
    async def test_error_recovery_stt_failure(
        self, orchestrator, mock_components
    ):
        """Test graceful handling of STT failure."""
        # STT returns None (failure)
        mock_components["transcriber"].transcribe_async.return_value = None

        # Simulate speech
        vad = mock_components["vad"]
        vad.process_chunk.side_effect = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        for i in range(6):
            audio_frame = np.random.randn(512).astype(np.float32)
            await orchestrator.process_audio_frame(123, "TestUser", audio_frame)
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.5)

        # Pipeline should return to IDLE without crashing
        pipeline = orchestrator.pipelines[123]
        assert pipeline.state.value in ["idle", "listening"]

    @pytest.mark.asyncio
    async def test_latency_tracking(self, orchestrator, mock_components):
        """Test that latency is tracked for each stage."""
        # Simulate full conversation
        vad = mock_components["vad"]
        vad.process_chunk.side_effect = [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]

        for i in range(8):
            audio_frame = np.random.randn(512).astype(np.float32)
            await orchestrator.process_audio_frame(123, "TestUser", audio_frame)
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.8)

        # Check that latencies were tracked
        pipeline = orchestrator.pipelines[123]
        latencies = pipeline.stage_latencies

        # At least some stages should have latency recorded
        assert len(latencies) > 0

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, orchestrator, mock_components):
        """Test statistics aggregation across users."""
        # Create multiple pipelines
        orchestrator.get_or_create_pipeline(123, "User1")
        orchestrator.get_or_create_pipeline(456, "User2")

        # Update stats
        orchestrator.pipelines[123].total_utterances = 5
        orchestrator.pipelines[123].total_responses = 3
        orchestrator.pipelines[456].total_utterances = 7
        orchestrator.pipelines[456].total_responses = 5

        stats = orchestrator.get_stats()

        assert stats["active_users"] == 2
        assert stats["total_utterances"] == 12
        assert stats["total_responses"] == 8

    @pytest.mark.asyncio
    async def test_pipeline_cleanup_on_user_leave(self, orchestrator):
        """Test pipeline cleanup when user leaves."""
        # Create pipeline
        orchestrator.get_or_create_pipeline(123, "TestUser")
        assert 123 in orchestrator.pipelines

        # User leaves
        orchestrator.remove_pipeline(123)
        assert 123 not in orchestrator.pipelines


class TestAPIIntegration:
    """Test FastAPI server integration."""

    @pytest.fixture
    def mock_engines(self):
        """Create mock TTS and STT engines."""
        # TTS
        tts = Mock(spec=TTSSynthesizer)
        tts.engine = Mock()
        tts.engine.config = Mock()
        tts.engine.config.device = "cpu"
        tts.engine.config.sample_rate = 24000
        tts.voice_map = {"jarvis": Path("jarvis.wav")}
        tts.synthesize = AsyncMock(
            return_value=np.random.randn(24000).astype(np.float32)
        )
        tts.get_stats = Mock(return_value={})

        # STT
        stt = Mock(spec=STTTranscriber)
        stt.engine = Mock()
        stt.engine.device = "cpu"
        stt.transcribe_async = AsyncMock(
            return_value=TranscriptionResult(
                text="Test transcription",
                language="en",
                segments=[],
                duration=1.0,
                word_count=2,
            )
        )
        stt.get_stats = Mock(return_value={})

        return {"tts": tts, "stt": stt}

    @pytest.mark.asyncio
    async def test_api_server_initialization(self, mock_engines):
        """Test API server can be initialized."""
        from server.app import create_api_server

        server = create_api_server(
            tts_synthesizer=mock_engines["tts"],
            stt_transcriber=mock_engines["stt"],
        )

        assert server is not None
        assert server.total_tts_requests == 0
        assert server.total_stt_requests == 0

    @pytest.mark.asyncio
    async def test_concurrent_discord_and_api_requests(
        self, orchestrator, mock_components, mock_engines
    ):
        """Test Discord bot and API server can run concurrently."""
        from server.app import create_api_server

        # Create API server
        api_server = create_api_server(
            tts_synthesizer=mock_engines["tts"],
            stt_transcriber=mock_engines["stt"],
        )

        # Simulate Discord request
        vad = mock_components["vad"]
        vad.process_chunk.return_value = True

        audio_frame = np.random.randn(512).astype(np.float32)
        discord_task = asyncio.create_task(
            orchestrator.process_audio_frame(123, "User1", audio_frame)
        )

        # Both should work without interference
        await discord_task

        # Verify both systems operational
        assert 123 in orchestrator.pipelines
        assert api_server.total_tts_requests == 0  # No API calls yet


class TestMemoryLeaks:
    """Test for memory leaks in long-running scenarios."""

    @pytest.mark.asyncio
    async def test_audio_buffer_no_memory_leak(self):
        """Test audio buffer doesn't leak memory."""
        buffer = AudioRingBuffer(duration_seconds=10.0)

        # Write many frames
        for i in range(10000):
            audio = np.random.randn(512).astype(np.float32)
            buffer.write(audio)

        # Buffer should maintain constant size
        # (maxlen enforced by deque)
        assert len(buffer._buffer) <= buffer._buffer.maxlen

    @pytest.mark.asyncio
    async def test_transcript_manager_no_memory_leak(self):
        """Test transcript manager doesn't leak memory."""
        manager = TranscriptManager(max_age_seconds=90.0, max_entries=20)

        # Add many entries
        for i in range(1000):
            manager.add_entry(
                speaker=f"User{i % 5}",
                text=f"Message {i}",
            )

        # Should only keep max_entries
        assert len(manager._entries) <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
