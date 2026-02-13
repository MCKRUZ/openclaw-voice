"""Unit tests for FastAPI Server."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import VoiceAPIServer, create_api_server
from server.stt import STTTranscriber, TranscriptionResult
from server.tts import TTSSynthesizer


class TestVoiceAPIServer:
    """Test VoiceAPIServer class."""

    @pytest.fixture
    def mock_tts_synthesizer(self):
        """Create mock TTS synthesizer."""
        synthesizer = Mock(spec=TTSSynthesizer)

        # Mock engine config
        synthesizer.engine = Mock()
        synthesizer.engine.config = Mock()
        synthesizer.engine.config.device = "cpu"
        synthesizer.engine.config.sample_rate = 24000

        # Mock voice map
        synthesizer.voice_map = {"jarvis": Path("jarvis.wav"), "sage": Path("sage.wav")}

        # Mock synthesize
        synthesizer.synthesize = AsyncMock(
            return_value=np.random.randn(24000).astype(np.float32)  # 1 second
        )

        # Mock stats
        synthesizer.get_stats = Mock(
            return_value={
                "total_syntheses": 10,
                "total_failures": 0,
            }
        )

        return synthesizer

    @pytest.fixture
    def mock_stt_transcriber(self):
        """Create mock STT transcriber."""
        transcriber = Mock(spec=STTTranscriber)

        # Mock engine
        transcriber.engine = Mock()
        transcriber.engine.device = "cpu"

        # Mock transcribe
        transcriber.transcribe_async = AsyncMock(
            return_value=TranscriptionResult(
                text="Test transcription",
                language="en",
                segments=[],
                duration=1.0,
                word_count=2,
            )
        )

        # Mock stats
        transcriber.get_stats = Mock(
            return_value={
                "total_transcriptions": 5,
                "total_failures": 0,
            }
        )

        return transcriber

    @pytest.fixture
    def api_server(self, mock_tts_synthesizer, mock_stt_transcriber):
        """Create API server instance."""
        return VoiceAPIServer(
            tts_synthesizer=mock_tts_synthesizer,
            stt_transcriber=mock_stt_transcriber,
        )

    @pytest.fixture
    def client(self, api_server):
        """Create test client."""
        return TestClient(api_server.app)

    def test_create_api_server(self, api_server):
        """Test creating API server."""
        assert api_server.total_tts_requests == 0
        assert api_server.total_stt_requests == 0
        assert api_server.total_errors == 0

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Jarvis Voice API"
        assert "endpoints" in data

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_health_check_with_gpu(
        self, mock_gpu_props, mock_cuda_available, client
    ):
        """Test health check with GPU available."""
        mock_cuda_available.return_value = True
        mock_gpu_props.return_value = Mock(total_memory=32 * 1e9)  # 32GB

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["gpu"]["available"] is True
        assert data["gpu"]["memory_gb"] == 32.0
        assert "models" in data
        assert data["uptime"] > 0

    @patch("torch.cuda.is_available")
    def test_health_check_without_gpu(self, mock_cuda_available, client):
        """Test health check without GPU."""
        mock_cuda_available.return_value = False

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["gpu"]["available"] is False

    def test_tts_endpoint_wav_format(self, client, mock_tts_synthesizer):
        """Test TTS endpoint with WAV format."""
        request_data = {
            "model": "chatterbox",
            "input": "Hello, this is a test.",
            "voice": "jarvis",
            "response_format": "wav",
        }

        response = client.post("/v1/audio/speech", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0

        # Verify TTS was called
        assert mock_tts_synthesizer.synthesize.called

    def test_tts_endpoint_pcm_format(self, client, mock_tts_synthesizer):
        """Test TTS endpoint with PCM format."""
        request_data = {
            "input": "Test PCM",
            "voice": "sage",
            "response_format": "pcm",
        }

        response = client.post("/v1/audio/speech", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) > 0

    def test_tts_endpoint_invalid_voice(self, client):
        """Test TTS endpoint with invalid voice."""
        request_data = {
            "input": "Test",
            "voice": "invalid_voice",
            "response_format": "wav",
        }

        response = client.post("/v1/audio/speech", json=request_data)

        assert response.status_code == 400
        assert "Invalid voice" in response.json()["detail"]

    def test_tts_endpoint_synthesis_failure(
        self, client, mock_tts_synthesizer
    ):
        """Test TTS endpoint when synthesis fails."""
        mock_tts_synthesizer.synthesize.return_value = None

        request_data = {
            "input": "Test",
            "voice": "jarvis",
            "response_format": "wav",
        }

        response = client.post("/v1/audio/speech", json=request_data)

        assert response.status_code == 500
        assert "TTS generation failed" in response.json()["detail"]

    def test_stt_endpoint_success(self, client, mock_stt_transcriber):
        """Test STT endpoint with successful transcription."""
        # Create test audio file
        audio = np.random.randn(16000).astype(np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 16000, format="WAV")
        audio_buffer.seek(0)

        files = {"file": ("test.wav", audio_buffer, "audio/wav")}
        data = {"model": "whisper-1"}

        response = client.post("/v1/audio/transcriptions", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        assert "text" in result
        assert result["text"] == "Test transcription"

        # Verify STT was called
        assert mock_stt_transcriber.transcribe_async.called

    def test_stt_endpoint_with_language(self, client, mock_stt_transcriber):
        """Test STT endpoint with language hint."""
        audio = np.random.randn(16000).astype(np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 16000, format="WAV")
        audio_buffer.seek(0)

        files = {"file": ("test.wav", audio_buffer, "audio/wav")}
        data = {"model": "whisper-1", "language": "en"}

        response = client.post("/v1/audio/transcriptions", files=files, data=data)

        assert response.status_code == 200

    def test_stt_endpoint_stereo_audio(self, client, mock_stt_transcriber):
        """Test STT endpoint with stereo audio (should convert to mono)."""
        # Create stereo audio
        audio = np.random.randn(16000, 2).astype(np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 16000, format="WAV")
        audio_buffer.seek(0)

        files = {"file": ("test_stereo.wav", audio_buffer, "audio/wav")}
        data = {"model": "whisper-1"}

        response = client.post("/v1/audio/transcriptions", files=files, data=data)

        assert response.status_code == 200

    def test_stt_endpoint_transcription_failure(
        self, client, mock_stt_transcriber
    ):
        """Test STT endpoint when transcription fails."""
        mock_stt_transcriber.transcribe_async.return_value = None

        audio = np.random.randn(16000).astype(np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 16000, format="WAV")
        audio_buffer.seek(0)

        files = {"file": ("test.wav", audio_buffer, "audio/wav")}
        data = {"model": "whisper-1"}

        response = client.post("/v1/audio/transcriptions", files=files, data=data)

        assert response.status_code == 500

    def test_convert_audio_pcm(self, api_server):
        """Test audio conversion to PCM."""
        audio = np.random.randn(1000).astype(np.float32)

        audio_bytes = api_server._convert_audio(audio, 16000, "pcm")

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) == 1000 * 2  # int16 = 2 bytes per sample

    def test_convert_audio_wav(self, api_server):
        """Test audio conversion to WAV."""
        audio = np.random.randn(1000).astype(np.float32)

        audio_bytes = api_server._convert_audio(audio, 16000, "wav")

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000 * 2  # WAV has header

    def test_convert_audio_invalid_format(self, api_server):
        """Test audio conversion with invalid format."""
        audio = np.random.randn(1000).astype(np.float32)

        with pytest.raises(ValueError):
            api_server._convert_audio(audio, 16000, "invalid")

    def test_get_stats(self, api_server):
        """Test getting API server stats."""
        stats = api_server.get_stats()

        assert "uptime" in stats
        assert "total_tts_requests" in stats
        assert "total_stt_requests" in stats
        assert "total_errors" in stats
        assert "tts_stats" in stats
        assert "stt_stats" in stats

    def test_stats_updated_after_requests(
        self, client, mock_tts_synthesizer, mock_stt_transcriber, api_server
    ):
        """Test that stats are updated after requests."""
        # Initial stats
        assert api_server.total_tts_requests == 0

        # TTS request
        request_data = {
            "input": "Test",
            "voice": "jarvis",
            "response_format": "wav",
        }
        client.post("/v1/audio/speech", json=request_data)

        assert api_server.total_tts_requests == 1

        # STT request
        audio = np.random.randn(16000).astype(np.float32)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 16000, format="WAV")
        audio_buffer.seek(0)

        files = {"file": ("test.wav", audio_buffer, "audio/wav")}
        client.post("/v1/audio/transcriptions", files=files)

        assert api_server.total_stt_requests == 1

    def test_error_count_updated(self, client, api_server):
        """Test that error count is updated on failures."""
        assert api_server.total_errors == 0

        # Invalid voice (should increment error count)
        request_data = {
            "input": "Test",
            "voice": "invalid",
            "response_format": "wav",
        }
        client.post("/v1/audio/speech", json=request_data)

        assert api_server.total_errors == 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_api_server(self):
        """Test creating API server with convenience function."""
        mock_tts = Mock(spec=TTSSynthesizer)
        mock_tts.engine = Mock()
        mock_tts.engine.config = Mock()
        mock_tts.engine.config.device = "cpu"
        mock_tts.engine.config.sample_rate = 24000
        mock_tts.voice_map = {"jarvis": Path("jarvis.wav")}
        mock_tts.get_stats = Mock(return_value={})

        mock_stt = Mock(spec=STTTranscriber)
        mock_stt.engine = Mock()
        mock_stt.engine.device = "cpu"
        mock_stt.get_stats = Mock(return_value={})

        server = create_api_server(
            tts_synthesizer=mock_tts,
            stt_transcriber=mock_stt,
        )

        assert isinstance(server, VoiceAPIServer)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
