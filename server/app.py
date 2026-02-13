"""FastAPI Server - OpenAI-compatible TTS/STT API.

Provides HTTP endpoints for:
- Text-to-Speech (OpenAI /v1/audio/speech compatible)
- Speech-to-Text (OpenAI /v1/audio/transcriptions compatible)
- Health checks and status

Shares STT and TTS engines with Discord bot for efficiency.
"""

import io
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from server.stt import FasterWhisperSTT, STTTranscriber
from server.tts import ChatterboxTTS, TTSSynthesizer
from utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""

    model: str = Field(
        default="chatterbox",
        description="TTS model to use (ignored, using configured model)",
    )
    input: str = Field(..., description="Text to synthesize", max_length=4000)
    voice: str = Field(
        ..., description="Voice to use (jarvis, sage, or configured voices)"
    )
    response_format: Literal["pcm", "wav", "mp3"] = Field(
        default="wav", description="Audio format"
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Playback speed (not supported)"
    )


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models: dict
    gpu: dict
    uptime: float


# ============================================================================
# FastAPI Application
# ============================================================================


class VoiceAPIServer:
    """
    Voice API server.

    Provides OpenAI-compatible TTS and STT endpoints.
    Shares engines with Discord bot for efficiency.
    """

    def __init__(
        self,
        tts_synthesizer: TTSSynthesizer,
        stt_transcriber: STTTranscriber,
    ):
        """
        Initialize API server.

        Args:
            tts_synthesizer: TTS synthesizer instance
            stt_transcriber: STT transcriber instance
        """
        self.tts_synthesizer = tts_synthesizer
        self.stt_transcriber = stt_transcriber
        self.start_time = time.time()

        # Create FastAPI app
        self.app = FastAPI(
            title="Jarvis Voice API",
            description="OpenAI-compatible TTS/STT API",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on security needs
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

        # Stats
        self.total_tts_requests = 0
        self.total_stt_requests = 0
        self.total_errors = 0

        logger.info("Voice API server initialized")

    def _register_routes(self) -> None:
        """Register API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return await self._health_check()

        @self.app.post("/v1/audio/speech")
        async def create_speech(request: TTSRequest):
            """
            OpenAI-compatible TTS endpoint.

            Generate speech from text.
            """
            return await self._create_speech(request)

        @self.app.post(
            "/v1/audio/transcriptions", response_model=TranscriptionResponse
        )
        async def create_transcription(
            file: UploadFile = File(...),
            model: str = Form(default="whisper-1"),
            language: Optional[str] = Form(default=None),
            prompt: Optional[str] = Form(default=None),
            response_format: str = Form(default="json"),
            temperature: float = Form(default=0.0),
        ):
            """
            OpenAI-compatible STT endpoint.

            Transcribe audio to text.
            """
            return await self._create_transcription(
                file=file,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Jarvis Voice API",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "tts": "/v1/audio/speech",
                    "stt": "/v1/audio/transcriptions",
                },
            }

    async def _health_check(self) -> HealthResponse:
        """
        Health check.

        Returns:
            Health status
        """
        try:
            # Check GPU availability
            import torch

            gpu_available = torch.cuda.is_available()
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_available
                else 0
            )

            return HealthResponse(
                status="ok",
                models={
                    "tts": self.tts_synthesizer.engine.config.device,
                    "stt": self.stt_transcriber.engine.device,
                },
                gpu={
                    "available": gpu_available,
                    "memory_gb": round(gpu_memory, 2),
                },
                uptime=time.time() - self.start_time,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="degraded",
                models={"tts": "unknown", "stt": "unknown"},
                gpu={"available": False, "memory_gb": 0},
                uptime=time.time() - self.start_time,
            )

    async def _create_speech(self, request: TTSRequest) -> Response:
        """
        Generate speech from text.

        Args:
            request: TTS request

        Returns:
            Audio response
        """
        try:
            logger.info(
                f"TTS request: voice={request.voice}, "
                f"format={request.response_format}, "
                f"text='{request.input[:50]}...'"
            )

            # Validate voice
            voice_lower = request.voice.lower()
            if voice_lower not in self.tts_synthesizer.voice_map:
                available_voices = ", ".join(
                    self.tts_synthesizer.voice_map.keys()
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid voice '{request.voice}'. "
                    f"Available: {available_voices}",
                )

            # Generate audio
            audio = await self.tts_synthesizer.synthesize(
                agent=voice_lower, text=request.input
            )

            if audio is None:
                raise HTTPException(
                    status_code=500, detail="TTS generation failed"
                )

            # Convert to requested format
            audio_bytes = self._convert_audio(
                audio=audio,
                sample_rate=self.tts_synthesizer.engine.config.sample_rate,
                format=request.response_format,
            )

            # Determine content type
            content_type = {
                "pcm": "audio/pcm",
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
            }[request.response_format]

            self.total_tts_requests += 1

            return Response(content=audio_bytes, media_type=content_type)

        except HTTPException:
            self.total_errors += 1
            raise

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            self.total_errors += 1
            raise HTTPException(status_code=500, detail=str(e))

    async def _create_transcription(
        self,
        file: UploadFile,
        model: str,
        language: Optional[str],
        prompt: Optional[str],
        response_format: str,
        temperature: float,
    ) -> TranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            file: Audio file
            model: Model name (ignored)
            language: Language hint
            prompt: Prompt for context
            response_format: Response format (json only supported)
            temperature: Temperature (ignored)

        Returns:
            Transcription response
        """
        try:
            logger.info(
                f"STT request: filename={file.filename}, "
                f"content_type={file.content_type}"
            )

            # Read audio file
            audio_bytes = await file.read()

            # Load audio with soundfile
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Convert to float32
            audio = audio.astype(np.float32)

            # Resample if needed (STT expects 16kHz)
            if sample_rate != 16000:
                from scipy import signal

                audio = signal.resample(
                    audio, int(len(audio) * 16000 / sample_rate)
                )

            # Transcribe
            result = await self.stt_transcriber.transcribe_async(audio)

            if not result or not result.text:
                raise HTTPException(
                    status_code=500, detail="Transcription failed"
                )

            self.total_stt_requests += 1

            return TranscriptionResponse(text=result.text)

        except HTTPException:
            self.total_errors += 1
            raise

        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            self.total_errors += 1
            raise HTTPException(status_code=500, detail=str(e))

    def _convert_audio(
        self, audio: np.ndarray, sample_rate: int, format: str
    ) -> bytes:
        """
        Convert audio to requested format.

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate
            format: Target format (pcm, wav, mp3)

        Returns:
            Audio bytes
        """
        if format == "pcm":
            # Convert to int16 PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            return audio_int16.tobytes()

        elif format == "wav":
            # Write WAV file
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read()

        elif format == "mp3":
            # MP3 encoding requires additional library (pydub, ffmpeg)
            # For now, return WAV and document MP3 needs ffmpeg
            logger.warning("MP3 format not fully supported, returning WAV")
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read()

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_stats(self) -> dict:
        """
        Get API server statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "uptime": time.time() - self.start_time,
            "total_tts_requests": self.total_tts_requests,
            "total_stt_requests": self.total_stt_requests,
            "total_errors": self.total_errors,
            "tts_stats": self.tts_synthesizer.get_stats(),
            "stt_stats": self.stt_transcriber.get_stats(),
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_api_server(
    tts_synthesizer: TTSSynthesizer,
    stt_transcriber: STTTranscriber,
) -> VoiceAPIServer:
    """
    Create API server with default settings.

    Args:
        tts_synthesizer: TTS synthesizer instance
        stt_transcriber: STT transcriber instance

    Returns:
        VoiceAPIServer instance
    """
    return VoiceAPIServer(
        tts_synthesizer=tts_synthesizer,
        stt_transcriber=stt_transcriber,
    )
