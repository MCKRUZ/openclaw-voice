"""Pipeline stage for speech-to-text transcription.

Integrates STT engine into the audio processing pipeline.
"""

import asyncio
from typing import Callable, Optional

import numpy as np

from server.stt import STTTranscriber, TranscriptionResult
from utils.logging import get_logger

logger = get_logger(__name__)


class PipelineTranscriber:
    """
    Pipeline transcription stage.

    Receives speech segments from turn detector and produces transcripts.
    """

    def __init__(
        self,
        transcriber: STTTranscriber,
        transcription_callback: Optional[
            Callable[[int, TranscriptionResult], None]
        ] = None,
    ):
        """
        Initialize pipeline transcriber.

        Args:
            transcriber: STT transcriber instance
            transcription_callback: Async callback when transcription completes
        """
        self.transcriber = transcriber
        self.transcription_callback = transcription_callback

        # Stats
        self.total_transcriptions = 0
        self.total_failures = 0

    async def process_speech(
        self,
        user_id: int,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> Optional[TranscriptionResult]:
        """
        Process speech segment and transcribe.

        Args:
            user_id: User ID
            audio: Audio segment (float32, mono, 16kHz)
            language: Optional language hint

        Returns:
            TranscriptionResult if successful, None on error
        """
        try:
            # Transcribe
            result = await self.transcriber.transcribe(
                audio=audio,
                user_id=user_id,
                language=language,
            )

            # Update stats
            self.total_transcriptions += 1

            # Invoke callback
            if self.transcription_callback:
                await self.transcription_callback(user_id, result)

            return result

        except Exception as e:
            logger.error(f"Failed to transcribe for user {user_id}: {e}")
            self.total_failures += 1
            return None

    def get_stats(self) -> dict:
        """
        Get transcription statistics.

        Returns:
            Dictionary with stats
        """
        transcriber_stats = self.transcriber.get_stats()

        return {
            **transcriber_stats,
            "total_transcriptions": self.total_transcriptions,
            "total_failures": self.total_failures,
            "success_rate": (
                self.total_transcriptions
                / (self.total_transcriptions + self.total_failures)
                if (self.total_transcriptions + self.total_failures) > 0
                else 0.0
            ),
        }


async def create_pipeline_transcriber(
    transcriber: STTTranscriber,
    transcription_callback: Optional[
        Callable[[int, TranscriptionResult], None]
    ] = None,
) -> PipelineTranscriber:
    """
    Create pipeline transcriber.

    Args:
        transcriber: STT transcriber instance
        transcription_callback: Async callback for transcriptions

    Returns:
        PipelineTranscriber instance
    """
    return PipelineTranscriber(
        transcriber=transcriber,
        transcription_callback=transcription_callback,
    )
