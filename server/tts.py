"""Text-to-Speech using Chatterbox TTS (or alternatives).

GPU-accelerated TTS with emotion control and paralinguistic support.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""

    voice_ref_dir: Path = Path("server/voices")
    device: str = "cuda"
    sample_rate: int = 24000  # Common for neural TTS
    emotion_exaggeration: float = 1.0  # 0.0-2.0
    streaming_chunk_size: int = 4800  # ~200ms @ 24kHz
    max_generation_time: float = 10.0  # Timeout for generation


@dataclass
class EmotionTag:
    """Represents an emotion tag in text."""

    tag: str  # e.g., "laugh", "chuckle", "sigh"
    position: int  # Character position in text
    text: str  # Original text with brackets


class ChatterboxTTS:
    """
    Chatterbox TTS engine wrapper.

    Supports emotion control and paralinguistic tags.
    Falls back to stub implementation if not available.
    """

    # Supported emotion tags
    EMOTION_TAGS = {
        "laugh": "laughter",
        "chuckle": "soft laughter",
        "sigh": "exhalation",
        "gasp": "inhalation",
        "whisper": "quiet speech",
        "excited": "high energy",
        "sad": "low energy",
    }

    def __init__(
        self,
        config: TTSConfig,
        voice_references: Dict[str, Path],
    ):
        """
        Initialize Chatterbox TTS engine.

        Args:
            config: TTS configuration
            voice_references: Map of agent_name -> reference audio file
        """
        self.config = config
        self.voice_references = voice_references

        # TTS model (stub - to be replaced with actual Chatterbox)
        self.model = None

        # Load engine
        self._load_engine()

        # Stats
        self.total_generations = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

    def _load_engine(self) -> None:
        """Load TTS engine."""
        try:
            logger.info(
                f"Loading Chatterbox TTS engine "
                f"(device: {self.config.device})"
            )

            # TODO: Replace with actual Chatterbox TTS initialization
            # from chatterbox import ChatterboxModel
            # self.model = ChatterboxModel(
            #     device=self.config.device,
            #     sample_rate=self.config.sample_rate,
            # )

            logger.warning(
                "Chatterbox TTS not available - using stub implementation"
            )
            self.model = "stub"  # Placeholder

        except Exception as e:
            logger.error(f"Failed to load Chatterbox TTS: {e}")
            logger.warning("Using stub implementation")
            self.model = "stub"

    def validate_voice_reference(self, voice_ref_path: Path) -> bool:
        """
        Validate voice reference file.

        Args:
            voice_ref_path: Path to voice reference audio

        Returns:
            True if valid, False otherwise
        """
        if not voice_ref_path.exists():
            logger.error(f"Voice reference not found: {voice_ref_path}")
            return False

        # Check file size (should be at least 100KB for 10s of audio)
        file_size = voice_ref_path.stat().st_size
        if file_size < 100_000:
            logger.warning(
                f"Voice reference may be too short: {voice_ref_path} "
                f"({file_size} bytes)"
            )
            return False

        # TODO: Validate audio format, sample rate, duration
        # import soundfile as sf
        # audio, sr = sf.read(voice_ref_path)
        # if len(audio) / sr < 10.0:
        #     logger.error("Voice reference should be at least 10 seconds")
        #     return False

        logger.info(f"Voice reference validated: {voice_ref_path}")
        return True

    def parse_emotion_tags(self, text: str) -> Tuple[str, List[EmotionTag]]:
        """
        Parse emotion tags from text.

        Args:
            text: Text with emotion tags like "Hello [laugh]"

        Returns:
            Tuple of (cleaned_text, emotion_tags)
        """
        emotion_tags = []
        pattern = r"\[(\w+)\]"

        # Find all emotion tags
        for match in re.finditer(pattern, text):
            tag = match.group(1).lower()
            if tag in self.EMOTION_TAGS:
                emotion_tags.append(
                    EmotionTag(
                        tag=tag,
                        position=match.start(),
                        text=match.group(0),
                    )
                )

        # Remove tags from text
        cleaned_text = re.sub(pattern, "", text)

        # Clean up extra spaces
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text, emotion_tags

    def generate(
        self,
        text: str,
        voice_ref_path: Path,
        emotion_exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice_ref_path: Path to voice reference audio
            emotion_exaggeration: Emotion control (0.0-2.0, None = use default)

        Returns:
            Audio array (float32, sample_rate from config)
        """
        start_time = time.time()

        # Parse emotion tags
        cleaned_text, emotion_tags = self.parse_emotion_tags(text)

        if self.model is None or self.model == "stub":
            logger.warning("Using stub TTS - returning silence")
            # Stub: generate silence
            duration = len(cleaned_text) / 15.0  # ~15 chars/second
            duration = max(1.0, min(duration, 10.0))  # Clamp to 1-10s
            audio = np.zeros(
                int(duration * self.config.sample_rate), dtype=np.float32
            )
        else:
            logger.info(
                f"Generating TTS for: '{cleaned_text[:50]}...' "
                f"({len(emotion_tags)} emotion tags)"
            )

            # TODO: Replace with actual Chatterbox TTS generation
            # audio = self.model.generate(
            #     text=cleaned_text,
            #     voice_ref=voice_ref_path,
            #     emotion_tags=emotion_tags,
            #     emotion_exaggeration=emotion_exaggeration or self.config.emotion_exaggeration,
            # )

            # Stub: generate silence
            duration = len(cleaned_text) / 15.0  # ~15 chars/second
            duration = max(1.0, min(duration, 10.0))  # Clamp to 1-10s
            audio = np.zeros(
                int(duration * self.config.sample_rate), dtype=np.float32
            )

        # Update stats
        processing_time = time.time() - start_time
        duration = len(audio) / self.config.sample_rate
        self.total_generations += 1
        self.total_audio_duration += duration
        self.total_processing_time += processing_time

        logger.info(
            f"Generated {duration:.2f}s audio in {processing_time:.2f}s "
            f"(RTF: {processing_time / duration:.2f})"
        )

        return audio

    async def generate_async(
        self,
        text: str,
        voice_ref_path: Path,
        emotion_exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Async wrapper for generate().

        Args:
            text: Text to synthesize
            voice_ref_path: Voice reference path
            emotion_exaggeration: Emotion control

        Returns:
            Audio array
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            text,
            voice_ref_path,
            emotion_exaggeration,
        )

    async def generate_streaming(
        self,
        text: str,
        voice_ref_path: Path,
        emotion_exaggeration: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Generate speech in streaming chunks.

        Args:
            text: Text to synthesize
            voice_ref_path: Voice reference path
            emotion_exaggeration: Emotion control

        Returns:
            List of audio chunks
        """
        # TODO: Implement actual streaming generation
        # For now, generate full audio and split into chunks
        full_audio = await self.generate_async(
            text, voice_ref_path, emotion_exaggeration
        )

        # Split into chunks
        chunk_size = self.config.streaming_chunk_size
        chunks = []

        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i : i + chunk_size]
            chunks.append(chunk)

        logger.debug(f"Split audio into {len(chunks)} streaming chunks")
        return chunks

    def get_stats(self) -> dict:
        """
        Get TTS statistics.

        Returns:
            Dictionary with stats
        """
        avg_duration = (
            self.total_audio_duration / self.total_generations
            if self.total_generations > 0
            else 0.0
        )

        avg_processing = (
            self.total_processing_time / self.total_generations
            if self.total_generations > 0
            else 0.0
        )

        rtf = (
            avg_processing / avg_duration if avg_duration > 0 else 0.0
        )  # Real-time factor

        return {
            "engine": "Chatterbox TTS (stub)",
            "device": self.config.device,
            "sample_rate": self.config.sample_rate,
            "total_generations": self.total_generations,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "avg_audio_duration": avg_duration,
            "avg_processing_time": avg_processing,
            "real_time_factor": rtf,
        }


class TTSSynthesizer:
    """
    Pipeline TTS synthesizer.

    Handles voice selection, generation, and error handling.
    """

    def __init__(
        self,
        engine: ChatterboxTTS,
        voice_map: Dict[str, Path],
    ):
        """
        Initialize TTS synthesizer.

        Args:
            engine: TTS engine instance
            voice_map: Map of agent_name -> voice reference path
        """
        self.engine = engine
        self.voice_map = voice_map

        # Validate voice references
        for agent, ref_path in voice_map.items():
            if not self.engine.validate_voice_reference(ref_path):
                logger.warning(
                    f"Invalid voice reference for {agent}: {ref_path}"
                )

        # Stats
        self.total_syntheses = 0
        self.total_failures = 0

    async def synthesize(
        self,
        agent: str,
        text: str,
        emotion_exaggeration: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech for an agent.

        Args:
            agent: Agent name
            text: Text to synthesize
            emotion_exaggeration: Emotion control

        Returns:
            Audio array if successful, None on error
        """
        try:
            # Get voice reference
            agent_lower = agent.lower()
            if agent_lower not in self.voice_map:
                logger.error(f"No voice reference for agent: {agent}")
                self.total_failures += 1
                return None

            voice_ref = self.voice_map[agent_lower]

            # Generate audio
            audio = await self.engine.generate_async(
                text=text,
                voice_ref_path=voice_ref,
                emotion_exaggeration=emotion_exaggeration,
            )

            self.total_syntheses += 1

            logger.info(
                f"Synthesized {len(audio) / self.engine.config.sample_rate:.2f}s "
                f"for {agent}: '{text[:50]}...'"
            )

            return audio

        except Exception as e:
            logger.error(f"TTS synthesis failed for {agent}: {e}")
            self.total_failures += 1
            return None

    async def synthesize_streaming(
        self,
        agent: str,
        text: str,
        emotion_exaggeration: Optional[float] = None,
    ) -> Optional[List[np.ndarray]]:
        """
        Synthesize speech in streaming chunks.

        Args:
            agent: Agent name
            text: Text to synthesize
            emotion_exaggeration: Emotion control

        Returns:
            List of audio chunks if successful, None on error
        """
        try:
            agent_lower = agent.lower()
            if agent_lower not in self.voice_map:
                logger.error(f"No voice reference for agent: {agent}")
                self.total_failures += 1
                return None

            voice_ref = self.voice_map[agent_lower]

            # Generate streaming chunks
            chunks = await self.engine.generate_streaming(
                text=text,
                voice_ref_path=voice_ref,
                emotion_exaggeration=emotion_exaggeration,
            )

            self.total_syntheses += 1

            return chunks

        except Exception as e:
            logger.error(f"Streaming TTS failed for {agent}: {e}")
            self.total_failures += 1
            return None

    def get_stats(self) -> dict:
        """
        Get synthesizer statistics.

        Returns:
            Dictionary with stats
        """
        engine_stats = self.engine.get_stats()

        return {
            **engine_stats,
            "total_syntheses": self.total_syntheses,
            "total_failures": self.total_failures,
            "success_rate": (
                self.total_syntheses / (self.total_syntheses + self.total_failures)
                if (self.total_syntheses + self.total_failures) > 0
                else 0.0
            ),
        }


# Convenience function
async def create_tts_synthesizer(
    voice_refs: Dict[str, str],
    device: str = "cuda",
    sample_rate: int = 24000,
) -> TTSSynthesizer:
    """
    Create TTS synthesizer with default settings.

    Args:
        voice_refs: Map of agent_name -> voice reference file path (string)
        device: Device (cuda/cpu)
        sample_rate: Audio sample rate

    Returns:
        TTSSynthesizer instance
    """
    # Convert string paths to Path objects
    voice_map = {agent: Path(path) for agent, path in voice_refs.items()}

    # Create config
    config = TTSConfig(
        device=device,
        sample_rate=sample_rate,
    )

    # Create engine
    engine = ChatterboxTTS(
        config=config,
        voice_references=voice_map,
    )

    # Create synthesizer
    synthesizer = TTSSynthesizer(
        engine=engine,
        voice_map=voice_map,
    )

    return synthesizer
