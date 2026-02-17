"""Text-to-Speech using Chatterbox-Turbo engine directly.

Integrated Chatterbox-Turbo TTS with zero-shot voice cloning.
Supports native paralinguistic sounds ([laugh], [sigh], etc.)
"""

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""

    voice_ref_dir: Path = Path("server/voices")
    device: str = "cuda"
    sample_rate: int = 24000
    emotion_exaggeration: float = 1.0  # Maps to temperature (0.0-2.0)
    streaming_chunk_size: int = 4800  # ~200ms @ 24kHz
    max_generation_time: float = 10.0  # Timeout for generation


@dataclass
class EmotionTag:
    """Represents an emotion tag in text."""

    tag: str  # e.g., "laugh", "chuckle", "sigh"
    position: int  # Character position in text
    text: str  # Original text with brackets


# Emotion presets (Turbo uses temperature only)
EMOTION_PRESETS: dict[str, dict] = {
    "neutral":    {"temperature": 0.8},
    "warm":       {"temperature": 0.8},
    "witty":      {"temperature": 0.9},
    "sarcastic":  {"temperature": 0.9},
    "angry":      {"temperature": 0.95},
    "tender":     {"temperature": 0.7},
    "excited":    {"temperature": 0.95},
    "guarded":    {"temperature": 0.7},
    "flirty":     {"temperature": 0.85},
    "protective": {"temperature": 0.85},
}

# Turbo's native paralinguistic tags
_TURBO_TAGS = {"laugh", "sigh", "chuckle", "gasp", "cough"}

# Map action words from various formats to Turbo's native tags
_ACTION_TO_TAG: dict[str, str] = {
    # Sigh variants
    "sigh": "sigh", "sighs": "sigh", "sighing": "sigh",
    # Laugh variants
    "laugh": "laugh", "laughs": "laugh", "laughing": "laugh",
    "giggle": "laugh", "giggles": "laugh", "giggling": "laugh",
    # Chuckle variants
    "chuckle": "chuckle", "chuckles": "chuckle", "chuckling": "chuckle",
    # Gasp variants
    "gasp": "gasp", "gasps": "gasp", "gasping": "gasp",
    # Cough variants
    "cough": "cough", "coughs": "cough", "coughing": "cough",
    # Close approximations mapped to nearest tag
    "groan": "sigh", "groans": "sigh", "groaning": "sigh",
    "scoff": "chuckle", "scoffs": "chuckle", "scoffing": "chuckle",
    "snort": "laugh", "snorts": "laugh", "snorting": "laugh",
    "sob": "sigh", "sobs": "sigh", "sobbing": "sigh",
    "sniff": "sigh", "sniffs": "sigh", "sniffing": "sigh",
    "hum": "sigh", "hums": "sigh", "humming": "sigh",
}

# Patterns to extract action content from markers: *text*, (text), ~text~
_MARKER_PATTERNS = [
    re.compile(r"\*([^*]+)\*"),
    re.compile(r"\(([^)]+)\)"),
    re.compile(r"~([^~]+)~"),
]

# Separate pattern for square brackets
_BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")


def _replace_marker(match: re.Match) -> str:
    """Convert action marker to Turbo paralinguistic tag or strip entirely."""
    inner = match.group(1).strip().lower()
    words = inner.split()

    for word in words:
        clean_word = word.strip(".,!?")
        if clean_word in _ACTION_TO_TAG:
            return f" [{_ACTION_TO_TAG[clean_word]}] "

    # Unknown action - strip to preserve voice clone
    return " "


def _replace_bracket(match: re.Match) -> str:
    """Handle [bracket] markers - pass through Turbo tags, convert others."""
    inner = match.group(1).strip().lower()

    # Already a native Turbo tag - pass through as-is
    if inner in _TURBO_TAGS:
        return match.group(0)

    # Check if it maps to a Turbo tag
    words = inner.split()
    for word in words:
        clean_word = word.strip(".,!?")
        if clean_word in _ACTION_TO_TAG:
            return f" [{_ACTION_TO_TAG[clean_word]}] "

    # Unknown - strip to preserve voice clone
    return " "


def clean_text_for_tts(text: str) -> str:
    """Convert action markers to Turbo paralinguistic tags.

    Strategy:
    - Known sounds (*sighs*, (laughs), ~gasps~) -> Turbo tags ([sigh], [laugh], [gasp])
    - [sigh], [laugh], etc. -> passed through directly (already Turbo format)
    - Unknown actions -> stripped entirely (preserves voice clone quality)
    """
    cleaned = text

    # Process *text*, (text), ~text~ markers
    for pattern in _MARKER_PATTERNS:
        cleaned = pattern.sub(_replace_marker, cleaned)

    # Process [text] markers (preserve native Turbo tags)
    cleaned = _BRACKET_PATTERN.sub(_replace_bracket, cleaned)

    # Replace newlines with spaces
    cleaned = cleaned.replace("\n", " ")

    # Strip emojis and other non-speech unicode
    cleaned = re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"   # symbols & pictographs
        r"\U0001F680-\U0001F6FF"   # transport & map
        r"\U0001F1E0-\U0001F1FF"   # flags
        r"\U00002702-\U000027B0"   # dingbats
        r"\U0000FE00-\U0000FE0F"   # variation selectors
        r"\U0000200D"              # zero-width joiner
        r"\U000025A0-\U000025FF"   # geometric shapes
        r"\U00002600-\U000026FF"   # misc symbols
        r"\U00002B50-\U00002B55"   # stars
        r"]+", "", cleaned
    )

    # Collapse multiple spaces
    cleaned = re.sub(r"  +", " ", cleaned)

    return cleaned.strip()


class ChatterboxTTS:
    """
    Chatterbox-Turbo TTS engine with zero-shot voice cloning.

    Supports emotion control and paralinguistic tags natively.
    """

    def __init__(
        self,
        config: TTSConfig,
        voice_references: Dict[str, Path],
    ):
        """
        Initialize Chatterbox-Turbo TTS engine.

        Args:
            config: TTS configuration
            voice_references: Map of agent_name -> reference audio file
        """
        self.config = config
        self.voice_references = voice_references

        # Lazy-load model on first use
        self._model = None

        logger.info(f"Initialized Chatterbox-Turbo TTS engine (device: {config.device})")

        # Stats
        self.total_generations = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

    @property
    def model(self):
        """Lazy-load the TTS model."""
        if self._model is None:
            logger.info(f"Loading Chatterbox-Turbo on {self.config.device}...")
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.config.device)
            logger.info(f"Model loaded. Sample rate: {self._model.sr}Hz")
        return self._model

    def validate_voice_reference(self, voice_ref_path: Path) -> bool:
        """
        Validate voice reference audio file.

        Args:
            voice_ref_path: Path to voice reference audio

        Returns:
            True if valid, False otherwise
        """
        if not voice_ref_path.exists():
            logger.warning(f"Voice reference not found: {voice_ref_path}")
            return False

        if voice_ref_path.suffix not in [".wav", ".flac", ".mp3"]:
            logger.warning(f"Unsupported audio format: {voice_ref_path.suffix}")
            return False

        return True

    def parse_emotion_tags(self, text: str) -> Tuple[str, List[EmotionTag]]:
        """
        Parse emotion tags from text.

        Args:
            text: Text with emotion tags like "Hello [laugh]"

        Returns:
            Tuple of (cleaned_text, emotion_tags_list)
        """
        emotion_tags = []
        pattern = r"\[(\w+)\]"

        # Find all emotion tags for logging
        for match in re.finditer(pattern, text):
            tag = match.group(1).lower()
            if tag in _TURBO_TAGS:
                emotion_tags.append(
                    EmotionTag(
                        tag=tag,
                        position=match.start(),
                        text=match.group(0),
                    )
                )

        # Clean text (converts action markers, preserves Turbo tags)
        cleaned_text = clean_text_for_tts(text)

        return cleaned_text, emotion_tags

    async def generate_async(
        self,
        text: str,
        voice_ref_path: Path,
        emotion_exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize (with emotion tags like [laugh])
            voice_ref_path: Voice reference path
            emotion_exaggeration: Temperature (0.0-2.0, default from config)

        Returns:
            Audio array (float32, 24kHz sample rate)
        """
        start_time = time.time()

        # Parse and clean text
        cleaned_text, emotion_tags = self.parse_emotion_tags(text)

        logger.info(
            f"Generating TTS for '{voice_ref_path.stem}': '{text[:50]}...' "
            f"({len(emotion_tags)} emotion tags)"
        )

        if not cleaned_text:
            logger.warning("No speakable text after cleaning, returning silence")
            duration = 1.0
            # Return 16kHz audio (processing format)
            audio = np.zeros(
                int(duration * 16000), dtype=np.float32
            )
            return audio

        try:
            # Get temperature (emotion exaggeration)
            temperature = emotion_exaggeration if emotion_exaggeration is not None else self.config.emotion_exaggeration

            # Generate audio (run in thread to not block event loop)
            import asyncio
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: self.model.generate(
                    cleaned_text,
                    audio_prompt_path=str(voice_ref_path),
                    temperature=temperature,
                )
            )

            # Convert to numpy float32
            audio = wav.squeeze().cpu().numpy()

            # Resample from 24kHz (Chatterbox) to 16kHz (processing format)
            # This is required for Discord audio bridge compatibility
            from scipy import signal as scipy_signal
            target_samples = int(len(audio) * 16000 / 24000)
            audio = scipy_signal.resample(audio, target_samples).astype(np.float32)

            # Update stats
            processing_time = time.time() - start_time
            duration = len(audio) / 16000  # Now at 16kHz
            self.total_generations += 1
            self.total_audio_duration += duration
            self.total_processing_time += processing_time

            logger.info(
                f"Generated {duration:.2f}s audio in {processing_time:.2f}s "
                f"(RTF: {processing_time / duration:.2f})"
            )

            return audio

        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            # Return silence on error (16kHz processing format)
            duration = 2.0
            audio = np.zeros(
                int(duration * 16000), dtype=np.float32
            )
            return audio

    def generate(
        self,
        text: str,
        voice_ref_path: Path,
        emotion_exaggeration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synchronous wrapper for generate_async.

        Args:
            text: Text to synthesize
            voice_ref_path: Voice reference path
            emotion_exaggeration: Emotion control

        Returns:
            Audio array
        """
        import asyncio
        # Since Chatterbox-Turbo is synchronous, we can call directly
        return asyncio.run(self.generate_async(text, voice_ref_path, emotion_exaggeration))

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
        # Generate full audio
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
            "engine": f"Chatterbox-Turbo (local)",
            "device": self.config.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "sample_rate": self.config.sample_rate,
            "total_generations": self.total_generations,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "avg_audio_duration": avg_duration,
            "avg_processing_time": avg_processing,
            "real_time_factor": rtf,
        }

    async def close(self):
        """Cleanup resources."""
        # Nothing to close for local engine
        pass


class TTSSynthesizer:
    """
    Pipeline TTS synthesizer.

    Handles voice selection, generation, and error handling.
    Includes phrase caching for common responses.
    """

    # Common phrases to pre-generate for each agent
    COMMON_PHRASES = {
        "jarvis": [
            "Yes, sir.",
            "Right away, sir.",
            "At your service, sir.",
            "Of course, sir.",
            "Certainly, sir.",
            "One moment, sir.",
            "Let me check.",
            "Good question.",
            "I'm on it.",
            "Understood.",
            "Very good, sir.",
            "As you wish, sir.",
            "I'll take care of that.",
            "Allow me.",
            "Indeed, sir.",
        ],
        "sage": [
            "Yes.",
            "I understand.",
            "Let me consider that.",
            "Indeed.",
            "Certainly.",
            "Of course.",
            "Good question.",
            "Let me think.",
            "I see.",
            "Interesting.",
            "Very well.",
            "Allow me to explain.",
        ],
    }

    def __init__(
        self,
        engine: ChatterboxTTS,
        voice_map: Dict[str, Path],
        enable_cache: bool = True,
    ):
        """
        Initialize TTS synthesizer.

        Args:
            engine: TTS engine instance
            voice_map: Map of agent_name -> voice reference path
            enable_cache: Enable phrase caching (default: True)
        """
        self.engine = engine
        self.voice_map = voice_map
        self.enable_cache = enable_cache

        # Validate voice references
        for agent, ref_path in voice_map.items():
            if not self.engine.validate_voice_reference(ref_path):
                logger.warning(
                    f"Invalid voice reference for {agent}: {ref_path}"
                )

        # Phrase cache: (agent, normalized_text) -> audio
        self.phrase_cache: Dict[tuple[str, str], np.ndarray] = {}

        # Stats
        self.total_syntheses = 0
        self.total_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _normalize_text_for_cache(self, text: str) -> str:
        """
        Normalize text for cache key matching.

        Strips whitespace and punctuation for fuzzy matching.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Remove leading/trailing whitespace
        normalized = text.strip()
        # Convert to lowercase
        normalized = normalized.lower()
        # Remove trailing punctuation
        normalized = normalized.rstrip('.!?,;:')
        return normalized

    async def synthesize(
        self,
        agent: str,
        text: str,
        emotion_exaggeration: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech for an agent.

        Checks cache first for common phrases.

        Args:
            agent: Agent name
            text: Text to synthesize
            emotion_exaggeration: Emotion control (temperature)

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

            # Check cache if enabled
            if self.enable_cache:
                cache_key = (agent_lower, self._normalize_text_for_cache(text))
                if cache_key in self.phrase_cache:
                    self.cache_hits += 1
                    logger.info(
                        f"Cache hit for {agent}: '{text}' "
                        f"(hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses):.1%})"
                    )
                    return self.phrase_cache[cache_key].copy()

                self.cache_misses += 1

            # Generate audio
            audio = await self.engine.generate_async(
                text=text,
                voice_ref_path=voice_ref,
                emotion_exaggeration=emotion_exaggeration,
            )

            self.total_syntheses += 1

            logger.info(
                f"Synthesized {len(audio) / 16000:.2f}s "
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

    async def warmup(self) -> None:
        """
        Warmup TTS engine and pre-generate common phrases.

        Call this at startup to cache common responses.
        """
        if not self.enable_cache:
            logger.info("Cache disabled, skipping warmup")
            return

        logger.info("Warming up TTS engine and pre-generating common phrases...")
        start_time = time.time()

        total_phrases = 0
        for agent, phrases in self.COMMON_PHRASES.items():
            agent_lower = agent.lower()

            # Skip if agent not in voice map
            if agent_lower not in self.voice_map:
                logger.warning(f"Skipping warmup for {agent}: no voice reference")
                continue

            voice_ref = self.voice_map[agent_lower]

            logger.info(f"Pre-generating {len(phrases)} phrases for {agent}...")

            for phrase in phrases:
                try:
                    # Generate audio
                    audio = await self.engine.generate_async(
                        text=phrase,
                        voice_ref_path=voice_ref,
                        emotion_exaggeration=None,  # Use default
                    )

                    # Cache it
                    cache_key = (agent_lower, self._normalize_text_for_cache(phrase))
                    self.phrase_cache[cache_key] = audio

                    total_phrases += 1
                    logger.debug(f"Cached phrase for {agent}: '{phrase}'")

                except Exception as e:
                    logger.warning(f"Failed to cache phrase '{phrase}' for {agent}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Warmup complete: cached {total_phrases} phrases in {elapsed:.1f}s "
            f"({total_phrases / elapsed:.1f} phrases/sec)"
        )

    def get_stats(self) -> dict:
        """
        Get synthesizer statistics.

        Returns:
            Dictionary with stats
        """
        engine_stats = self.engine.get_stats()

        cache_stats = {
            "cache_enabled": self.enable_cache,
            "cache_size": len(self.phrase_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
        }

        return {
            **engine_stats,
            "total_syntheses": self.total_syntheses,
            "total_failures": self.total_failures,
            "success_rate": (
                self.total_syntheses / (self.total_syntheses + self.total_failures)
                if (self.total_syntheses + self.total_failures) > 0
                else 0.0
            ),
            **cache_stats,
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
        device: Device (cuda or cpu)
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
