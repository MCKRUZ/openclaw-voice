"""Unit tests for Text-to-Speech engine."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from server.tts import (
    ChatterboxTTS,
    EmotionTag,
    TTSConfig,
    TTSSynthesizer,
    create_tts_synthesizer,
)


class TestTTSConfig:
    """Test TTSConfig dataclass."""

    def test_create_config(self):
        """Test creating config with defaults."""
        config = TTSConfig()

        assert config.voice_ref_dir == Path("server/voices")
        assert config.device == "cuda"
        assert config.sample_rate == 24000
        assert config.emotion_exaggeration == 1.0

    def test_create_config_with_values(self):
        """Test creating config with custom values."""
        config = TTSConfig(
            device="cpu",
            sample_rate=16000,
            emotion_exaggeration=0.5,
        )

        assert config.device == "cpu"
        assert config.sample_rate == 16000
        assert config.emotion_exaggeration == 0.5


class TestEmotionTag:
    """Test EmotionTag dataclass."""

    def test_create_emotion_tag(self):
        """Test creating emotion tag."""
        tag = EmotionTag(
            tag="laugh",
            position=10,
            text="[laugh]",
        )

        assert tag.tag == "laugh"
        assert tag.position == 10
        assert tag.text == "[laugh]"


class TestChatterboxTTS:
    """Test ChatterboxTTS class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TTSConfig(device="cpu", sample_rate=16000)

    @pytest.fixture
    def voice_refs(self, tmp_path):
        """Create temporary voice reference files."""
        # Create dummy audio files
        jarvis_ref = tmp_path / "jarvis.wav"
        sage_ref = tmp_path / "sage.wav"

        # Write some data (at least 100KB)
        jarvis_ref.write_bytes(b"\x00" * 150000)
        sage_ref.write_bytes(b"\x00" * 150000)

        return {
            "jarvis": jarvis_ref,
            "sage": sage_ref,
        }

    def test_create_engine(self, config, voice_refs):
        """Test creating TTS engine."""
        engine = ChatterboxTTS(
            config=config,
            voice_references=voice_refs,
        )

        assert engine.config == config
        assert engine.voice_references == voice_refs
        assert engine.total_generations == 0

    def test_emotion_tags_constant(self):
        """Test emotion tags are defined."""
        assert "laugh" in ChatterboxTTS.EMOTION_TAGS
        assert "chuckle" in ChatterboxTTS.EMOTION_TAGS
        assert "sigh" in ChatterboxTTS.EMOTION_TAGS

    def test_validate_voice_reference_exists(self, config, voice_refs):
        """Test validating voice reference that exists."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        valid = engine.validate_voice_reference(voice_refs["jarvis"])
        assert valid is True

    def test_validate_voice_reference_not_found(self, config, voice_refs):
        """Test validating voice reference that doesn't exist."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        valid = engine.validate_voice_reference(Path("nonexistent.wav"))
        assert valid is False

    def test_validate_voice_reference_too_small(self, config, voice_refs, tmp_path):
        """Test validating voice reference that's too small."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        # Create tiny file
        small_file = tmp_path / "small.wav"
        small_file.write_bytes(b"\x00" * 1000)  # Only 1KB

        valid = engine.validate_voice_reference(small_file)
        assert valid is False  # Too small

    def test_parse_emotion_tags_none(self, config, voice_refs):
        """Test parsing text with no emotion tags."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        text = "Hello, how are you?"
        cleaned, tags = engine.parse_emotion_tags(text)

        assert cleaned == "Hello, how are you?"
        assert len(tags) == 0

    def test_parse_emotion_tags_single(self, config, voice_refs):
        """Test parsing text with single emotion tag."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        text = "That's funny [laugh]"
        cleaned, tags = engine.parse_emotion_tags(text)

        assert cleaned == "That's funny"
        assert len(tags) == 1
        assert tags[0].tag == "laugh"

    def test_parse_emotion_tags_multiple(self, config, voice_refs):
        """Test parsing text with multiple emotion tags."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        text = "Oh no [sigh] I can't believe it [gasp]"
        cleaned, tags = engine.parse_emotion_tags(text)

        assert cleaned == "Oh no I can't believe it"
        assert len(tags) == 2
        assert tags[0].tag == "sigh"
        assert tags[1].tag == "gasp"

    def test_parse_emotion_tags_unknown(self, config, voice_refs):
        """Test parsing text with unknown emotion tag."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        text = "Hello [unknown] there"
        cleaned, tags = engine.parse_emotion_tags(text)

        # Unknown tags are removed but not added to emotion_tags
        assert cleaned == "Hello there"
        assert len(tags) == 0

    def test_parse_emotion_tags_case_insensitive(self, config, voice_refs):
        """Test that emotion tag parsing is case-insensitive."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        text = "Wow [LAUGH] amazing"
        cleaned, tags = engine.parse_emotion_tags(text)

        assert cleaned == "Wow amazing"
        assert len(tags) == 1
        assert tags[0].tag == "laugh"  # Normalized to lowercase

    def test_generate_stub(self, config, voice_refs):
        """Test generating audio with stub."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        audio = engine.generate(
            text="Hello, how are you?",
            voice_ref_path=voice_refs["jarvis"],
        )

        # Stub returns silence
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_generate_with_emotion_tags(self, config, voice_refs):
        """Test generating audio with emotion tags."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        audio = engine.generate(
            text="That's amazing [laugh]",
            voice_ref_path=voice_refs["jarvis"],
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_generate_updates_stats(self, config, voice_refs):
        """Test that generation updates stats."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        assert engine.total_generations == 0

        engine.generate(
            text="Test",
            voice_ref_path=voice_refs["jarvis"],
        )

        assert engine.total_generations == 1
        assert engine.total_audio_duration > 0

    @pytest.mark.asyncio
    async def test_generate_async(self, config, voice_refs):
        """Test async generation."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        audio = await engine.generate_async(
            text="Hello world",
            voice_ref_path=voice_refs["jarvis"],
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_generate_streaming(self, config, voice_refs):
        """Test streaming generation."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        chunks = await engine.generate_streaming(
            text="This is a longer piece of text for testing streaming generation.",
            voice_ref_path=voice_refs["jarvis"],
        )

        # Should return list of chunks
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

    def test_get_stats_initial(self, config, voice_refs):
        """Test getting stats initially."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        stats = engine.get_stats()

        assert stats["engine"] == "Chatterbox TTS (stub)"
        assert stats["device"] == "cpu"
        assert stats["sample_rate"] == 16000
        assert stats["total_generations"] == 0

    def test_get_stats_after_generation(self, config, voice_refs):
        """Test getting stats after generation."""
        engine = ChatterboxTTS(config=config, voice_references=voice_refs)

        engine.generate("Test", voice_refs["jarvis"])

        stats = engine.get_stats()

        assert stats["total_generations"] == 1
        assert stats["avg_audio_duration"] > 0
        assert stats["real_time_factor"] >= 0


class TestTTSSynthesizer:
    """Test TTSSynthesizer class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TTSConfig(device="cpu", sample_rate=16000)

    @pytest.fixture
    def voice_map(self, tmp_path):
        """Create voice map with temp files."""
        jarvis_ref = tmp_path / "jarvis.wav"
        sage_ref = tmp_path / "sage.wav"

        jarvis_ref.write_bytes(b"\x00" * 150000)
        sage_ref.write_bytes(b"\x00" * 150000)

        return {
            "jarvis": jarvis_ref,
            "sage": sage_ref,
        }

    @pytest.fixture
    def synthesizer(self, config, voice_map):
        """Create synthesizer instance."""
        engine = ChatterboxTTS(config=config, voice_references=voice_map)
        return TTSSynthesizer(engine=engine, voice_map=voice_map)

    def test_create_synthesizer(self, synthesizer):
        """Test creating synthesizer."""
        assert synthesizer.total_syntheses == 0
        assert synthesizer.total_failures == 0

    @pytest.mark.asyncio
    async def test_synthesize_jarvis(self, synthesizer):
        """Test synthesizing for Jarvis."""
        audio = await synthesizer.synthesize(
            agent="Jarvis",
            text="Hello, I am Jarvis",
        )

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert synthesizer.total_syntheses == 1

    @pytest.mark.asyncio
    async def test_synthesize_sage(self, synthesizer):
        """Test synthesizing for Sage."""
        audio = await synthesizer.synthesize(
            agent="sage",
            text="Greetings, I am Sage",
        )

        assert audio is not None
        assert isinstance(audio, np.ndarray)

    @pytest.mark.asyncio
    async def test_synthesize_invalid_agent(self, synthesizer):
        """Test synthesizing for invalid agent."""
        audio = await synthesizer.synthesize(
            agent="invalid",
            text="Test",
        )

        assert audio is None
        assert synthesizer.total_failures == 1

    @pytest.mark.asyncio
    async def test_synthesize_with_emotion(self, synthesizer):
        """Test synthesizing with emotion exaggeration."""
        audio = await synthesizer.synthesize(
            agent="jarvis",
            text="That's amazing [laugh]",
            emotion_exaggeration=1.5,
        )

        assert audio is not None

    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, synthesizer):
        """Test streaming synthesis."""
        chunks = await synthesizer.synthesize_streaming(
            agent="jarvis",
            text="This is a test of streaming synthesis.",
        )

        assert chunks is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_synthesize_streaming_invalid_agent(self, synthesizer):
        """Test streaming with invalid agent."""
        chunks = await synthesizer.synthesize_streaming(
            agent="invalid",
            text="Test",
        )

        assert chunks is None
        assert synthesizer.total_failures == 1

    def test_get_stats(self, synthesizer):
        """Test getting synthesizer stats."""
        stats = synthesizer.get_stats()

        assert "total_syntheses" in stats
        assert "total_failures" in stats
        assert "success_rate" in stats
        assert stats["success_rate"] == 0.0  # No syntheses yet

    @pytest.mark.asyncio
    async def test_get_stats_after_synthesis(self, synthesizer):
        """Test stats after synthesis."""
        await synthesizer.synthesize("jarvis", "Test")

        stats = synthesizer.get_stats()

        assert stats["total_syntheses"] == 1
        assert stats["success_rate"] == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_create_tts_synthesizer(self, tmp_path):
        """Test creating synthesizer with convenience function."""
        # Create dummy voice files
        jarvis_ref = tmp_path / "jarvis.wav"
        sage_ref = tmp_path / "sage.wav"

        jarvis_ref.write_bytes(b"\x00" * 150000)
        sage_ref.write_bytes(b"\x00" * 150000)

        voice_refs = {
            "jarvis": str(jarvis_ref),
            "sage": str(sage_ref),
        }

        synthesizer = await create_tts_synthesizer(
            voice_refs=voice_refs,
            device="cpu",
            sample_rate=16000,
        )

        assert isinstance(synthesizer, TTSSynthesizer)
        assert synthesizer.engine.config.device == "cpu"
        assert synthesizer.engine.config.sample_rate == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
