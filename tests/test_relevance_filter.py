"""Unit tests for Relevance Filter."""

import asyncio
import json

import pytest

from pipeline.relevance_filter import (
    PerGuildRelevanceFilter,
    RelevanceFilter,
    RelevanceResult,
    create_relevance_filter,
)


class TestRelevanceResult:
    """Test RelevanceResult dataclass."""

    def test_create_result(self):
        """Test creating a relevance result."""
        result = RelevanceResult(
            should_respond=True,
            confidence=0.95,
            reason="Name mentioned",
            method="fast_path",
            latency_ms=5.2,
        )

        assert result.should_respond is True
        assert result.confidence == 0.95
        assert result.reason == "Name mentioned"
        assert result.method == "fast_path"
        assert result.latency_ms == 5.2


class TestRelevanceFilter:
    """Test RelevanceFilter class."""

    @pytest.fixture
    def filter(self):
        """Create filter instance."""
        return RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
        )

    @pytest.fixture
    def mock_llm_classifier(self):
        """Create mock LLM classifier."""

        async def classifier(prompt: str) -> str:
            # Return a mock response
            return json.dumps({
                "respond": True,
                "confidence": 0.85,
                "reason": "Question detected",
            })

        return classifier

    def test_create_filter(self, filter):
        """Test creating filter."""
        assert filter.agent_name == "Jarvis"
        assert filter.sensitivity == "medium"
        assert filter.total_classifications == 0

    def test_build_name_patterns(self):
        """Test building name patterns."""
        filter = RelevanceFilter(agent_name="Sage")

        patterns = filter._name_patterns

        # Should have multiple patterns
        assert len(patterns) >= 4

    @pytest.mark.asyncio
    async def test_fast_path_name_mention(self, filter):
        """Test fast path with name mention."""
        result = await filter.classify(
            utterance="Hey Jarvis, how are you?",
            speaker="Matt",
        )

        assert result.should_respond is True
        assert result.confidence == 1.0
        assert result.method == "fast_path"
        assert "mentioned" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_fast_path_name_variations(self, filter):
        """Test fast path with various name mentions."""
        test_cases = [
            "jarvis, what do you think?",  # Lowercase
            "JARVIS!",  # Uppercase
            "Hey Jarvis",  # Greeting + name
            "Jarvis?",  # Name with punctuation
            "Hi jarvis how are you",  # No punctuation
        ]

        for utterance in test_cases:
            result = await filter.classify(utterance, speaker="Test")
            assert result.should_respond is True, f"Failed for: {utterance}"
            assert result.method == "fast_path"

    @pytest.mark.asyncio
    async def test_fast_path_no_name_mention(self, filter):
        """Test fast path without name mention."""
        # Should use fast path for low sensitivity
        filter.sensitivity = "low"

        result = await filter.classify(
            utterance="What's the weather like?",
            speaker="Matt",
        )

        assert result.should_respond is False
        assert result.method == "fast_path"
        assert "low sensitivity" in result.reason

    @pytest.mark.asyncio
    async def test_slow_path_with_llm(self, mock_llm_classifier):
        """Test slow path with LLM classifier."""
        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
            llm_classifier=mock_llm_classifier,
        )

        result = await filter.classify(
            utterance="What's the capital of France?",
            speaker="Matt",
            transcript="[Previous conversation]",
        )

        assert result.should_respond is True
        assert result.confidence == 0.85
        assert result.method == "slow_path"

    @pytest.mark.asyncio
    async def test_slow_path_below_threshold(self):
        """Test slow path with confidence below threshold."""

        async def low_confidence_llm(prompt: str) -> str:
            return json.dumps({
                "respond": False,
                "confidence": 0.3,
                "reason": "Casual banter",
            })

        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",  # Threshold 0.75
            llm_classifier=low_confidence_llm,
        )

        result = await filter.classify(
            utterance="lol nice",
            speaker="Matt",
        )

        assert result.should_respond is False
        assert result.confidence == 0.3
        assert "below threshold" in result.reason

    @pytest.mark.asyncio
    async def test_sensitivity_low(self, filter):
        """Test low sensitivity (fast path only)."""
        filter.sensitivity = "low"

        # No name mention
        result = await filter.classify(
            utterance="What do you think?",
            speaker="Matt",
        )

        assert result.should_respond is False
        assert result.method == "fast_path"

        # With name mention
        result = await filter.classify(
            utterance="Jarvis, what do you think?",
            speaker="Matt",
        )

        assert result.should_respond is True
        assert result.method == "fast_path"

    @pytest.mark.asyncio
    async def test_sensitivity_medium(self, mock_llm_classifier):
        """Test medium sensitivity (threshold 0.75)."""
        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
            llm_classifier=mock_llm_classifier,
        )

        result = await filter.classify(
            utterance="What's the weather?",
            speaker="Matt",
        )

        # Mock returns 0.85, above 0.75 threshold
        assert result.should_respond is True

    @pytest.mark.asyncio
    async def test_sensitivity_high(self):
        """Test high sensitivity (threshold 0.5)."""

        async def medium_confidence_llm(prompt: str) -> str:
            return json.dumps({
                "respond": True,
                "confidence": 0.6,
                "reason": "Might be relevant",
            })

        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="high",  # Threshold 0.5
            llm_classifier=medium_confidence_llm,
        )

        result = await filter.classify(
            utterance="Interesting topic",
            speaker="Matt",
        )

        # 0.6 is above 0.5 threshold for high sensitivity
        assert result.should_respond is True

    @pytest.mark.asyncio
    async def test_caching(self, filter):
        """Test result caching."""
        utterance = "Hey Jarvis"

        # First call
        result1 = await filter.classify(utterance, speaker="Matt")
        assert filter.cache_hits == 0

        # Second call - should hit cache
        result2 = await filter.classify(utterance, speaker="Matt")
        assert filter.cache_hits == 1

        # Results should be identical
        assert result1.should_respond == result2.should_respond
        assert result1.confidence == result2.confidence

    @pytest.mark.asyncio
    async def test_cache_normalization(self, filter):
        """Test cache key normalization."""
        # Different whitespace and case
        result1 = await filter.classify("Hey   JARVIS", speaker="Matt")
        result2 = await filter.classify("hey jarvis", speaker="Matt")

        # Should hit cache (normalized to same key)
        assert filter.cache_hits == 1

    @pytest.mark.asyncio
    async def test_llm_timeout(self):
        """Test LLM classification timeout."""

        async def slow_llm(prompt: str) -> str:
            await asyncio.sleep(5.0)  # Longer than timeout
            return json.dumps({"respond": True, "confidence": 0.9})

        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
            llm_classifier=slow_llm,
            slow_path_timeout=0.1,  # Very short timeout
        )

        result = await filter.classify(
            utterance="What's the time?",
            speaker="Matt",
        )

        # Should timeout and fallback
        assert result.should_respond is False
        assert "timeout" in result.reason.lower() or "failed" in result.reason.lower()
        assert filter.slow_path_timeouts == 1

    @pytest.mark.asyncio
    async def test_llm_invalid_json(self):
        """Test LLM returning invalid JSON."""

        async def invalid_json_llm(prompt: str) -> str:
            return "This is not JSON"

        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
            llm_classifier=invalid_json_llm,
        )

        result = await filter.classify(
            utterance="Test",
            speaker="Matt",
        )

        # Should fallback to no response
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_llm_error(self):
        """Test LLM raising an error."""

        async def error_llm(prompt: str) -> str:
            raise RuntimeError("LLM error")

        filter = RelevanceFilter(
            agent_name="Jarvis",
            sensitivity="medium",
            llm_classifier=error_llm,
        )

        result = await filter.classify(
            utterance="Test",
            speaker="Matt",
        )

        # Should fallback to no response
        assert result.should_respond is False

    def test_is_question(self, filter):
        """Test question detection."""
        questions = [
            "What is the weather?",
            "How are you?",
            "Can you help me?",
            "Do you know Python?",
            "Tell me about AI",
        ]

        for q in questions:
            assert filter._is_question(q), f"Failed to detect: {q}"

        non_questions = [
            "That's interesting",
            "I agree",
            "Nice work",
        ]

        for nq in non_questions:
            assert not filter._is_question(nq), f"False positive: {nq}"

    def test_set_sensitivity(self, filter):
        """Test updating sensitivity."""
        filter.set_sensitivity("high")
        assert filter.sensitivity == "high"

        filter.set_sensitivity("low")
        assert filter.sensitivity == "low"

    def test_set_sensitivity_invalid(self, filter):
        """Test setting invalid sensitivity."""
        with pytest.raises(ValueError) as exc:
            filter.set_sensitivity("invalid")

        assert "Invalid sensitivity" in str(exc.value)

    def test_clear_cache(self, filter):
        """Test clearing cache."""
        # Add to cache
        filter._add_to_cache(
            "test",
            RelevanceResult(True, 1.0, "test", "fast_path", 0.0)
        )

        assert len(filter._cache) == 1

        # Clear
        filter.clear_cache()

        assert len(filter._cache) == 0

    def test_get_stats(self, filter):
        """Test getting statistics."""
        stats = filter.get_stats()

        assert stats["agent_name"] == "Jarvis"
        assert stats["sensitivity"] == "medium"
        assert stats["threshold"] == 0.75
        assert stats["total_classifications"] == 0
        assert stats["fast_path_count"] == 0
        assert stats["slow_path_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, filter):
        """Test stats tracking."""
        # Fast path
        await filter.classify("Hey Jarvis", speaker="Matt")

        stats = filter.get_stats()
        assert stats["total_classifications"] == 1
        assert stats["fast_path_count"] == 1

    def test_build_classification_prompt(self, filter):
        """Test building LLM prompt."""
        prompt = filter._build_classification_prompt(
            utterance="What's the weather?",
            speaker="Matt",
            transcript="[Previous conversation]",
        )

        # Check prompt contains key elements
        assert "Jarvis" in prompt
        assert "What's the weather?" in prompt
        assert "Matt" in prompt
        assert "[Previous conversation]" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, filter):
        """Test cache size limit."""
        filter.cache_size = 3

        # Add 5 entries
        for i in range(5):
            await filter.classify(f"Test {i}", speaker="Matt")

        # Should only keep last 3
        assert len(filter._cache) <= 3


class TestPerGuildRelevanceFilter:
    """Test PerGuildRelevanceFilter class."""

    @pytest.fixture
    def manager(self):
        """Create per-guild manager."""
        return PerGuildRelevanceFilter(
            default_agent="Jarvis",
            default_sensitivity="medium",
        )

    def test_create_manager(self, manager):
        """Test creating per-guild manager."""
        assert manager.default_agent == "Jarvis"
        assert manager.default_sensitivity == "medium"

    def test_get_or_create(self, manager):
        """Test getting or creating guild filter."""
        filter = manager.get_or_create(guild_id=123)

        assert isinstance(filter, RelevanceFilter)
        assert filter.agent_name == "Jarvis"
        assert filter.sensitivity == "medium"

        # Getting again should return same instance
        filter2 = manager.get_or_create(guild_id=123)
        assert filter is filter2

    def test_multiple_guilds(self, manager):
        """Test managing multiple guilds."""
        filter1 = manager.get_or_create(guild_id=111)
        filter2 = manager.get_or_create(guild_id=222)

        # Should be different instances
        assert filter1 is not filter2

    def test_get_or_create_with_overrides(self, manager):
        """Test creating with overrides."""
        filter = manager.get_or_create(
            guild_id=123,
            agent_name="Sage",
            sensitivity="high",
        )

        assert filter.agent_name == "Sage"
        assert filter.sensitivity == "high"

    @pytest.mark.asyncio
    async def test_classify(self, manager):
        """Test classifying via per-guild manager."""
        result = await manager.classify(
            guild_id=123,
            utterance="Hey Jarvis",
            speaker="Matt",
        )

        assert result.should_respond is True
        assert result.method == "fast_path"

    def test_set_agent(self, manager):
        """Test setting agent for a guild."""
        manager.set_agent(guild_id=123, agent_name="Sage")

        filter = manager.get_or_create(guild_id=123)
        assert filter.agent_name == "Sage"

    def test_set_sensitivity(self, manager):
        """Test setting sensitivity for a guild."""
        manager.set_sensitivity(guild_id=123, sensitivity="high")

        filter = manager.get_or_create(guild_id=123)
        assert filter.sensitivity == "high"

    def test_remove_guild(self, manager):
        """Test removing guild filter."""
        manager.get_or_create(guild_id=123)
        assert 123 in manager._filters

        manager.remove_guild(guild_id=123)
        assert 123 not in manager._filters

    def test_remove_nonexistent_guild(self, manager):
        """Test removing guild that doesn't exist."""
        # Should not raise error
        manager.remove_guild(guild_id=999)

    @pytest.mark.asyncio
    async def test_get_all_stats(self, manager):
        """Test getting stats for all guilds."""
        # Create filters for two guilds
        await manager.classify(111, "Hey Jarvis", "Matt")
        await manager.classify(222, "Hello Sage", "Jake")

        all_stats = manager.get_all_stats()

        assert 111 in all_stats
        assert 222 in all_stats
        assert all_stats[111]["total_classifications"] >= 1
        assert all_stats[222]["total_classifications"] >= 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_relevance_filter(self):
        """Test creating filter with convenience function."""
        filter = create_relevance_filter(
            agent_name="Sage",
            sensitivity="high",
        )

        assert isinstance(filter, RelevanceFilter)
        assert filter.agent_name == "Sage"
        assert filter.sensitivity == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
