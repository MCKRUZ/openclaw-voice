"""Relevance filter for determining when bot should respond.

Two-tier system:
1. Fast path: keyword matching (name mentions)
2. Slow path: LLM classification for ambiguous cases
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RelevanceResult:
    """Result of relevance classification."""

    should_respond: bool
    confidence: float  # 0.0-1.0
    reason: str
    method: str  # "fast_path" or "slow_path"
    latency_ms: float


class RelevanceFilter:
    """
    Determines if bot should respond to an utterance.

    Uses two-tier system:
    - Fast path: keyword matching for name mentions
    - Slow path: LLM classification for context-dependent decisions
    """

    # Sensitivity thresholds
    SENSITIVITY_THRESHOLDS = {
        "low": 1.0,  # Fast path only (always >1.0, so slow path never used)
        "medium": 0.75,  # LLM confidence must be >= 0.75
        "high": 0.5,  # LLM confidence must be >= 0.5
    }

    def __init__(
        self,
        agent_name: str,
        sensitivity: str = "medium",
        llm_classifier=None,
        cache_size: int = 100,
        slow_path_timeout: float = 2.0,
    ):
        """
        Initialize relevance filter.

        Args:
            agent_name: Name of agent (e.g., "Jarvis", "Sage")
            sensitivity: Sensitivity level ("low", "medium", "high")
            llm_classifier: Optional LLM classifier (async callable)
            cache_size: Number of recent classifications to cache
            slow_path_timeout: Timeout for LLM classification (seconds)
        """
        self.agent_name = agent_name
        self.sensitivity = sensitivity
        self.llm_classifier = llm_classifier
        self.cache_size = cache_size
        self.slow_path_timeout = slow_path_timeout

        # Name patterns for fast path
        self._name_patterns = self._build_name_patterns(agent_name)

        # Question patterns
        self._question_patterns = [
            r"\b(what|where|when|why|who|how|can|could|would|should|do|does|did|is|are|was|were)\b.*\?",
            r"\b(tell me|show me|explain|help|assist)\b",
            r"\b(do you know|can you|would you|could you)\b",
        ]

        # Cache for recent classifications (utterance -> result)
        self._cache: Dict[str, RelevanceResult] = {}

        # Stats
        self.total_classifications = 0
        self.fast_path_count = 0
        self.slow_path_count = 0
        self.cache_hits = 0
        self.slow_path_timeouts = 0

    def _build_name_patterns(self, agent_name: str) -> list[re.Pattern]:
        """
        Build regex patterns for name matching.

        Args:
            agent_name: Agent name (e.g., "Jarvis")

        Returns:
            List of compiled regex patterns
        """
        name_lower = agent_name.lower()

        patterns = [
            # Direct name mention
            re.compile(rf"\b{re.escape(name_lower)}\b", re.IGNORECASE),
            # Hey/Hi + name
            re.compile(rf"\b(hey|hi|hello|yo)\s+{re.escape(name_lower)}\b", re.IGNORECASE),
            # Name at start of sentence
            re.compile(rf"^{re.escape(name_lower)}\b", re.IGNORECASE),
            # Name with punctuation
            re.compile(rf"\b{re.escape(name_lower)}[,!?]", re.IGNORECASE),
        ]

        return patterns

    def _check_fast_path(self, utterance: str) -> Optional[RelevanceResult]:
        """
        Check fast path (keyword matching).

        Args:
            utterance: User's utterance

        Returns:
            RelevanceResult if fast path matched, None otherwise
        """
        start_time = time.time()

        # Check for name mentions
        for pattern in self._name_patterns:
            if pattern.search(utterance):
                latency_ms = (time.time() - start_time) * 1000

                logger.debug(
                    f"Fast path: name mention detected in: '{utterance[:50]}...'"
                )

                return RelevanceResult(
                    should_respond=True,
                    confidence=1.0,
                    reason=f"{self.agent_name} was mentioned by name",
                    method="fast_path",
                    latency_ms=latency_ms,
                )

        # No fast path match
        return None

    def _is_question(self, utterance: str) -> bool:
        """
        Check if utterance is a question.

        Args:
            utterance: User's utterance

        Returns:
            True if likely a question
        """
        # Check question mark
        if "?" in utterance:
            return True

        # Check question patterns
        for pattern in self._question_patterns:
            if re.search(pattern, utterance, re.IGNORECASE):
                return True

        return False

    def _build_classification_prompt(
        self, utterance: str, speaker: str, transcript: str
    ) -> str:
        """
        Build prompt for LLM classification.

        Args:
            utterance: Latest utterance
            speaker: Speaker name
            transcript: Recent conversation context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are deciding whether an AI assistant named {self.agent_name} should speak in a voice conversation. {self.agent_name} is a participant in a Discord voice channel.

{self.agent_name} should respond when:
- Directly addressed by name
- Asked a question (even if not by name) that they can answer
- A factual correction is warranted
- They can add genuine value to the topic being discussed
- The conversation is in their domain of expertise

{self.agent_name} should stay SILENT when:
- Casual banter between humans
- Someone else has already answered
- The topic doesn't need AI input
- Speaking would interrupt the flow
- The response would just be "I agree" or "interesting"

Recent conversation:
{transcript}

Latest utterance by {speaker}:
"{utterance}"

Should {self.agent_name} respond? Reply with ONLY a JSON object:
{{"respond": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        return prompt

    async def _classify_with_llm(
        self, utterance: str, speaker: str, transcript: str
    ) -> Optional[RelevanceResult]:
        """
        Classify using LLM (slow path).

        Args:
            utterance: Latest utterance
            speaker: Speaker name
            transcript: Recent conversation context

        Returns:
            RelevanceResult if successful, None on error/timeout
        """
        if self.llm_classifier is None:
            logger.warning("No LLM classifier configured, skipping slow path")
            return None

        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_classification_prompt(utterance, speaker, transcript)

            # Call LLM with timeout
            response = await asyncio.wait_for(
                self.llm_classifier(prompt),
                timeout=self.slow_path_timeout,
            )

            # Parse JSON response
            result = json.loads(response)

            latency_ms = (time.time() - start_time) * 1000

            should_respond = result.get("respond", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")

            logger.debug(
                f"Slow path: respond={should_respond}, "
                f"confidence={confidence:.2f}, "
                f"reason='{reason}'"
            )

            return RelevanceResult(
                should_respond=should_respond,
                confidence=confidence,
                reason=reason,
                method="slow_path",
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"LLM classification timeout after {latency_ms:.0f}ms"
            )
            self.slow_path_timeouts += 1
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

    def _cache_key(self, utterance: str) -> str:
        """
        Generate cache key for utterance.

        Args:
            utterance: User's utterance

        Returns:
            Cache key (lowercase, normalized)
        """
        # Normalize: lowercase, strip, collapse whitespace
        normalized = " ".join(utterance.lower().strip().split())
        return normalized

    def _get_from_cache(self, utterance: str) -> Optional[RelevanceResult]:
        """
        Get cached result for utterance.

        Args:
            utterance: User's utterance

        Returns:
            Cached RelevanceResult if found, None otherwise
        """
        key = self._cache_key(utterance)

        if key in self._cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for: '{utterance[:50]}...'")
            return self._cache[key]

        return None

    def _add_to_cache(self, utterance: str, result: RelevanceResult) -> None:
        """
        Add result to cache.

        Args:
            utterance: User's utterance
            result: Classification result
        """
        key = self._cache_key(utterance)

        # Add to cache
        self._cache[key] = result

        # Prune if too large (simple FIFO)
        if len(self._cache) > self.cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    async def classify(
        self,
        utterance: str,
        speaker: str,
        transcript: str = "",
    ) -> RelevanceResult:
        """
        Classify whether bot should respond to utterance.

        Args:
            utterance: Latest utterance
            speaker: Speaker name
            transcript: Recent conversation context

        Returns:
            RelevanceResult with decision and metadata
        """
        self.total_classifications += 1

        # Check cache
        cached = self._get_from_cache(utterance)
        if cached is not None:
            return cached

        # Fast path: name mentions
        fast_result = self._check_fast_path(utterance)
        if fast_result is not None:
            self.fast_path_count += 1
            self._add_to_cache(utterance, fast_result)
            return fast_result

        # Get sensitivity threshold
        threshold = self.SENSITIVITY_THRESHOLDS.get(self.sensitivity, 0.75)

        # Low sensitivity: fast path only
        if self.sensitivity == "low":
            result = RelevanceResult(
                should_respond=False,
                confidence=0.0,
                reason="No name mention detected (low sensitivity)",
                method="fast_path",
                latency_ms=0.0,
            )
            self.fast_path_count += 1
            self._add_to_cache(utterance, result)
            return result

        # Slow path: LLM classification
        llm_result = await self._classify_with_llm(utterance, speaker, transcript)

        if llm_result is not None:
            self.slow_path_count += 1

            # Apply threshold
            if llm_result.confidence >= threshold:
                self._add_to_cache(utterance, llm_result)
                return llm_result
            else:
                # Below threshold - don't respond
                result = RelevanceResult(
                    should_respond=False,
                    confidence=llm_result.confidence,
                    reason=f"Confidence {llm_result.confidence:.2f} below threshold {threshold:.2f}",
                    method="slow_path",
                    latency_ms=llm_result.latency_ms,
                )
                self._add_to_cache(utterance, result)
                return result

        # LLM failed/timeout - fallback to conservative default
        logger.warning("LLM classification failed, defaulting to no response")

        result = RelevanceResult(
            should_respond=False,
            confidence=0.0,
            reason="LLM classification failed or timed out",
            method="slow_path_fallback",
            latency_ms=0.0,
        )
        self.slow_path_count += 1
        return result

    def set_sensitivity(self, sensitivity: str) -> None:
        """
        Update sensitivity level.

        Args:
            sensitivity: New sensitivity ("low", "medium", "high")
        """
        if sensitivity not in self.SENSITIVITY_THRESHOLDS:
            raise ValueError(
                f"Invalid sensitivity: {sensitivity}. "
                f"Choose from: {list(self.SENSITIVITY_THRESHOLDS.keys())}"
            )

        old_sensitivity = self.sensitivity
        self.sensitivity = sensitivity

        logger.info(
            f"Sensitivity updated: {old_sensitivity} → {sensitivity} "
            f"(threshold: {self.SENSITIVITY_THRESHOLDS[sensitivity]})"
        )

    def clear_cache(self) -> None:
        """Clear classification cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {cache_size} cached classifications")

    def get_stats(self) -> dict:
        """
        Get filter statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "agent_name": self.agent_name,
            "sensitivity": self.sensitivity,
            "threshold": self.SENSITIVITY_THRESHOLDS[self.sensitivity],
            "total_classifications": self.total_classifications,
            "fast_path_count": self.fast_path_count,
            "slow_path_count": self.slow_path_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._cache),
            "slow_path_timeouts": self.slow_path_timeouts,
            "fast_path_ratio": (
                self.fast_path_count / self.total_classifications
                if self.total_classifications > 0
                else 0.0
            ),
        }


class PerGuildRelevanceFilter:
    """
    Manages separate relevance filters for multiple Discord guilds.

    Each guild can have different agent/sensitivity settings.
    """

    def __init__(
        self,
        default_agent: str = "Jarvis",
        default_sensitivity: str = "medium",
        llm_classifier=None,
    ):
        """
        Initialize per-guild filter manager.

        Args:
            default_agent: Default agent name
            default_sensitivity: Default sensitivity level
            llm_classifier: LLM classifier callable
        """
        self.default_agent = default_agent
        self.default_sensitivity = default_sensitivity
        self.llm_classifier = llm_classifier

        # Per-guild filters
        self._filters: Dict[int, RelevanceFilter] = {}

    def get_or_create(
        self,
        guild_id: int,
        agent_name: Optional[str] = None,
        sensitivity: Optional[str] = None,
    ) -> RelevanceFilter:
        """
        Get or create relevance filter for a guild.

        Args:
            guild_id: Discord guild ID
            agent_name: Override agent name (None = use default)
            sensitivity: Override sensitivity (None = use default)

        Returns:
            RelevanceFilter for this guild
        """
        if guild_id not in self._filters:
            self._filters[guild_id] = RelevanceFilter(
                agent_name=agent_name or self.default_agent,
                sensitivity=sensitivity or self.default_sensitivity,
                llm_classifier=self.llm_classifier,
            )
            logger.info(
                f"Created relevance filter for guild {guild_id} "
                f"(agent: {agent_name or self.default_agent}, "
                f"sensitivity: {sensitivity or self.default_sensitivity})"
            )

        return self._filters[guild_id]

    async def classify(
        self,
        guild_id: int,
        utterance: str,
        speaker: str,
        transcript: str = "",
    ) -> RelevanceResult:
        """
        Classify utterance for a guild.

        Args:
            guild_id: Discord guild ID
            utterance: Latest utterance
            speaker: Speaker name
            transcript: Recent conversation context

        Returns:
            RelevanceResult
        """
        filter_instance = self.get_or_create(guild_id)
        return await filter_instance.classify(utterance, speaker, transcript)

    def set_agent(self, guild_id: int, agent_name: str) -> None:
        """
        Set agent for a guild.

        Args:
            guild_id: Discord guild ID
            agent_name: Agent name
        """
        filter_instance = self.get_or_create(guild_id)
        filter_instance.agent_name = agent_name
        filter_instance._name_patterns = filter_instance._build_name_patterns(agent_name)
        logger.info(f"Guild {guild_id} agent set to: {agent_name}")

    def set_sensitivity(self, guild_id: int, sensitivity: str) -> None:
        """
        Set sensitivity for a guild.

        Args:
            guild_id: Discord guild ID
            sensitivity: Sensitivity level
        """
        filter_instance = self.get_or_create(guild_id)
        filter_instance.set_sensitivity(sensitivity)

    def remove_guild(self, guild_id: int) -> None:
        """
        Remove filter for a guild.

        Args:
            guild_id: Discord guild ID
        """
        if guild_id in self._filters:
            del self._filters[guild_id]
            logger.info(f"Removed relevance filter for guild {guild_id}")

    def get_all_stats(self) -> Dict[int, dict]:
        """
        Get stats for all guilds.

        Returns:
            Dictionary mapping guild_id -> stats
        """
        return {
            guild_id: filter_instance.get_stats()
            for guild_id, filter_instance in self._filters.items()
        }


# Convenience function
def create_relevance_filter(
    agent_name: str = "Jarvis",
    sensitivity: str = "medium",
    llm_classifier=None,
) -> RelevanceFilter:
    """
    Create relevance filter with default settings.

    Args:
        agent_name: Name of agent
        sensitivity: Sensitivity level
        llm_classifier: LLM classifier callable

    Returns:
        RelevanceFilter instance
    """
    return RelevanceFilter(
        agent_name=agent_name,
        sensitivity=sensitivity,
        llm_classifier=llm_classifier,
    )
