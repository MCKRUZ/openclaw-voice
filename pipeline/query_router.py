"""Smart Query Router - Route queries to optimal Claude model based on complexity.

Routes to:
- Haiku (claude-haiku-3.5): Simple queries, ~100ms first token
- Sonnet (claude-sonnet-4): Medium complexity, ~300ms first token
- Opus (claude-opus-4-6): Complex queries, ~800ms first token
"""

import re
from dataclasses import dataclass
from typing import Literal

from utils.logging import get_logger

logger = get_logger(__name__)


ModelType = Literal["haiku", "sonnet", "opus"]


@dataclass
class RoutingDecision:
    """Result of query routing."""

    model: ModelType
    model_id: str
    reason: str
    confidence: float  # 0.0-1.0


class QueryRouter:
    """
    Routes voice queries to the fastest appropriate Claude model.

    Uses pattern matching for instant classification without LLM calls.
    """

    # Model identifiers for OpenClaw Gateway
    MODEL_IDS = {
        "haiku": "claude-haiku-3.5",
        "sonnet": "claude-sonnet-4",
        "opus": "claude-opus-4-6",
    }

    # Patterns for simple queries (route to Haiku)
    SIMPLE_PATTERNS = [
        # Greetings
        re.compile(r"^(hey|hi|hello|good morning|good afternoon|good evening|what's up|sup|yo)", re.IGNORECASE),
        # Confirmations
        re.compile(r"^(yes|no|yeah|nah|yep|nope|sure|okay|ok|alright|got it|sounds good)", re.IGNORECASE),
        # Thanks
        re.compile(r"^(thanks|thank you|thx|ty|appreciated|cheers)", re.IGNORECASE),
        # Time/date
        re.compile(r"(what time|what day|what's the time|what's the date|current time|current date)", re.IGNORECASE),
        # Weather (basic)
        re.compile(r"^(what's the weather|how's the weather|weather today)", re.IGNORECASE),
        # Simple questions
        re.compile(r"^(who are you|what are you|are you there|can you hear me)", re.IGNORECASE),
        # Single word queries
        re.compile(r"^\w+\?*$"),  # Single word (with optional ?)
    ]

    # Patterns for complex queries (route to Opus)
    COMPLEX_PATTERNS = [
        # Analysis requests
        re.compile(r"(analyze|compare|evaluate|assess|review|critique)", re.IGNORECASE),
        # Creative writing
        re.compile(r"(write me|draft|compose|create a|generate a)", re.IGNORECASE),
        # Research/investigation
        re.compile(r"(research|investigate|look into|find out about|tell me about .{50,})", re.IGNORECASE),
        # Explanations
        re.compile(r"(explain why|explain how|what do you think about|your opinion on)", re.IGNORECASE),
        # Strategy/planning
        re.compile(r"(strategy|plan for|how should I|what's the best way)", re.IGNORECASE),
        # Long, detailed questions (>100 chars usually complex)
        re.compile(r"^.{100,}"),
        # Multiple questions
        re.compile(r"\?.+\?"),  # Contains multiple question marks
    ]

    # Patterns for medium complexity (route to Sonnet) - checked after simple/complex
    MEDIUM_PATTERNS = [
        # Information requests
        re.compile(r"(what is|what are|who is|who are|when did|where is|how does)", re.IGNORECASE),
        # Action requests
        re.compile(r"(can you|could you|would you|please|help me)", re.IGNORECASE),
        # Queries with context
        re.compile(r"(tell me|show me|give me|find me)", re.IGNORECASE),
    ]

    def __init__(self, default_model: ModelType = "sonnet"):
        """
        Initialize query router.

        Args:
            default_model: Default model for uncertain classifications
        """
        self.default_model = default_model
        self.default_model_id = self.MODEL_IDS[default_model]

        # Stats
        self.total_routes = 0
        self.routes_by_model = {"haiku": 0, "sonnet": 0, "opus": 0}

        logger.info(
            f"Query router initialized (default: {default_model})"
        )

    def route(self, query: str) -> RoutingDecision:
        """
        Route query to appropriate model.

        Args:
            query: User's transcribed query

        Returns:
            RoutingDecision with model selection and reasoning
        """
        query_clean = query.strip()

        # Empty query - use default
        if not query_clean:
            return self._make_decision(
                self.default_model,
                "empty_query",
                0.5,
            )

        # Check simple patterns first (highest priority for speed)
        for pattern in self.SIMPLE_PATTERNS:
            if pattern.search(query_clean):
                return self._make_decision(
                    "haiku",
                    f"matched_simple_pattern: {pattern.pattern[:50]}",
                    0.9,
                )

        # Check complex patterns (second priority)
        for pattern in self.COMPLEX_PATTERNS:
            if pattern.search(query_clean):
                return self._make_decision(
                    "opus",
                    f"matched_complex_pattern: {pattern.pattern[:50]}",
                    0.85,
                )

        # Check medium patterns
        for pattern in self.MEDIUM_PATTERNS:
            if pattern.search(query_clean):
                return self._make_decision(
                    "sonnet",
                    f"matched_medium_pattern: {pattern.pattern[:50]}",
                    0.8,
                )

        # Default fallback - use Sonnet as safe middle ground
        return self._make_decision(
            self.default_model,
            "no_pattern_match_fallback",
            0.6,
        )

    def _make_decision(
        self, model: ModelType, reason: str, confidence: float
    ) -> RoutingDecision:
        """
        Create routing decision and update stats.

        Args:
            model: Model to route to
            reason: Reason for routing
            confidence: Confidence in decision

        Returns:
            RoutingDecision
        """
        self.total_routes += 1
        self.routes_by_model[model] += 1

        decision = RoutingDecision(
            model=model,
            model_id=self.MODEL_IDS[model],
            reason=reason,
            confidence=confidence,
        )

        logger.debug(
            f"Routed to {model} (confidence: {confidence:.2f}, reason: {reason})"
        )

        return decision

    def get_stats(self) -> dict:
        """
        Get routing statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_routes": self.total_routes,
            "routes_by_model": self.routes_by_model.copy(),
            "distribution": {
                model: (
                    count / self.total_routes if self.total_routes > 0 else 0.0
                )
                for model, count in self.routes_by_model.items()
            },
            "default_model": self.default_model,
        }

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.total_routes = 0
        self.routes_by_model = {"haiku": 0, "sonnet": 0, "opus": 0}
        logger.info("Router stats reset")
