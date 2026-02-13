"""OpenClaw API client for agent response generation.

Stubbed implementation using direct LLM API for testing.
Will be replaced with actual OpenClaw API integration.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpenClawConfig:
    """Configuration for OpenClaw client."""

    base_url: str = "http://your-synology-nas:port"  # TODO: Set actual Synology NAS URL
    auth_token: Optional[str] = None  # TODO: Set actual auth token
    timeout: float = 5.0  # First attempt timeout
    retry_timeout: float = 10.0  # Retry timeout
    max_retries: int = 1


class OpenClawClient:
    """
    Client for OpenClaw API.

    Currently stubbed with direct LLM API for testing.
    Replace with actual OpenClaw integration when available.
    """

    # Agent personalities (for stub implementation)
    AGENT_PERSONALITIES = {
        "jarvis": (
            "You are Jarvis, an intelligent and helpful AI assistant "
            "participating in a Discord voice conversation. You are knowledgeable, "
            "professional, and provide thoughtful, concise responses. "
            "You speak naturally in conversation, avoiding overly formal language."
        ),
        "sage": (
            "You are Sage, a wise and insightful AI assistant "
            "participating in a Discord voice conversation. You offer deep insights "
            "and thoughtful perspectives. You are calm, measured, and speak with "
            "clarity and wisdom."
        ),
    }

    def __init__(
        self,
        config: OpenClawConfig,
        llm_client=None,
    ):
        """
        Initialize OpenClaw client.

        Args:
            config: Client configuration
            llm_client: Optional LLM client for stubbed implementation
        """
        self.config = config
        self.llm_client = llm_client

        # Stats
        self.total_requests = 0
        self.total_failures = 0
        self.total_retries = 0
        self.total_latency = 0.0

    async def send_message(
        self,
        agent: str,
        message: str,
        context: str = "",
        speaker: Optional[str] = None,
    ) -> str:
        """
        Send message to agent and get response.

        Args:
            agent: Agent name ("jarvis" or "sage")
            message: User's message/utterance
            context: Recent conversation context
            speaker: Speaker name (optional)

        Returns:
            Agent's response text

        Raises:
            RuntimeError: If request fails after retries
            ValueError: If agent is invalid
        """
        agent_lower = agent.lower()
        if agent_lower not in self.AGENT_PERSONALITIES:
            raise ValueError(
                f"Invalid agent: {agent}. "
                f"Choose from: {list(self.AGENT_PERSONALITIES.keys())}"
            )

        self.total_requests += 1
        start_time = time.time()

        try:
            # Try with normal timeout
            response = await self._send_with_timeout(
                agent_lower, message, context, speaker, self.config.timeout
            )

            latency = time.time() - start_time
            self.total_latency += latency

            logger.info(
                f"Agent {agent} responded in {latency:.2f}s: "
                f'"{response[:50]}..."'
            )

            return response

        except asyncio.TimeoutError:
            logger.warning(
                f"First attempt timeout ({self.config.timeout}s), retrying..."
            )
            self.total_retries += 1

            try:
                # Retry with extended timeout
                response = await self._send_with_timeout(
                    agent_lower,
                    message,
                    context,
                    speaker,
                    self.config.retry_timeout,
                )

                latency = time.time() - start_time
                self.total_latency += latency

                logger.info(
                    f"Agent {agent} responded on retry in {latency:.2f}s"
                )

                return response

            except Exception as e:
                self.total_failures += 1
                logger.error(f"OpenClaw request failed after retry: {e}")
                raise RuntimeError(
                    f"Failed to get response from {agent} after retry: {e}"
                )

        except Exception as e:
            self.total_failures += 1
            logger.error(f"OpenClaw request failed: {e}")
            raise RuntimeError(f"Failed to get response from {agent}: {e}")

    async def _send_with_timeout(
        self,
        agent: str,
        message: str,
        context: str,
        speaker: Optional[str],
        timeout: float,
    ) -> str:
        """
        Send request with timeout.

        Args:
            agent: Agent name
            message: User's message
            context: Conversation context
            speaker: Speaker name
            timeout: Timeout in seconds

        Returns:
            Agent's response

        Raises:
            asyncio.TimeoutError: If request times out
        """
        return await asyncio.wait_for(
            self._send_request(agent, message, context, speaker),
            timeout=timeout,
        )

    async def _send_request(
        self,
        agent: str,
        message: str,
        context: str,
        speaker: Optional[str],
    ) -> str:
        """
        Send request to agent (stubbed implementation).

        TODO: Replace with actual OpenClaw API when available.

        Args:
            agent: Agent name
            message: User's message
            context: Conversation context
            speaker: Speaker name

        Returns:
            Agent's response
        """
        # Format message for voice context
        if speaker:
            formatted_message = f"[Voice] {speaker} said: {message}"
        else:
            formatted_message = f"[Voice] {message}"

        # Build system prompt with personality and context
        personality = self.AGENT_PERSONALITIES[agent]
        system_prompt = f"{personality}\n\n"

        if context:
            system_prompt += f"Recent conversation:\n{context}\n\n"

        system_prompt += "Respond naturally and concisely to the voice message. Keep your response brief (1-3 sentences) since this is a spoken conversation."

        # Stub: Use direct LLM API if available
        if self.llm_client is not None:
            logger.debug(f"Using LLM client stub for agent {agent}")
            response = await self.llm_client(
                system_prompt=system_prompt,
                user_message=formatted_message,
            )
            return response

        # Fallback: Return placeholder response
        logger.warning(
            "No LLM client configured, returning placeholder response"
        )
        return f"[{agent.title()}] I received your message about: {message[:30]}... (Stub response - configure LLM client for real responses)"

    def format_context(self, transcript: str) -> str:
        """
        Format transcript for context.

        Args:
            transcript: Raw transcript text

        Returns:
            Formatted context
        """
        if not transcript:
            return ""

        # Already formatted by TranscriptManager
        return transcript

    def get_stats(self) -> dict:
        """
        Get client statistics.

        Returns:
            Dictionary with stats
        """
        avg_latency = (
            self.total_latency / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_retries": self.total_retries,
            "success_rate": (
                (self.total_requests - self.total_failures) / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "avg_latency": avg_latency,
        }


class PerGuildOpenClawClient:
    """
    Manages separate OpenClaw sessions for multiple Discord guilds.

    Each guild can maintain independent conversation state.
    """

    def __init__(
        self,
        config: OpenClawConfig,
        llm_client=None,
    ):
        """
        Initialize per-guild client manager.

        Args:
            config: Default client configuration
            llm_client: LLM client for stubbed implementation
        """
        self.config = config
        self.llm_client = llm_client

        # Per-guild clients (for session management in future)
        self._clients: Dict[int, OpenClawClient] = {}

    def get_or_create(self, guild_id: int) -> OpenClawClient:
        """
        Get or create client for a guild.

        Args:
            guild_id: Discord guild ID

        Returns:
            OpenClawClient for this guild
        """
        if guild_id not in self._clients:
            self._clients[guild_id] = OpenClawClient(
                config=self.config,
                llm_client=self.llm_client,
            )
            logger.info(f"Created OpenClaw client for guild {guild_id}")

        return self._clients[guild_id]

    async def send_message(
        self,
        guild_id: int,
        agent: str,
        message: str,
        context: str = "",
        speaker: Optional[str] = None,
    ) -> str:
        """
        Send message for a guild.

        Args:
            guild_id: Discord guild ID
            agent: Agent name
            message: User's message
            context: Conversation context
            speaker: Speaker name

        Returns:
            Agent's response
        """
        client = self.get_or_create(guild_id)
        return await client.send_message(agent, message, context, speaker)

    def remove_guild(self, guild_id: int) -> None:
        """
        Remove client for a guild.

        Args:
            guild_id: Discord guild ID
        """
        if guild_id in self._clients:
            del self._clients[guild_id]
            logger.info(f"Removed OpenClaw client for guild {guild_id}")

    def get_all_stats(self) -> Dict[int, dict]:
        """
        Get stats for all guilds.

        Returns:
            Dictionary mapping guild_id -> stats
        """
        return {
            guild_id: client.get_stats()
            for guild_id, client in self._clients.items()
        }


# Convenience function
def create_client(
    base_url: str = "http://localhost:8080",
    auth_token: Optional[str] = None,
    timeout: float = 5.0,
    llm_client=None,
) -> OpenClawClient:
    """
    Create OpenClaw client with default settings.

    Args:
        base_url: OpenClaw API base URL
        auth_token: Authentication token
        timeout: Request timeout (seconds)
        llm_client: LLM client for stubbed implementation

    Returns:
        OpenClawClient instance
    """
    config = OpenClawConfig(
        base_url=base_url,
        auth_token=auth_token,
        timeout=timeout,
    )

    return OpenClawClient(config=config, llm_client=llm_client)
