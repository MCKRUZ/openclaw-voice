"""OpenClaw Gateway LLM client wrapper.

Provides a simple callable interface for the pipeline orchestrator.
"""

from typing import Optional

from openclaw_client import OpenClawConfig, PerGuildOpenClawClient
from utils.logging import get_logger

logger = get_logger(__name__)


class OpenClawLLMWrapper:
    """
    Wraps OpenClaw Gateway client for pipeline orchestrator.

    Provides a callable interface that matches the orchestrator's expectations:
        async def llm_client(agent: str, message: str, context: str, speaker: str) -> str
    """

    def __init__(self, config: OpenClawConfig, guild_id: int):
        """
        Initialize wrapper.

        Args:
            config: OpenClaw configuration
            guild_id: Discord guild ID
        """
        self.config = config
        self.guild_id = guild_id
        self.client_manager = PerGuildOpenClawClient(config)

    async def __call__(
        self,
        agent: str,
        message: str,
        context: str,
        speaker: str,
    ) -> str:
        """
        Send message to OpenClaw Gateway and get response.

        Args:
            agent: Agent name (jarvis, sage, etc.)
            message: User's message text
            context: Conversation context (managed by Gateway, not used)
            speaker: Speaker identifier (user ID or name)

        Returns:
            Agent's response text
        """
        # Get or create client for this guild
        client = self.client_manager.get_or_create(self.guild_id)

        # Send message to Gateway
        # Note: context is ignored because Gateway manages it internally
        response = await client.send_message(
            agent=agent,
            message=message,
            context="",  # Gateway manages context
            speaker=speaker,
        )

        return response

    async def disconnect(self):
        """Disconnect the OpenClaw client."""
        client = self.client_manager.get_or_create(self.guild_id)
        await client.disconnect()
        self.client_manager.remove_guild(self.guild_id)

    def get_stats(self) -> dict:
        """Get client statistics."""
        client = self.client_manager.get_or_create(self.guild_id)
        return client.get_stats()
