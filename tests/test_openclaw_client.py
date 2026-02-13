"""Unit tests for OpenClaw Client."""

import asyncio

import pytest

from openclaw_client import (
    OpenClawClient,
    OpenClawConfig,
    PerGuildOpenClawClient,
    create_client,
)


class TestOpenClawConfig:
    """Test OpenClawConfig dataclass."""

    def test_create_config(self):
        """Test creating config with defaults."""
        config = OpenClawConfig()

        assert "synology" in config.base_url.lower()
        assert config.auth_token is None
        assert config.timeout == 5.0
        assert config.retry_timeout == 10.0
        assert config.max_retries == 1

    def test_create_config_with_values(self):
        """Test creating config with custom values."""
        config = OpenClawConfig(
            base_url="http://192.168.1.100:8080",
            auth_token="test-token",
            timeout=3.0,
        )

        assert config.base_url == "http://192.168.1.100:8080"
        assert config.auth_token == "test-token"
        assert config.timeout == 3.0


class TestOpenClawClient:
    """Test OpenClawClient class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OpenClawConfig(
            base_url="http://test.local:8080",
            auth_token="test-token",
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""

        async def llm_client(system_prompt: str, user_message: str) -> str:
            # Simple mock that echoes back
            return f"Mock response to: {user_message}"

        return llm_client

    def test_create_client(self, config):
        """Test creating client."""
        client = OpenClawClient(config=config)

        assert client.config == config
        assert client.total_requests == 0
        assert client.total_failures == 0

    def test_agent_personalities(self):
        """Test agent personalities are defined."""
        assert "jarvis" in OpenClawClient.AGENT_PERSONALITIES
        assert "sage" in OpenClawClient.AGENT_PERSONALITIES

        # Check they're non-empty strings
        assert len(OpenClawClient.AGENT_PERSONALITIES["jarvis"]) > 0
        assert len(OpenClawClient.AGENT_PERSONALITIES["sage"]) > 0

    @pytest.mark.asyncio
    async def test_send_message_jarvis(self, config, mock_llm_client):
        """Test sending message to Jarvis."""
        client = OpenClawClient(config=config, llm_client=mock_llm_client)

        response = await client.send_message(
            agent="Jarvis",
            message="What's the weather?",
            speaker="Matt",
        )

        assert "Mock response" in response
        assert client.total_requests == 1
        assert client.total_failures == 0

    @pytest.mark.asyncio
    async def test_send_message_sage(self, config, mock_llm_client):
        """Test sending message to Sage."""
        client = OpenClawClient(config=config, llm_client=mock_llm_client)

        response = await client.send_message(
            agent="sage",
            message="Tell me about philosophy",
            speaker="Jake",
        )

        assert "Mock response" in response
        assert client.total_requests == 1

    @pytest.mark.asyncio
    async def test_send_message_with_context(self, config, mock_llm_client):
        """Test sending message with conversation context."""
        client = OpenClawClient(config=config, llm_client=mock_llm_client)

        context = "[8:31:02 PM] Matt: Hello\n[8:31:05 PM] Jarvis: Hi Matt"

        response = await client.send_message(
            agent="jarvis",
            message="How are you?",
            context=context,
            speaker="Matt",
        )

        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_send_message_invalid_agent(self, config):
        """Test sending message to invalid agent."""
        client = OpenClawClient(config=config)

        with pytest.raises(ValueError) as exc:
            await client.send_message(
                agent="invalid",
                message="Test",
            )

        assert "Invalid agent" in str(exc.value)

    @pytest.mark.asyncio
    async def test_send_message_without_llm_client(self, config):
        """Test sending message without LLM client (placeholder response)."""
        client = OpenClawClient(config=config, llm_client=None)

        response = await client.send_message(
            agent="jarvis",
            message="Test message",
        )

        # Should return placeholder
        assert "Stub response" in response
        assert "Test message" in response

    @pytest.mark.asyncio
    async def test_send_message_timeout_and_retry(self, config):
        """Test timeout and retry logic."""
        call_count = 0

        async def slow_llm_client(system_prompt: str, user_message: str) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: timeout
                await asyncio.sleep(10.0)
                return "Should timeout"
            else:
                # Retry: succeed
                return "Success on retry"

        config.timeout = 0.1  # Very short timeout
        config.retry_timeout = 1.0

        client = OpenClawClient(config=config, llm_client=slow_llm_client)

        response = await client.send_message(
            agent="jarvis",
            message="Test",
        )

        assert "Success on retry" in response
        assert client.total_retries == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_send_message_timeout_both_attempts(self, config):
        """Test timeout on both attempts."""

        async def always_slow_llm(system_prompt: str, user_message: str) -> str:
            await asyncio.sleep(10.0)
            return "Never gets here"

        config.timeout = 0.1
        config.retry_timeout = 0.2

        client = OpenClawClient(config=config, llm_client=always_slow_llm)

        with pytest.raises(RuntimeError) as exc:
            await client.send_message(
                agent="jarvis",
                message="Test",
            )

        assert "Failed to get response" in str(exc.value)
        assert client.total_failures == 1

    @pytest.mark.asyncio
    async def test_send_message_llm_error(self, config):
        """Test LLM client raising an error."""

        async def error_llm(system_prompt: str, user_message: str) -> str:
            raise RuntimeError("LLM error")

        client = OpenClawClient(config=config, llm_client=error_llm)

        with pytest.raises(RuntimeError) as exc:
            await client.send_message(
                agent="jarvis",
                message="Test",
            )

        assert "Failed to get response" in str(exc.value)
        assert client.total_failures == 1

    def test_format_context(self, config):
        """Test formatting context."""
        client = OpenClawClient(config=config)

        transcript = "[8:31:02 PM] Matt: Hello"
        formatted = client.format_context(transcript)

        # Currently just returns as-is (already formatted by TranscriptManager)
        assert formatted == transcript

    def test_format_context_empty(self, config):
        """Test formatting empty context."""
        client = OpenClawClient(config=config)

        formatted = client.format_context("")

        assert formatted == ""

    def test_get_stats_initial(self, config):
        """Test getting stats initially."""
        client = OpenClawClient(config=config)

        stats = client.get_stats()

        assert stats["total_requests"] == 0
        assert stats["total_failures"] == 0
        assert stats["total_retries"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_after_requests(self, config, mock_llm_client):
        """Test getting stats after requests."""
        client = OpenClawClient(config=config, llm_client=mock_llm_client)

        # Send successful request
        await client.send_message(agent="jarvis", message="Test 1")

        stats = client.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_failures"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_latency"] > 0.0

    @pytest.mark.asyncio
    async def test_get_stats_with_failures(self, config):
        """Test stats with failures."""

        async def error_llm(system_prompt: str, user_message: str) -> str:
            raise RuntimeError("Error")

        client = OpenClawClient(config=config, llm_client=error_llm)

        # Try request that will fail
        try:
            await client.send_message(agent="jarvis", message="Test")
        except RuntimeError:
            pass

        stats = client.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_failures"] == 1
        assert stats["success_rate"] == 0.0


class TestPerGuildOpenClawClient:
    """Test PerGuildOpenClawClient class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OpenClawConfig(
            base_url="http://test.local:8080",
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""

        async def llm_client(system_prompt: str, user_message: str) -> str:
            return f"Response: {user_message}"

        return llm_client

    def test_create_manager(self, config):
        """Test creating per-guild manager."""
        manager = PerGuildOpenClawClient(config=config)

        assert manager.config == config

    def test_get_or_create(self, config):
        """Test getting or creating guild client."""
        manager = PerGuildOpenClawClient(config=config)

        client = manager.get_or_create(guild_id=123)

        assert isinstance(client, OpenClawClient)

        # Getting again should return same instance
        client2 = manager.get_or_create(guild_id=123)
        assert client is client2

    def test_multiple_guilds(self, config):
        """Test managing multiple guilds."""
        manager = PerGuildOpenClawClient(config=config)

        client1 = manager.get_or_create(guild_id=111)
        client2 = manager.get_or_create(guild_id=222)

        # Should be different instances
        assert client1 is not client2

    @pytest.mark.asyncio
    async def test_send_message(self, config, mock_llm_client):
        """Test sending message via per-guild manager."""
        manager = PerGuildOpenClawClient(
            config=config, llm_client=mock_llm_client
        )

        response = await manager.send_message(
            guild_id=123,
            agent="jarvis",
            message="Test",
            speaker="Matt",
        )

        assert "Response" in response

    def test_remove_guild(self, config):
        """Test removing guild client."""
        manager = PerGuildOpenClawClient(config=config)

        manager.get_or_create(guild_id=123)
        assert 123 in manager._clients

        manager.remove_guild(guild_id=123)
        assert 123 not in manager._clients

    def test_remove_nonexistent_guild(self, config):
        """Test removing guild that doesn't exist."""
        manager = PerGuildOpenClawClient(config=config)

        # Should not raise error
        manager.remove_guild(guild_id=999)

    @pytest.mark.asyncio
    async def test_get_all_stats(self, config, mock_llm_client):
        """Test getting stats for all guilds."""
        manager = PerGuildOpenClawClient(
            config=config, llm_client=mock_llm_client
        )

        # Send messages to two guilds
        await manager.send_message(111, "jarvis", "Test 1", speaker="Matt")
        await manager.send_message(222, "sage", "Test 2", speaker="Jake")

        all_stats = manager.get_all_stats()

        assert 111 in all_stats
        assert 222 in all_stats
        assert all_stats[111]["total_requests"] == 1
        assert all_stats[222]["total_requests"] == 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_client(self):
        """Test creating client with convenience function."""

        async def mock_llm(system_prompt: str, user_message: str) -> str:
            return "Mock"

        client = create_client(
            base_url="http://test.local:8080",
            auth_token="token",
            timeout=3.0,
            llm_client=mock_llm,
        )

        assert isinstance(client, OpenClawClient)
        assert client.config.base_url == "http://test.local:8080"
        assert client.config.auth_token == "token"
        assert client.config.timeout == 3.0
        assert client.llm_client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
