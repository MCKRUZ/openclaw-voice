"""OpenClaw Gateway WebSocket JSON-RPC client.

Implements the OpenClaw Gateway protocol for agent response generation.
Connects via WebSocket to OpenClaw Gateway running on Synology NAS.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig:
    """Configuration for OpenClaw Gateway client."""

    # WebSocket URL for OpenClaw Gateway
    base_url: str = "ws://192.168.50.9:18789"

    # Authentication token (from OPENCLAW_AUTH_TOKEN env var)
    auth_token: Optional[str] = None

    # Request timeout (seconds)
    timeout: float = 8.0

    # Retry timeout for second attempt
    retry_timeout: float = 15.0

    # Maximum number of retries
    max_retries: int = 1

    # Agent ID for session keys
    agent_id: str = "main"

    # Session scope: "per-peer" or "shared"
    session_scope: str = "per-peer"


class OpenClawClient:
    """
    WebSocket client for OpenClaw Gateway JSON-RPC protocol.

    Manages connection, handshake, and chat message exchange with
    OpenClaw Gateway running on Synology NAS.
    """

    # Agent personalities (for system context)
    AGENT_PERSONALITIES = {
        "main": (
            "You are an intelligent and helpful AI assistant "
            "participating in a Discord voice conversation. You are knowledgeable, "
            "professional, and provide thoughtful, concise responses. "
            "You speak naturally in conversation, avoiding overly formal language."
        ),
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

    def __init__(self, config: OpenClawConfig):
        """
        Initialize OpenClaw Gateway client.

        Args:
            config: Client configuration
        """
        self.config = config

        # WebSocket connection
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False

        # Request/response tracking
        self._pending: Dict[str, asyncio.Future] = {}
        self._chat_waiters: Dict[str, asyncio.Future] = {}
        self._stream_queues: Dict[str, asyncio.Queue] = {}  # For streaming responses

        # Background listener task
        self._listener_task: Optional[asyncio.Task] = None

        # Reconnection lock
        self._reconnect_lock = asyncio.Lock()

        # Stats
        self.total_requests = 0
        self.total_failures = 0
        self.total_retries = 0
        self.total_latency = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Gateway."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish WebSocket connection and complete the handshake.

        Protocol:
        1. Connect to WebSocket
        2. Wait for connect.challenge event
        3. Send connect request with auth
        4. Wait for hello-ok response
        5. Start background listener

        Raises:
            ConnectionError: If handshake fails
        """
        url = self.config.base_url
        logger.info(f"Connecting to OpenClaw Gateway at {url}")

        # Connect WebSocket
        self._ws = await websockets.connect(url, max_size=10 * 1024 * 1024)

        # Wait for connect.challenge
        challenge_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
        challenge = json.loads(challenge_msg)

        if challenge.get("event") != "connect.challenge":
            raise ConnectionError(
                f"Expected connect.challenge, got: {challenge.get('event')}"
            )

        nonce = challenge["payload"]["nonce"]
        logger.debug(f"Received challenge nonce: {nonce}")

        # Send connect request
        connect_params = {
            "minProtocol": 3,
            "maxProtocol": 5,
            "client": {
                "id": "gateway-client",
                "displayName": "OpenClaw Voice Bot",
                "version": "1.0.0",
                "platform": "custom",
                "mode": "backend",
            },
            "role": "operator",
            "caps": [],
            "commands": [],
            "permissions": {},
            "scopes": ["chat", "operator.read", "operator.write"],
            "auth": {},
        }

        if self.config.auth_token:
            connect_params["auth"] = {"token": self.config.auth_token}

        connect_id = self._new_id()
        frame = {
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": connect_params,
        }
        await self._ws.send(json.dumps(frame))

        # Read hello response
        resp_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
        resp = json.loads(resp_msg)

        if not resp.get("ok"):
            error = resp.get("error", {})
            raise ConnectionError(
                f"Gateway connect failed: {error.get('message', 'unknown')}"
            )

        server_info = resp.get("payload", {}).get("server", {})
        logger.info(
            f"Connected to OpenClaw Gateway "
            f"(version={server_info.get('version', '?')}, "
            f"connId={server_info.get('connId', '?')})"
        )
        self._connected = True

        # Start background listener for subsequent messages
        self._listener_task = asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        """Gracefully close the Gateway connection."""
        self._connected = False

        if self._listener_task:
            self._listener_task.cancel()
            self._listener_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        # Cancel all pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()

        for fut in self._chat_waiters.values():
            if not fut.done():
                fut.cancel()

        self._pending.clear()
        self._chat_waiters.clear()
        self._stream_queues.clear()

    async def send_message(
        self,
        agent: str,
        message: str,
        context: str = "",
        speaker: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Send message to agent and get response.

        Args:
            agent: Agent name ("jarvis" or "sage")
            message: User's message/utterance
            context: Recent conversation context (not used with Gateway)
            speaker: Speaker name/ID (used for session key)
            model: Optional model override (e.g., "claude-haiku-3.5", "claude-sonnet-4")

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
            # Ensure connected
            await self._ensure_connected()

            # Build session key
            session_key = self._build_session_key(speaker or "default")

            # Try with normal timeout
            response = await self._send_chat(
                session_key, message, timeout=self.config.timeout, model=model
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
                await self._ensure_connected()
                session_key = self._build_session_key(speaker or "default")

                response = await self._send_chat(
                    session_key, message, timeout=self.config.retry_timeout, model=model
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

    async def _send_chat(
        self, session_key: str, message: str, timeout: float = 120, model: Optional[str] = None
    ) -> str:
        """
        Send a chat message and wait for the final response text.

        Args:
            session_key: OpenClaw session key (e.g. "agent:main:discord:dm:123")
            message: User's transcribed speech
            timeout: Max seconds to wait for AI response
            model: Optional model override (e.g., "claude-haiku-3.5")

        Returns:
            Agent's response text

        Raises:
            RuntimeError: If chat.send fails
            asyncio.TimeoutError: If response takes too long
        """
        idempotency_key = f"voice-{uuid.uuid4().hex[:12]}"
        req_id = self._new_id()

        try:
            # Build chat.send params
            params = {
                "sessionKey": session_key,
                "message": message,
                "deliver": True,
                "idempotencyKey": idempotency_key,
                "timeoutMs": int(timeout * 1000),
            }

            # Add model override if specified
            if model:
                params["model"] = model

            # Send chat.send request
            await self._send_request(
                req_id,
                "chat.send",
                params,
            )

            # Wait for RPC acknowledgement to get server-assigned runId
            resp = await self._wait_response(req_id, timeout=15)
            if not resp.get("ok"):
                error = resp.get("error", {})
                raise RuntimeError(
                    f"chat.send failed: {error.get('message', 'unknown')}"
                )

            # Use server-assigned runId as waiter key
            run_id = resp.get("payload", {}).get("runId", idempotency_key)

            # Create waiter for final response
            waiter: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._chat_waiters[run_id] = waiter

            try:
                result = await asyncio.wait_for(waiter, timeout=timeout)
                return result
            finally:
                self._chat_waiters.pop(run_id, None)

        except Exception:
            # Clean up any waiter that might have been registered
            self._chat_waiters.pop(idempotency_key, None)
            raise

    async def send_message_streaming(
        self,
        agent: str,
        message: str,
        context: str = "",
        speaker: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Send message and stream response chunks in real-time.

        Args:
            agent: Agent name ("jarvis" or "sage")
            message: User's message/utterance
            context: Recent conversation context (not used with Gateway)
            speaker: Speaker name/ID (used for session key)
            model: Optional model override

        Yields:
            Text chunks as they arrive from the LLM

        Raises:
            RuntimeError: If request fails
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
            # Ensure connected
            await self._ensure_connected()

            # Build session key
            session_key = self._build_session_key(speaker or "default")

            # Stream the chat response
            async for chunk in self._send_chat_streaming(
                session_key, message, model=model
            ):
                yield chunk

            latency = time.time() - start_time
            self.total_latency += latency

            logger.info(
                f"Agent {agent} streaming response completed in {latency:.2f}s"
            )

        except Exception as e:
            self.total_failures += 1
            logger.error(f"OpenClaw streaming request failed: {e}")
            raise RuntimeError(f"Failed to get streaming response from {agent}: {e}")

    async def _send_chat_streaming(
        self, session_key: str, message: str, model: Optional[str] = None, timeout: float = 120
    ) -> AsyncIterator[str]:
        """
        Send a chat message and stream response chunks.

        Args:
            session_key: OpenClaw session key
            message: User's transcribed speech
            model: Optional model override
            timeout: Max seconds to wait for response

        Yields:
            Text deltas as they arrive

        Raises:
            RuntimeError: If chat.send fails
            asyncio.TimeoutError: If response takes too long
        """
        idempotency_key = f"voice-stream-{uuid.uuid4().hex[:12]}"
        req_id = self._new_id()

        try:
            # Build chat.send params
            params = {
                "sessionKey": session_key,
                "message": message,
                "deliver": True,
                "idempotencyKey": idempotency_key,
                "timeoutMs": int(timeout * 1000),
            }

            if model:
                params["model"] = model

            # Send chat.send request
            await self._send_request(req_id, "chat.send", params)

            # Wait for RPC acknowledgement
            resp = await self._wait_response(req_id, timeout=15)
            if not resp.get("ok"):
                error = resp.get("error", {})
                raise RuntimeError(
                    f"chat.send failed: {error.get('message', 'unknown')}"
                )

            # Use server-assigned runId as stream key
            run_id = resp.get("payload", {}).get("runId", idempotency_key)

            # Create queue for streaming chunks
            stream_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            self._stream_queues[run_id] = stream_queue

            try:
                # Stream chunks from queue
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream_queue.get(), timeout=timeout
                        )

                        if chunk is None:
                            # End of stream sentinel
                            break

                        yield chunk

                    except asyncio.TimeoutError:
                        logger.warning(f"Stream timeout waiting for chunk (runId: {run_id})")
                        break

            finally:
                self._stream_queues.pop(run_id, None)

        except Exception:
            self._stream_queues.pop(idempotency_key, None)
            raise

    async def abort_chat(self, session_key: str) -> None:
        """
        Abort any in-flight chat for the session.

        Args:
            session_key: OpenClaw session key
        """
        await self._ensure_connected()
        req_id = self._new_id()
        await self._send_request(
            req_id, "chat.abort", {"sessionKey": session_key}
        )

    async def _ensure_connected(self) -> None:
        """Reconnect if disconnected."""
        if self._connected and self._ws:
            return

        async with self._reconnect_lock:
            if self._connected and self._ws:
                return
            logger.warning("Gateway disconnected, reconnecting...")
            await self.connect()

    async def _send_request(
        self, req_id: str, method: str, params: dict
    ) -> None:
        """
        Send a JSON-RPC request frame.

        Args:
            req_id: Request ID
            method: RPC method name
            params: Method parameters
        """
        frame = {
            "type": "req",
            "id": req_id,
            "method": method,
            "params": params,
        }

        if not self._ws:
            raise ConnectionError("Not connected to Gateway")

        await self._ws.send(json.dumps(frame))

    async def _wait_response(self, req_id: str, timeout: float = 30) -> dict:
        """
        Wait for a response matching the given request ID.

        Args:
            req_id: Request ID to wait for
            timeout: Timeout in seconds

        Returns:
            Response payload
        """
        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    async def _listen(self) -> None:
        """Background task that reads all incoming WebSocket messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from Gateway")
                    continue

                msg_type = msg.get("type")

                if msg_type == "res":
                    # RPC response
                    req_id = msg.get("id")
                    fut = self._pending.get(req_id)
                    if fut and not fut.done():
                        fut.set_result(msg)

                elif msg_type == "event":
                    # Event notification
                    event_name = msg.get("event")
                    if event_name == "chat":
                        self._handle_chat_event(msg.get("payload", {}))

        except ConnectionClosed:
            logger.warning("Gateway WebSocket closed")
            self._connected = False
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Gateway listener error")
            self._connected = False

    def _handle_chat_event(self, payload: dict) -> None:
        """
        Process incoming chat events, resolve waiters on 'final'.

        Args:
            payload: Chat event payload
        """
        run_id = payload.get("runId", "")
        state = payload.get("state", "")

        if state == "final":
            # Extract text content from final message
            message = payload.get("message", {})
            content = message.get("content", [])
            text_parts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            response_text = "\n".join(text_parts).strip()

            # Resolve waiting future (non-streaming)
            fut = self._chat_waiters.get(run_id)
            if fut and not fut.done():
                fut.set_result(response_text)

            # Signal end of stream (streaming)
            stream_queue = self._stream_queues.get(run_id)
            if stream_queue:
                # Send None sentinel to indicate stream end
                stream_queue.put_nowait(None)

        elif state == "error":
            # Chat error
            error_msg = payload.get("errorMessage", "Unknown error")
            logger.error(f"Chat error for runId {run_id}: {error_msg}")

            fut = self._chat_waiters.get(run_id)
            if fut and not fut.done():
                fut.set_exception(RuntimeError(f"Chat error: {error_msg}"))

            stream_queue = self._stream_queues.get(run_id)
            if stream_queue:
                stream_queue.put_nowait(None)

        elif state == "aborted":
            # Chat aborted
            fut = self._chat_waiters.get(run_id)
            if fut and not fut.done():
                fut.set_exception(asyncio.CancelledError("Chat aborted"))

            stream_queue = self._stream_queues.get(run_id)
            if stream_queue:
                stream_queue.put_nowait(None)

        elif state == "delta":
            # Streaming delta - extract text and send to stream queue
            delta = payload.get("delta", {})
            text_delta = ""

            # Extract text from delta content blocks
            if "content" in delta:
                for block in delta.get("content", []):
                    if block.get("type") == "text":
                        text_delta += block.get("text", "")

            # Send delta to stream queue if we have one
            if text_delta:
                stream_queue = self._stream_queues.get(run_id)
                if stream_queue:
                    stream_queue.put_nowait(text_delta)

    def _build_session_key(self, user_id: str) -> str:
        """
        Build OpenClaw session key for user.

        Format: agent:<agentId>:discord:dm:<userId>

        Args:
            user_id: Discord user ID

        Returns:
            Session key
        """
        uid = str(user_id).strip().lower()

        if self.config.session_scope == "per-peer":
            return f"agent:{self.config.agent_id}:discord:dm:{uid}"
        else:
            return f"agent:{self.config.agent_id}:main"

    def format_context(self, transcript: str) -> str:
        """
        Format transcript for context.

        Note: OpenClaw Gateway maintains conversation history internally,
        so we don't need to send explicit context.

        Args:
            transcript: Raw transcript text

        Returns:
            Formatted context (empty for Gateway)
        """
        return ""

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
            "connected": self._connected,
        }

    @staticmethod
    def _new_id() -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())


class PerGuildOpenClawClient:
    """
    Manages separate OpenClaw sessions for multiple Discord guilds.

    Each guild can maintain independent conversation state.
    """

    def __init__(self, config: OpenClawConfig):
        """
        Initialize per-guild client manager.

        Args:
            config: Default client configuration
        """
        self.config = config

        # Per-guild clients (for session management)
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
            self._clients[guild_id] = OpenClawClient(config=self.config)
            logger.info(f"Created OpenClaw client for guild {guild_id}")

        return self._clients[guild_id]

    async def send_message(
        self,
        guild_id: int,
        agent: str,
        message: str,
        context: str = "",
        speaker: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Send message for a guild.

        Args:
            guild_id: Discord guild ID
            agent: Agent name
            message: User's message
            context: Conversation context
            speaker: Speaker name
            model: Optional model override

        Returns:
            Agent's response
        """
        client = self.get_or_create(guild_id)
        return await client.send_message(agent, message, context, speaker, model)

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
    base_url: str = "ws://192.168.50.9:18789",
    auth_token: Optional[str] = None,
    timeout: float = 8.0,
    agent_id: str = "main",
) -> OpenClawClient:
    """
    Create OpenClaw Gateway client with default settings.

    Args:
        base_url: OpenClaw Gateway WebSocket URL
        auth_token: Authentication token
        timeout: Request timeout (seconds)
        agent_id: Agent ID for session keys

    Returns:
        OpenClawClient instance
    """
    config = OpenClawConfig(
        base_url=base_url,
        auth_token=auth_token,
        timeout=timeout,
        agent_id=agent_id,
    )

    return OpenClawClient(config=config)
