"""Test OpenClaw Gateway connection."""

import asyncio
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from openclaw_client import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_gateway_connection():
    """Test OpenClaw Gateway connection."""
    print("=" * 70)
    print("OpenClaw Gateway Connection Test")
    print("=" * 70)
    print()

    # Get credentials from environment
    base_url = os.getenv("OPENCLAW_BASE_URL", "ws://192.168.50.9:18789")
    auth_token = os.getenv("OPENCLAW_AUTH_TOKEN")
    agent_id = os.getenv("OPENCLAW_AGENT_ID", "main")

    print(f"Gateway URL: {base_url}")
    print(f"Agent ID: {agent_id}")
    print(f"Auth Token: {'***' + auth_token[-4:] if auth_token else 'None'}")
    print()

    try:
        # Create client
        print("Creating OpenClaw client...")
        client = create_client(
            base_url=base_url,
            auth_token=auth_token,
            agent_id=agent_id,
            timeout=8.0,
        )
        print("[OK] Client created")
        print()

        # Connect to Gateway
        print("Connecting to Gateway...")
        await client.connect()
        print("[OK] Connected to Gateway")
        print()

        # Test message for Jarvis
        print("Sending test message to Jarvis agent...")
        response = await client.send_message(
            agent="jarvis",
            message="Hello, this is a test from openclaw-voice. Please respond briefly.",
            speaker="test_user_123",
        )
        print(f"[OK] Received response from Jarvis:")
        # Encode to ASCII, replacing Unicode characters with '?'
        print(f"  {response.encode('ascii', 'replace').decode('ascii')}")
        print()

        # Test message for Sage
        print("Sending test message to Sage agent...")
        response = await client.send_message(
            agent="sage",
            message="Hello Sage, this is a test. Please respond briefly.",
            speaker="test_user_456",
        )
        print(f"[OK] Received response from Sage:")
        # Encode to ASCII, replacing Unicode characters with '?'
        print(f"  {response.encode('ascii', 'replace').decode('ascii')}")
        print()

        # Get stats
        stats = client.get_stats()
        print("Client Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"  Avg latency: {stats['avg_latency']:.2f}s")
        print(f"  Connected: {stats['connected']}")
        print()

        # Disconnect
        print("Disconnecting from Gateway...")
        await client.disconnect()
        print("[OK] Disconnected")
        print()

        print("=" * 70)
        print("SUCCESS: ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("FAILED: TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gateway_connection())
    sys.exit(0 if success else 1)
