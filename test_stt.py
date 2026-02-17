"""Test STT (Speech-To-Text) to verify microphone input is working.

This script will:
1. Load the STT model
2. Wait for you to speak in Discord
3. Show exactly what it transcribes in real-time
"""

import asyncio
import numpy as np
from pathlib import Path

from utils.config import load_config
from server.stt import create_stt_transcriber
from utils.logging import get_logger

logger = get_logger(__name__)


async def test_stt():
    """Test STT with sample audio."""
    print("\n" + "="*70)
    print("STT (Speech-To-Text) Test")
    print("="*70 + "\n")

    # Load config
    config = load_config(Path("config.yaml"))

    # Create STT transcriber
    print("Loading STT model (this may take a moment)...")
    transcriber = await create_stt_transcriber(config.stt)
    print(f"✓ STT model loaded: {config.stt.model} on {config.stt.device}\n")

    # Create test scenarios
    print("Testing different audio scenarios:\n")

    # Test 1: Silent audio (should return empty or [silence])
    print("Test 1: Silent audio (0.5s of silence)")
    silent_audio = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz
    result = await transcriber.transcribe(silent_audio, user_id=0)
    print(f"  Result: '{result.text}' (confidence: {result.confidence:.2f})")
    print(f"  Expected: Empty or '[silence]'\n")

    # Test 2: Generate a simple tone (not speech, but tests processing)
    print("Test 2: Tone audio (should not detect speech)")
    tone_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.1
    result = await transcriber.transcribe(tone_audio, user_id=0)
    print(f"  Result: '{result.text}'")
    print(f"  Expected: Empty or noise\n")

    print("="*70)
    print("\nSTT Test Complete!")
    print("\nNext steps:")
    print("1. Join Discord voice channel with the bot")
    print("2. Speak clearly: 'Jarvis, can you hear me?'")
    print("3. Check the bot logs to see the transcription:")
    print("   tail -f /tmp/bot-final.log | grep 'Transcribed'")
    print("\nIf you see correct transcriptions in the logs, STT is working!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_stt())
