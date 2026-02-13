"""
Jarvis Voice Bot - Main Entry Point

This script starts both the Discord bot and FastAPI server.
"""

import asyncio
import signal
import sys
from pathlib import Path

from utils.config import load_config
from utils.logging import get_logger, setup_logging


# Global shutdown event
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n\nShutdown signal received. Cleaning up...\n")
    shutdown_event.set()


async def main():
    """Main application entry point."""
    logger = None

    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()

        # Setup logging
        setup_logging(config.logging)
        logger = get_logger(__name__)

        logger.info("=" * 70)
        logger.info("Jarvis Voice Bot Starting")
        logger.info("=" * 70)

        # Validate required configuration
        logger.info("Validating configuration...")

        if not config.discord.token:
            logger.error("Discord token not configured!")
            logger.error("Set DISCORD_TOKEN environment variable in .env file")
            return 1

        logger.info("✓ Discord token configured")

        # Check voice reference files
        from utils.config import get_voices_dir

        voices_dir = get_voices_dir()
        jarvis_voice = voices_dir / config.agents.jarvis.voice_file
        sage_voice = voices_dir / config.agents.sage.voice_file

        if not jarvis_voice.exists():
            logger.warning(f"Jarvis voice file not found: {jarvis_voice}")
            logger.warning("TTS will not work until voice file is provided")

        if not sage_voice.exists():
            logger.warning(f"Sage voice file not found: {sage_voice}")
            logger.warning("TTS will not work until voice file is provided")

        # Display configuration summary
        logger.info("")
        logger.info("Configuration Summary:")
        logger.info(f"  Default Agent: {config.agents.default}")
        logger.info(f"  STT Model: {config.pipeline.stt.model_size}")
        logger.info(f"  STT Device: {config.pipeline.stt.device}")
        logger.info(f"  TTS Engine: {config.pipeline.tts.engine}")
        logger.info(f"  TTS Device: {config.pipeline.tts.device}")
        logger.info(f"  Server Port: {config.server.port}")
        logger.info(f"  Latency Tracking: {config.logging.track_latency}")
        logger.info("")

        # Initialize shared TTS and STT engines
        logger.info("Initializing TTS and STT engines...")

        from server.stt import create_transcriber
        from server.tts import create_tts_synthesizer

        # Create voice references map
        voice_refs = {
            "jarvis": str(jarvis_voice),
            "sage": str(sage_voice),
        }

        # Initialize TTS synthesizer (shared between Discord and API)
        tts_synthesizer = await create_tts_synthesizer(
            voice_refs=voice_refs,
            device=config.pipeline.tts.device,
            sample_rate=config.pipeline.tts.sample_rate,
        )
        logger.info(f"✓ TTS engine initialized ({config.pipeline.tts.device})")

        # Initialize STT transcriber (shared between Discord and API)
        stt_transcriber = await create_transcriber(
            model_size=config.pipeline.stt.model_size,
            device=config.pipeline.stt.device,
            compute_type=config.pipeline.stt.compute_type,
        )
        logger.info(
            f"✓ STT engine initialized "
            f"({config.pipeline.stt.model_size} on {config.pipeline.stt.device})"
        )

        # Initialize FastAPI server
        logger.info("Initializing API server...")
        from server.app import create_api_server
        import uvicorn

        api_server = create_api_server(
            tts_synthesizer=tts_synthesizer,
            stt_transcriber=stt_transcriber,
        )
        logger.info(
            f"✓ API server initialized (port {config.server.port})"
        )

        # Initialize Discord bot
        logger.info("Initializing Discord bot...")
        from discord_bot.bot import run_bot

        logger.info("")
        logger.info("=" * 70)
        logger.info("Starting services...")
        logger.info("=" * 70)
        logger.info("")

        # Create tasks for both servers
        discord_task = asyncio.create_task(
            run_bot(config), name="discord_bot"
        )
        logger.info("✓ Discord bot started")

        # Create uvicorn server config
        uvicorn_config = uvicorn.Config(
            api_server.app,
            host=config.server.host,
            port=config.server.port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
        api_task = asyncio.create_task(
            uvicorn_server.serve(), name="api_server"
        )
        logger.info(
            f"✓ API server started on {config.server.host}:{config.server.port}"
        )

        logger.info("")
        logger.info("All services running. Press Ctrl+C to stop.")
        logger.info("")

        # Run both servers concurrently
        await asyncio.gather(discord_task, api_task, return_exceptions=True)

        return 0

    except FileNotFoundError as e:
        if logger:
            logger.error(f"Configuration error: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        if logger:
            logger.error(f"Configuration validation error: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        if logger:
            logger.info("Keyboard interrupt received")
        return 0

    except Exception as e:
        if logger:
            logger.exception(f"Unexpected error: {e}")
        else:
            print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    finally:
        if logger:
            logger.info("Shutdown complete")


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
