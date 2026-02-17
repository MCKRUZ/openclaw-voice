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

        # Validate OpenClaw Gateway configuration
        if not config.openclaw.base_url:
            logger.error("OpenClaw Gateway URL not configured!")
            logger.error("Set OPENCLAW_BASE_URL environment variable in .env file")
            return 1

        if not config.openclaw.token:
            logger.error("OpenClaw Gateway token not configured!")
            logger.error("Set OPENCLAW_AUTH_TOKEN environment variable in .env file")
            return 1

        logger.info("✓ OpenClaw Gateway configured")

        # Display configuration summary
        logger.info("")
        logger.info("Configuration Summary:")
        logger.info(f"  Default Agent: {config.agents.default}")
        logger.info(f"  OpenClaw Gateway: {config.openclaw.base_url}")
        logger.info(f"  OpenClaw Agent ID: {config.openclaw.agent_id}")
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
            sample_rate=24000,  # Default sample rate for Chatterbox TTS
        )
        logger.info(f"✓ TTS engine initialized ({config.pipeline.tts.device})")

        # Warmup TTS and cache common phrases
        logger.info("Warming up TTS engine and caching common phrases...")
        await tts_synthesizer.warmup()
        logger.info(f"✓ TTS warmup complete ({len(tts_synthesizer.phrase_cache)} phrases cached)")

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

        # Initialize OpenClaw Gateway client
        logger.info("Initializing OpenClaw Gateway client...")
        from openclaw_client import OpenClawConfig

        openclaw_config = OpenClawConfig(
            base_url=config.openclaw.base_url,
            auth_token=config.openclaw.token,
            timeout=config.openclaw.timeout,
            retry_timeout=config.openclaw.retry_timeout,
            agent_id=config.openclaw.agent_id,
            session_scope=config.openclaw.session_scope,
        )
        logger.info(f"✓ OpenClaw Gateway client initialized ({config.openclaw.base_url})")

        # Initialize Pipeline Components
        logger.info("Initializing voice processing pipeline...")

        from pipeline import (
            SileroVAD,
            SmartTurnDetector,
            PipelineTranscriber,
            TranscriptManager,
            RelevanceFilter,
            PipelineOrchestrator,
            PipelineConfig,
            QueryRouter,
        )
        from openclaw_client import OpenClawClient

        # Create pipeline components
        vad = SileroVAD()
        logger.info("✓ VAD initialized (Silero)")

        turn_detector = SmartTurnDetector(
            model_path=Path("models") / config.pipeline.turn_detection.model_path,
            threshold=config.pipeline.turn_detection.threshold,
        )
        logger.info("✓ Smart Turn v3 detector initialized")

        stt_pipeline = PipelineTranscriber(
            transcriber=stt_transcriber,
        )
        logger.info("✓ STT pipeline wrapped")

        transcript_manager = TranscriptManager(
            max_age_seconds=config.pipeline.transcript.window_duration,
            max_entries=config.pipeline.transcript.max_turns,
        )
        logger.info("✓ Transcript manager initialized")

        relevance_filter = RelevanceFilter(
            agent_name=config.agents.default,
            sensitivity=config.pipeline.relevance.default_sensitivity,
        )
        logger.info("✓ Relevance filter initialized")

        query_router = QueryRouter(default_model="sonnet")
        logger.info("✓ Query router initialized")

        # Create OpenClaw client instance for pipeline
        openclaw_client = OpenClawClient(openclaw_config)

        # Create audio output callback (will be set by Discord bot)
        audio_output_callbacks = {}

        def audio_output_callback(user_id: int, audio_data):
            """Route audio output to appropriate callback."""
            if user_id in audio_output_callbacks:
                audio_output_callbacks[user_id](audio_data)

        # Create pipeline orchestrator
        pipeline_config = PipelineConfig(
            vad_silence_duration=config.pipeline.vad.silence_threshold,
            turn_completion_threshold=config.pipeline.turn_detection.threshold,
            turn_wait_timeout=config.pipeline.turn_detection.max_wait,
            stt_timeout=5.0,
            relevance_timeout=2.0,
            llm_timeout=10.0,
            tts_timeout=10.0,
            sample_rate=16000,
        )

        orchestrator = PipelineOrchestrator(
            config=pipeline_config,
            vad=vad,
            turn_detector=turn_detector,
            transcriber=stt_pipeline,
            transcript_manager=transcript_manager,
            relevance_filter=relevance_filter,
            llm_client=openclaw_client,
            tts_synthesizer=tts_synthesizer,
            audio_output_callback=audio_output_callback,
            query_router=query_router,
        )

        logger.info("✓ Pipeline orchestrator initialized with all optimizations")
        logger.info("  - STT beam_size=1 optimization active")
        logger.info("  - Smart model router active (Haiku/Sonnet/Opus)")
        logger.info("  - Sentence-level streaming TTS active")
        logger.info("  - TTS phrase cache active")

        # Test OpenClaw Gateway connection
        logger.info("Testing OpenClaw Gateway connection...")
        try:
            await openclaw_client.connect()
            logger.info(f"✓ Connected to OpenClaw Gateway ({config.openclaw.base_url})")
        except Exception as e:
            logger.error(f"✗ Failed to connect to OpenClaw Gateway: {e}")
            logger.error("Check OPENCLAW_BASE_URL and OPENCLAW_AUTH_TOKEN in .env")
            logger.error("Ensure OpenClaw Gateway is running on Synology NAS")
            return 1

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
            run_bot(
                config=config,
                openclaw_config=openclaw_config,
                tts_synthesizer=tts_synthesizer,
                stt_transcriber=stt_transcriber,
                orchestrator=orchestrator,
                audio_output_callbacks=audio_output_callbacks,
            ),
            name="discord_bot",
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
