"""Pipeline Orchestrator - Event-driven coordinator for voice processing.

Wires all pipeline stages together:
    audio_in → vad → turn_detect → stt → relevance → respond → tts → audio_out

Per-user state machines with cancellation support.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from pipeline.audio_buffer import AudioRingBuffer
from pipeline.query_router import QueryRouter
from pipeline.relevance_filter import RelevanceFilter
from pipeline.sentence_splitter import split_streaming_response
from pipeline.transcriber import STTTranscriber
from pipeline.transcript_manager import TranscriptManager
from pipeline.turn_detector import SmartTurnDetector
from pipeline.vad import SileroVAD
from server.tts import TTSSynthesizer
from utils.logging import get_logger

logger = get_logger(__name__)


class PipelineState(Enum):
    """User pipeline states."""

    IDLE = "idle"  # Waiting for speech
    LISTENING = "listening"  # VAD detected speech start
    TURN_WAIT = "turn_wait"  # VAD silence, checking turn completion
    PROCESSING = "processing"  # Transcribing and deciding
    RESPONDING = "responding"  # Generating TTS and playing


@dataclass
class UserPipeline:
    """Per-user pipeline state."""

    user_id: int
    user_name: str
    state: PipelineState = PipelineState.IDLE

    # Audio buffer
    audio_buffer: AudioRingBuffer = field(
        default_factory=lambda: AudioRingBuffer(duration_seconds=10.0)
    )

    # Speech detection
    speech_start_time: Optional[float] = None
    last_speech_time: Optional[float] = None

    # Processing
    current_task: Optional[asyncio.Task] = None
    processing_start_time: Optional[float] = None

    # Latency tracking
    stage_latencies: Dict[str, float] = field(default_factory=dict)

    # Stats
    total_utterances: int = 0
    total_responses: int = 0
    total_cancellations: int = 0


@dataclass
class PipelineConfig:
    """Pipeline orchestrator configuration."""

    # VAD settings
    vad_silence_duration: float = 0.3  # Seconds of silence to detect speech end
    vad_chunk_size: int = 512  # Samples per VAD check (16kHz)

    # Smart Turn settings
    turn_wait_timeout: float = 3.0  # Max wait after silence for turn completion
    turn_completion_threshold: float = 0.7  # Probability threshold

    # Processing timeouts
    stt_timeout: float = 5.0
    relevance_timeout: float = 2.0
    llm_timeout: float = 10.0
    tts_timeout: float = 10.0

    # Concurrent processing
    max_concurrent_users: int = 5

    # Audio settings
    sample_rate: int = 16000


class PipelineOrchestrator:
    """
    Event-driven pipeline orchestrator.

    Coordinates voice processing for multiple users:
    - Per-user state machines
    - Cancellation and barge-in support
    - Latency tracking
    - Error handling and recovery
    """

    def __init__(
        self,
        config: PipelineConfig,
        vad: SileroVAD,
        turn_detector: SmartTurnDetector,
        transcriber: STTTranscriber,
        transcript_manager: TranscriptManager,
        relevance_filter: RelevanceFilter,
        llm_client: Callable,  # OpenClaw client
        tts_synthesizer: TTSSynthesizer,
        audio_output_callback: Callable[[int, np.ndarray], None],
        query_router: Optional[QueryRouter] = None,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration
            vad: VAD detector
            turn_detector: Smart Turn detector
            transcriber: STT transcriber
            transcript_manager: Transcript manager
            relevance_filter: Relevance filter
            llm_client: LLM client for responses (OpenClaw)
            tts_synthesizer: TTS synthesizer
            audio_output_callback: Callback for playing audio (user_id, audio)
            query_router: Query router for model selection (optional)
        """
        self.config = config
        self.vad = vad
        self.turn_detector = turn_detector
        self.transcriber = transcriber
        self.transcript_manager = transcript_manager
        self.relevance_filter = relevance_filter
        self.llm_client = llm_client
        self.tts_synthesizer = tts_synthesizer
        self.audio_output_callback = audio_output_callback
        self.query_router = query_router or QueryRouter(default_model="sonnet")

        # Per-user pipelines
        self.pipelines: Dict[int, UserPipeline] = {}

        # Global stats
        self.total_audio_frames = 0
        self.total_pipeline_runs = 0
        self.total_errors = 0

        # Semaphore for concurrent processing
        self._processing_semaphore = asyncio.Semaphore(
            config.max_concurrent_users
        )

        # Current agent
        self.current_agent = "jarvis"

        # Start speech timeout monitor
        self._shutdown = False
        self._monitor_task = asyncio.create_task(self._monitor_speech_timeouts())

        logger.info(f"Pipeline orchestrator initialized: {config}")

    def get_or_create_pipeline(
        self, user_id: int, user_name: str
    ) -> UserPipeline:
        """
        Get or create pipeline for user.

        Args:
            user_id: User ID
            user_name: User display name

        Returns:
            User pipeline instance
        """
        if user_id not in self.pipelines:
            self.pipelines[user_id] = UserPipeline(
                user_id=user_id, user_name=user_name
            )
            logger.info(f"Created pipeline for user: {user_name} ({user_id})")

        return self.pipelines[user_id]

    def remove_pipeline(self, user_id: int) -> None:
        """
        Remove user pipeline (e.g., user left channel).

        Args:
            user_id: User ID
        """
        if user_id in self.pipelines:
            pipeline = self.pipelines[user_id]

            # Cancel current task if any
            if pipeline.current_task and not pipeline.current_task.done():
                pipeline.current_task.cancel()

            del self.pipelines[user_id]
            logger.info(
                f"Removed pipeline for user: {pipeline.user_name} ({user_id})"
            )

    async def process_audio_frame(
        self, user_id: int, user_name: str, audio_frame: np.ndarray
    ) -> None:
        """
        Process incoming audio frame from user.

        Args:
            user_id: User ID
            user_name: User display name
            audio_frame: Audio data (float32, 16kHz mono)
        """
        pipeline = self.get_or_create_pipeline(user_id, user_name)

        # Add to buffer
        pipeline.audio_buffer.write(audio_frame)
        self.total_audio_frames += 1

        # Check if user is speaking during our response (barge-in)
        if pipeline.state == PipelineState.RESPONDING:
            logger.info(
                f"Barge-in detected: {user_name} spoke during response"
            )
            await self._cancel_pipeline(pipeline)
            pipeline.state = PipelineState.LISTENING
            pipeline.speech_start_time = time.time()
            return

        # Process VAD
        await self._process_vad(pipeline, audio_frame)

    async def _process_vad(
        self, pipeline: UserPipeline, audio_frame: np.ndarray
    ) -> None:
        """
        Process VAD on audio frame.

        Args:
            pipeline: User pipeline
            audio_frame: Audio chunk
        """
        # Run VAD (CPU, fast)
        state, speech_prob = self.vad.process_chunk(audio_frame)

        current_time = time.time()

        # Check if speech is detected
        from pipeline.vad import SpeechState
        is_speech = (state == SpeechState.SPEECH)

        if is_speech:
            # Speech detected
            if pipeline.state == PipelineState.IDLE:
                # Speech start
                pipeline.state = PipelineState.LISTENING
                pipeline.speech_start_time = current_time
                logger.debug(
                    f"Speech started: {pipeline.user_name} "
                    f"({pipeline.user_id})"
                )

            pipeline.last_speech_time = current_time

        else:
            # Silence detected
            if pipeline.state == PipelineState.LISTENING:
                # Check if silence duration exceeded
                silence_duration = current_time - (
                    pipeline.last_speech_time or current_time
                )

                if silence_duration >= self.config.vad_silence_duration:
                    # Speech end - proceed to turn detection
                    logger.debug(
                        f"Speech ended: {pipeline.user_name} "
                        f"(silence: {silence_duration:.2f}s)"
                    )
                    await self._handle_speech_end(pipeline)

    async def _monitor_speech_timeouts(self) -> None:
        """Background task to monitor for speech timeouts."""
        while not self._shutdown:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms

                current_time = time.time()
                for user_id, pipeline in list(self.pipelines.items()):
                    if pipeline.state == PipelineState.LISTENING:
                        if pipeline.last_speech_time:
                            silence_duration = current_time - pipeline.last_speech_time
                            if silence_duration >= self.config.vad_silence_duration:
                                # Speech ended due to timeout
                                logger.info(
                                    f"Speech ended (timeout): {pipeline.user_name} "
                                    f"(silence: {silence_duration:.2f}s)"
                                )
                                await self._handle_speech_end(pipeline)
            except Exception as e:
                logger.error(f"Error in speech timeout monitor: {e}", exc_info=True)

    async def _handle_speech_end(self, pipeline: UserPipeline) -> None:
        """
        Handle speech end - check turn completion.

        Args:
            pipeline: User pipeline
        """
        pipeline.state = PipelineState.TURN_WAIT

        # Get audio segment
        speech_duration = time.time() - (pipeline.speech_start_time or 0)
        audio_segment = pipeline.audio_buffer.read(duration_seconds=8.0)

        if len(audio_segment) == 0:
            logger.warning(
                f"Empty audio segment for {pipeline.user_name}, ignoring"
            )
            pipeline.state = PipelineState.IDLE
            return

        # Check turn completion with timeout
        try:
            turn_start = time.time()

            is_complete = await asyncio.wait_for(
                self._check_turn_completion(audio_segment),
                timeout=self.config.turn_wait_timeout,
            )

            turn_latency = time.time() - turn_start
            pipeline.stage_latencies["turn_detection"] = turn_latency

            if is_complete:
                # Turn complete - proceed to transcription
                logger.info(
                    f"Turn complete for {pipeline.user_name} "
                    f"(latency: {turn_latency:.3f}s)"
                )
                await self._start_processing(pipeline, audio_segment)
            else:
                # Turn not complete - wait for more speech
                logger.debug(
                    f"Turn incomplete for {pipeline.user_name}, "
                    f"waiting for more speech"
                )
                pipeline.state = PipelineState.LISTENING

        except asyncio.TimeoutError:
            # Timeout - assume turn complete
            logger.warning(
                f"Turn detection timeout for {pipeline.user_name}, "
                f"assuming complete"
            )
            await self._start_processing(pipeline, audio_segment)

    async def _check_turn_completion(
        self, audio_segment: np.ndarray
    ) -> bool:
        """
        Check if turn is complete using Smart Turn.

        Args:
            audio_segment: Audio segment

        Returns:
            True if turn is complete
        """
        probability = await self.turn_detector.detect_async(audio_segment)
        return probability >= self.config.turn_completion_threshold

    async def _start_processing(
        self, pipeline: UserPipeline, audio_segment: np.ndarray
    ) -> None:
        """
        Start processing pipeline for utterance.

        Args:
            pipeline: User pipeline
            audio_segment: Speech audio
        """
        pipeline.state = PipelineState.PROCESSING
        pipeline.processing_start_time = time.time()
        pipeline.total_utterances += 1

        # Create processing task
        task = asyncio.create_task(
            self._process_utterance(pipeline, audio_segment)
        )
        pipeline.current_task = task

    async def _process_utterance(
        self, pipeline: UserPipeline, audio_segment: np.ndarray
    ) -> None:
        """
        Process utterance through full pipeline.

        Args:
            pipeline: User pipeline
            audio_segment: Speech audio
        """
        try:
            async with self._processing_semaphore:
                # 1. Transcribe (STT)
                stt_start = time.time()
                transcript = await asyncio.wait_for(
                    self.transcriber.transcribe_async(audio_segment),
                    timeout=self.config.stt_timeout,
                )
                pipeline.stage_latencies["stt"] = time.time() - stt_start

                if not transcript or not transcript.text.strip():
                    logger.warning(
                        f"Empty transcription for {pipeline.user_name}"
                    )
                    pipeline.state = PipelineState.IDLE
                    return

                logger.info(
                    f"Transcribed ({pipeline.user_name}): "
                    f'"{transcript.text}" '
                    f"(latency: {pipeline.stage_latencies['stt']:.3f}s)"
                )

                # 2. Add to transcript context
                self.transcript_manager.add_entry(
                    speaker=pipeline.user_name, text=transcript.text
                )

                # 3. Check relevance
                rel_start = time.time()
                context = self.transcript_manager.get_context(format="readable")

                should_respond = await asyncio.wait_for(
                    self.relevance_filter.classify(
                        utterance=transcript.text,
                        speaker=pipeline.user_name,
                        transcript=context,
                        agent=self.current_agent,
                        sensitivity=self.relevance_filter.sensitivity,
                    ),
                    timeout=self.config.relevance_timeout,
                )
                pipeline.stage_latencies["relevance"] = time.time() - rel_start

                if not should_respond:
                    logger.info(
                        f"Not responding to {pipeline.user_name}: "
                        f'"{transcript.text}"'
                    )
                    pipeline.state = PipelineState.IDLE
                    return

                logger.info(
                    f"Responding to {pipeline.user_name}: "
                    f'"{transcript.text}" '
                    f"(latency: {pipeline.stage_latencies['relevance']:.3f}s)"
                )

                # 4. Route query to optimal model
                routing_start = time.time()
                routing_decision = self.query_router.route(transcript.text)
                pipeline.stage_latencies["routing"] = time.time() - routing_start

                logger.info(
                    f"Routed to {routing_decision.model} "
                    f"(confidence: {routing_decision.confidence:.2f}, "
                    f"reason: {routing_decision.reason})"
                )

                # 5. Generate response with streaming TTS
                pipeline.state = PipelineState.RESPONDING

                llm_start = time.time()
                first_audio_time = None
                full_response_text = []

                try:
                    # Stream LLM response and split into sentences
                    text_stream = self.llm_client.send_message_streaming(
                        agent=self.current_agent,
                        message=transcript.text,
                        context=context,
                        speaker=pipeline.user_name,
                        model=routing_decision.model_id,
                    )

                    sentence_stream = split_streaming_response(text_stream)

                    # Process each sentence as it arrives
                    async for sentence in sentence_stream:
                        # Record first sentence timing (critical metric)
                        if sentence.index == 0:
                            pipeline.stage_latencies["llm_first_sentence"] = time.time() - llm_start
                            logger.info(
                                f"First sentence from LLM in {pipeline.stage_latencies['llm_first_sentence']:.3f}s: "
                                f'"{sentence.text}"'
                            )

                        # Collect full text for transcript
                        full_response_text.append(sentence.text)

                        # Generate TTS for this sentence
                        tts_start = time.time()
                        audio_chunk = await asyncio.wait_for(
                            self.tts_synthesizer.synthesize(
                                agent=self.current_agent,
                                text=sentence.text,
                            ),
                            timeout=self.config.tts_timeout,
                        )

                        if sentence.index == 0:
                            pipeline.stage_latencies["tts_first_chunk"] = time.time() - tts_start

                        if audio_chunk is None:
                            logger.warning(f"TTS failed for sentence #{sentence.index}")
                            continue

                        # Play audio immediately
                        self.audio_output_callback(pipeline.user_id, audio_chunk)

                        # Track first audio playback time (time to first audio)
                        if first_audio_time is None:
                            first_audio_time = time.time() - llm_start
                            pipeline.stage_latencies["time_to_first_audio"] = first_audio_time
                            logger.info(
                                f"First audio playing in {first_audio_time:.3f}s "
                                f"(LLM: {pipeline.stage_latencies['llm_first_sentence']:.3f}s, "
                                f"TTS: {pipeline.stage_latencies['tts_first_chunk']:.3f}s)"
                            )

                        logger.debug(
                            f"Played sentence #{sentence.index} "
                            f"({len(audio_chunk) / self.config.sample_rate:.2f}s audio)"
                        )

                    # Streaming complete
                    pipeline.stage_latencies["llm"] = time.time() - llm_start
                    response_text = " ".join(full_response_text)

                    logger.info(
                        f"Streaming response complete ({self.current_agent}, {routing_decision.model}): "
                        f'"{response_text[:100]}..." '
                        f"(total latency: {pipeline.stage_latencies['llm']:.3f}s)"
                    )

                    # Add bot response to transcript
                    self.transcript_manager.add_entry(
                        speaker=self.current_agent.title(), text=response_text
                    )

                except Exception as e:
                    logger.error(f"Streaming TTS pipeline error: {e}", exc_info=True)
                    pipeline.state = PipelineState.IDLE
                    return

                # Update stats
                pipeline.total_responses += 1
                self.total_pipeline_runs += 1

                # Calculate total latency
                total_latency = time.time() - (
                    pipeline.processing_start_time or time.time()
                )
                pipeline.stage_latencies["total"] = total_latency

                logger.info(
                    f"Pipeline complete for {pipeline.user_name}: "
                    f"total latency {total_latency:.3f}s, "
                    f"stages: {pipeline.stage_latencies}"
                )

                # Return to idle
                pipeline.state = PipelineState.IDLE

        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled for {pipeline.user_name}")
            pipeline.total_cancellations += 1
            pipeline.state = PipelineState.IDLE
            raise

        except asyncio.TimeoutError as e:
            logger.error(
                f"Pipeline timeout for {pipeline.user_name}: {e}"
            )
            self.total_errors += 1
            pipeline.state = PipelineState.IDLE

        except Exception as e:
            logger.error(
                f"Pipeline error for {pipeline.user_name}: {e}", exc_info=True
            )
            self.total_errors += 1
            pipeline.state = PipelineState.IDLE

    async def _cancel_pipeline(self, pipeline: UserPipeline) -> None:
        """
        Cancel current pipeline processing.

        Args:
            pipeline: User pipeline
        """
        if pipeline.current_task and not pipeline.current_task.done():
            pipeline.current_task.cancel()
            try:
                await pipeline.current_task
            except asyncio.CancelledError:
                pass

        pipeline.state = PipelineState.IDLE

    def set_agent(self, agent: str) -> None:
        """
        Set current active agent.

        Args:
            agent: Agent name ("jarvis" or "sage")
        """
        self.current_agent = agent.lower()
        logger.info(f"Switched to agent: {self.current_agent}")

    def set_sensitivity(self, sensitivity: str) -> None:
        """
        Set relevance sensitivity.

        Args:
            sensitivity: Sensitivity level ("low", "medium", "high")
        """
        self.relevance_filter.sensitivity = sensitivity.lower()
        logger.info(f"Set sensitivity to: {sensitivity}")

    def get_stats(self) -> dict:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with stats
        """
        # Aggregate user stats
        total_utterances = sum(p.total_utterances for p in self.pipelines.values())
        total_responses = sum(p.total_responses for p in self.pipelines.values())
        total_cancellations = sum(
            p.total_cancellations for p in self.pipelines.values()
        )

        # Calculate average latencies
        avg_latencies = {}
        if total_responses > 0:
            for stage in [
                "stt",
                "routing",
                "relevance",
                "llm_first_sentence",
                "tts_first_chunk",
                "time_to_first_audio",
                "llm",
                "total",
            ]:
                latencies = [
                    p.stage_latencies.get(stage, 0)
                    for p in self.pipelines.values()
                    if stage in p.stage_latencies
                ]
                avg_latencies[f"avg_{stage}_latency"] = (
                    sum(latencies) / len(latencies) if latencies else 0.0
                )

        return {
            "active_users": len(self.pipelines),
            "current_agent": self.current_agent,
            "sensitivity": self.relevance_filter.sensitivity,
            "total_audio_frames": self.total_audio_frames,
            "total_utterances": total_utterances,
            "total_responses": total_responses,
            "total_cancellations": total_cancellations,
            "total_pipeline_runs": self.total_pipeline_runs,
            "total_errors": self.total_errors,
            "router_stats": self.query_router.get_stats(),
            **avg_latencies,
        }

    def get_user_stats(self, user_id: int) -> Optional[dict]:
        """
        Get stats for specific user.

        Args:
            user_id: User ID

        Returns:
            User stats or None if not found
        """
        if user_id not in self.pipelines:
            return None

        pipeline = self.pipelines[user_id]

        return {
            "user_id": pipeline.user_id,
            "user_name": pipeline.user_name,
            "state": pipeline.state.value,
            "total_utterances": pipeline.total_utterances,
            "total_responses": pipeline.total_responses,
            "total_cancellations": pipeline.total_cancellations,
            "stage_latencies": pipeline.stage_latencies,
        }
