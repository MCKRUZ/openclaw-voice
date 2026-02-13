"""Smart Turn v3 integration for turn completion detection.

Uses Pipecat AI's Smart Turn v3 model to determine if a speaker has
finished their turn or is just pausing.
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from utils.config import get_models_dir
from utils.logging import get_logger, log_latency

logger = get_logger(__name__)


class SmartTurnDetector:
    """
    Smart Turn v3 turn completion detector.

    Determines if a speaker has finished their turn based on audio analysis.
    Uses an ONNX model that expects exactly 8 seconds of 16kHz audio.
    """

    # Model details
    MODEL_SAMPLE_RATE = 16000
    MODEL_DURATION = 8.0  # seconds
    MODEL_SAMPLES = int(MODEL_SAMPLE_RATE * MODEL_DURATION)  # 128,000 samples

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.7,
        device: str = "cpu",
    ):
        """
        Initialize Smart Turn detector.

        Args:
            model_path: Path to ONNX model file (None = auto-download)
            threshold: Turn completion threshold (0.0-1.0)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.threshold = threshold
        self.device = device

        # Determine model path
        if model_path is None:
            models_dir = get_models_dir()
            model_path = models_dir / "smart_turn_v3.onnx"

        self.model_path = model_path

        # Load model
        self.session = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model."""
        try:
            # Download if not exists
            if not self.model_path.exists():
                logger.info(f"Smart Turn model not found at {self.model_path}")
                logger.info("Attempting to download from HuggingFace...")
                self._download_model()

            logger.info(f"Loading Smart Turn model from {self.model_path}")

            # Configure ONNX runtime
            providers = []
            if self.device == "cuda":
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )

            # Get model info
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            logger.info(
                f"Smart Turn model loaded successfully "
                f"(input: {input_name}, output: {output_name})"
            )

        except Exception as e:
            logger.error(f"Failed to load Smart Turn model: {e}")
            raise

    def _download_model(self) -> None:
        """
        Download Smart Turn v3 model from HuggingFace.

        Note: This is a placeholder. In production, you would use huggingface_hub
        to download the model automatically.
        """
        try:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading Smart Turn v3 from HuggingFace...")

            # Download model
            downloaded_path = hf_hub_download(
                repo_id="pipecat-ai/smart-turn-v3",
                filename="model.onnx",
                cache_dir=get_models_dir(),
            )

            # Copy to expected location
            import shutil

            shutil.copy(downloaded_path, self.model_path)

            logger.info(f"Model downloaded to {self.model_path}")

        except ImportError:
            logger.error(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface_hub"
            )
            logger.error(
                f"Please manually download the model from "
                f"https://huggingface.co/pipecat-ai/smart-turn-v3 "
                f"and place it at {self.model_path}"
            )
            raise

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.error(
                f"Please manually download from "
                f"https://huggingface.co/pipecat-ai/smart-turn-v3"
            )
            raise

    def prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio for Smart Turn model.

        Model expects exactly 8 seconds (128,000 samples) of 16kHz mono audio.
        - If audio is shorter: zero-pad at the beginning
        - If audio is longer: truncate from the beginning (keep most recent)

        Args:
            audio: Audio array (float32, mono, 16kHz)

        Returns:
            Prepared audio (exactly 128,000 samples)
        """
        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio.dtype}")

        current_samples = len(audio)

        if current_samples > self.MODEL_SAMPLES:
            # Too long - keep most recent 8 seconds
            audio = audio[-self.MODEL_SAMPLES :]

        elif current_samples < self.MODEL_SAMPLES:
            # Too short - zero-pad at beginning
            padding = np.zeros(
                self.MODEL_SAMPLES - current_samples, dtype=np.float32
            )
            audio = np.concatenate([padding, audio])

        return audio

    def detect(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Detect if turn is complete.

        Args:
            audio: Audio to analyze (float32, mono, 16kHz, any length)

        Returns:
            Tuple of (is_complete, confidence)
            - is_complete: True if turn completion confidence >= threshold
            - confidence: Turn completion probability (0.0-1.0)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        with log_latency(logger, "turn_detection"):
            # Prepare audio (pad/truncate to 8 seconds)
            prepared_audio = self.prepare_audio(audio)

            # Reshape for model: [1, num_samples]
            input_tensor = prepared_audio.reshape(1, -1).astype(np.float32)

            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            outputs = self.session.run(
                [output_name],
                {input_name: input_tensor},
            )

            # Extract probability (handle various output shapes)
            output = outputs[0]
            if isinstance(output, np.ndarray):
                probability = float(output.flatten()[0])
            else:
                probability = float(output)

            # Clamp to [0, 1]
            probability = max(0.0, min(1.0, probability))

            # Determine completion
            is_complete = probability >= self.threshold

            logger.debug(
                f"Turn detection: probability={probability:.3f}, "
                f"threshold={self.threshold:.3f}, "
                f"complete={is_complete}"
            )

            return is_complete, probability

    async def detect_async(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Async wrapper for detect().

        Args:
            audio: Audio to analyze

        Returns:
            Tuple of (is_complete, confidence)
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect, audio)

    def set_threshold(self, threshold: float) -> None:
        """
        Update turn completion threshold.

        Args:
            threshold: New threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        old_threshold = self.threshold
        self.threshold = threshold

        logger.info(
            f"Turn completion threshold updated: {old_threshold:.2f} → {threshold:.2f}"
        )

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        if self.session is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "path": str(self.model_path),
            "threshold": self.threshold,
            "sample_rate": self.MODEL_SAMPLE_RATE,
            "duration": self.MODEL_DURATION,
            "samples": self.MODEL_SAMPLES,
            "device": self.device,
        }


class TurnDetectionManager:
    """
    Manages turn detection with waiting and timeout logic.

    Handles the scenario where a user pauses mid-utterance:
    1. VAD detects silence
    2. Check turn completion
    3. If incomplete: wait for more speech (up to max_wait)
    4. If complete OR timeout: proceed to transcription
    """

    def __init__(
        self,
        detector: SmartTurnDetector,
        max_wait: float = 3.0,
        check_interval: float = 0.1,
    ):
        """
        Initialize turn detection manager.

        Args:
            detector: SmartTurnDetector instance
            max_wait: Maximum time to wait for turn completion (seconds)
            check_interval: How often to check for new audio (seconds)
        """
        self.detector = detector
        self.max_wait = max_wait
        self.check_interval = check_interval

        # State for waiting
        self._waiting_tasks: dict[int, asyncio.Task] = {}

    async def check_turn_complete(
        self,
        user_id: int,
        audio: np.ndarray,
        audio_callback: Optional[callable] = None,
    ) -> tuple[bool, float, bool]:
        """
        Check if turn is complete, potentially waiting for more speech.

        Args:
            user_id: User ID
            audio: Current audio accumulation
            audio_callback: Async callback to get updated audio (returns np.ndarray)

        Returns:
            Tuple of (is_complete, confidence, timed_out)
            - is_complete: True if turn complete or timed out
            - confidence: Turn completion probability
            - timed_out: True if max_wait exceeded
        """
        # Check turn completion
        is_complete, confidence = await self.detector.detect_async(audio)

        if is_complete:
            logger.debug(
                f"User {user_id} turn complete "
                f"(confidence: {confidence:.3f})"
            )
            return True, confidence, False

        # Turn not complete - wait for more speech (if callback provided)
        if audio_callback is None:
            # No way to get more audio, consider complete
            logger.debug(
                f"User {user_id} turn incomplete "
                f"(confidence: {confidence:.3f}) but no callback, proceeding"
            )
            return True, confidence, False

        # Wait for more speech
        logger.debug(
            f"User {user_id} turn incomplete "
            f"(confidence: {confidence:.3f}), waiting up to {self.max_wait}s"
        )

        start_time = asyncio.get_event_loop().time()

        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= self.max_wait:
                logger.debug(
                    f"User {user_id} max wait exceeded ({elapsed:.1f}s), "
                    f"forcing completion"
                )
                return True, confidence, True

            # Wait for new audio
            await asyncio.sleep(self.check_interval)

            # Get updated audio
            try:
                updated_audio = await audio_callback()
                if updated_audio is None or len(updated_audio) == len(audio):
                    # No new audio yet
                    continue

                # New audio available - check turn completion again
                audio = updated_audio
                is_complete, confidence = await self.detector.detect_async(audio)

                if is_complete:
                    logger.debug(
                        f"User {user_id} turn complete after waiting "
                        f"(confidence: {confidence:.3f}, elapsed: {elapsed:.1f}s)"
                    )
                    return True, confidence, False

            except Exception as e:
                logger.error(f"Error getting updated audio: {e}")
                # On error, proceed with what we have
                return True, confidence, True

    def cancel_waiting(self, user_id: int) -> None:
        """
        Cancel waiting for a user (e.g., if they leave or speak again).

        Args:
            user_id: User ID
        """
        if user_id in self._waiting_tasks:
            task = self._waiting_tasks.pop(user_id)
            task.cancel()
            logger.debug(f"Cancelled turn detection wait for user {user_id}")

    def cancel_all(self) -> None:
        """Cancel all waiting tasks."""
        for user_id in list(self._waiting_tasks.keys()):
            self.cancel_waiting(user_id)

        logger.debug("Cancelled all turn detection waits")


# Convenience function for basic usage
async def create_turn_detector(
    model_path: Optional[Path] = None,
    threshold: float = 0.7,
    max_wait: float = 3.0,
) -> TurnDetectionManager:
    """
    Create a turn detector with default settings.

    Args:
        model_path: Path to model (None = auto-download)
        threshold: Turn completion threshold
        max_wait: Maximum wait time

    Returns:
        TurnDetectionManager instance
    """
    detector = SmartTurnDetector(
        model_path=model_path,
        threshold=threshold,
    )

    manager = TurnDetectionManager(
        detector=detector,
        max_wait=max_wait,
    )

    return manager
