"""Structured logging with per-module configuration and latency tracking."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


# Global logger registry
_loggers: dict[str, logging.Logger] = {}
_latency_tracking_enabled: bool = True


def setup_logging(config: LoggingConfig) -> None:
    """
    Initialize logging system with configuration.

    Args:
        config: Logging configuration object
    """
    global _latency_tracking_enabled

    # Set latency tracking flag
    _latency_tracking_enabled = config.track_latency

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if config.rotation.get("enabled", False):
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                config.file,
                maxBytes=config.rotation.get("max_bytes", 10485760),
                backupCount=config.rotation.get("backup_count", 5),
            )
        else:
            file_handler = logging.FileHandler(config.file)

        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure per-module log levels
    for module_name, level in config.modules.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level.upper()))

    root_logger.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Logger name (typically __name__ of calling module)

    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


@contextmanager
def log_latency(logger: logging.Logger, operation: str, level: int = logging.DEBUG):
    """
    Context manager to track and log operation latency.

    Usage:
        with log_latency(logger, "transcribe_audio"):
            result = transcribe(audio)

    Args:
        logger: Logger instance
        operation: Operation name for logging
        level: Log level for latency message
    """
    if not _latency_tracking_enabled:
        yield
        return

    start_time = time.perf_counter()
    exception_occurred = False

    try:
        yield
    except Exception:
        exception_occurred = True
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if exception_occurred:
            logger.log(
                level,
                f"{operation} FAILED after {elapsed_ms:.2f}ms",
            )
        else:
            logger.log(
                level,
                f"{operation} completed in {elapsed_ms:.2f}ms",
            )


class LatencyTracker:
    """
    Track cumulative latency across multiple operations.

    Usage:
        tracker = LatencyTracker()

        with tracker.track("vad"):
            detect_speech(audio)

        with tracker.track("stt"):
            transcribe(audio)

        logger.info(tracker.summary())
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = {}
        self._current_operation: Optional[str] = None
        self._operation_start: Optional[float] = None

    @contextmanager
    def track(self, operation: str):
        """Track latency for an operation."""
        if not _latency_tracking_enabled:
            yield
            return

        self._current_operation = operation
        self._operation_start = time.perf_counter()

        try:
            yield
        finally:
            if self._operation_start is not None:
                elapsed = time.perf_counter() - self._operation_start
                if operation not in self._timings:
                    self._timings[operation] = []
                self._timings[operation].append(elapsed)

            self._current_operation = None
            self._operation_start = None

    def get_timing(self, operation: str) -> Optional[float]:
        """
        Get total time for an operation in milliseconds.

        Args:
            operation: Operation name

        Returns:
            Total time in ms, or None if operation not tracked
        """
        if operation not in self._timings:
            return None

        return sum(self._timings[operation]) * 1000

    def get_average(self, operation: str) -> Optional[float]:
        """
        Get average time for an operation in milliseconds.

        Args:
            operation: Operation name

        Returns:
            Average time in ms, or None if operation not tracked
        """
        if operation not in self._timings:
            return None

        timings = self._timings[operation]
        return (sum(timings) / len(timings)) * 1000

    def total_time_ms(self) -> float:
        """Get total time across all operations in milliseconds."""
        total = 0.0
        for timings in self._timings.values():
            total += sum(timings)
        return total * 1000

    def summary(self) -> str:
        """
        Generate a summary of all tracked operations.

        Returns:
            Formatted summary string
        """
        if not self._timings:
            return "No operations tracked"

        lines = ["Latency Summary:"]
        for operation, timings in self._timings.items():
            total_ms = sum(timings) * 1000
            count = len(timings)
            avg_ms = total_ms / count
            lines.append(f"  {operation}: {total_ms:.2f}ms total ({count}x, avg {avg_ms:.2f}ms)")

        lines.append(f"  TOTAL: {self.total_time_ms():.2f}ms")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked timings."""
        self._timings.clear()


# Example usage function for testing
def _example_usage():
    """Example of how to use logging utilities."""
    from .config import LoggingConfig

    # Setup logging
    config = LoggingConfig(level="DEBUG", track_latency=True)
    setup_logging(config)

    # Get logger
    logger = get_logger(__name__)

    # Simple logging
    logger.info("Starting operation")
    logger.debug("Debug information")

    # Latency tracking - single operation
    with log_latency(logger, "expensive_operation"):
        time.sleep(0.1)  # Simulate work

    # Latency tracking - multiple operations
    tracker = LatencyTracker()

    with tracker.track("step_1"):
        time.sleep(0.05)

    with tracker.track("step_2"):
        time.sleep(0.03)

    with tracker.track("step_1"):  # Same operation again
        time.sleep(0.02)

    logger.info(tracker.summary())


if __name__ == "__main__":
    _example_usage()
