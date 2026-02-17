"""Configuration loading with YAML and environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class DiscordConfig(BaseModel):
    """Discord bot configuration."""

    token: Optional[str] = None
    command_prefix: str = "/"
    status_message: str = "Listening in voice channels"
    auto_join: bool = False

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: Optional[str]) -> Optional[str]:
        """Validate Discord token is provided."""
        if v is None or v.strip() == "":
            env_token = os.getenv("DISCORD_TOKEN")
            if env_token:
                return env_token
            raise ValueError(
                "Discord token is required. Set DISCORD_TOKEN environment variable."
            )
        return v


class AgentVoiceConfig(BaseModel):
    """Per-agent voice configuration."""

    voice_file: str
    personality: str
    emotion_exaggeration: float = Field(ge=0.0, le=1.0, default=0.3)


class AgentsConfig(BaseModel):
    """Agents configuration."""

    default: str = "jarvis"
    jarvis: AgentVoiceConfig
    sage: AgentVoiceConfig


class OpenClawConfig(BaseModel):
    """OpenClaw Gateway WebSocket configuration."""

    base_url: Optional[str] = None
    token: Optional[str] = None
    timeout: float = 8.0
    retry_timeout: float = 15.0
    max_retries: int = 1
    model: str = "claude-sonnet-4"
    agent_id: str = "main"
    session_scope: str = "per-peer"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Get base URL from environment if not set."""
        if v is None or v.strip() == "":
            return os.getenv("OPENCLAW_BASE_URL")
        return v

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: Optional[str]) -> Optional[str]:
        """Get token from environment if not set."""
        if v is None or v.strip() == "":
            return os.getenv("OPENCLAW_AUTH_TOKEN")
        return v

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Get agent ID from environment if set."""
        env_value = os.getenv("OPENCLAW_AGENT_ID")
        return env_value if env_value else v


class VADConfig(BaseModel):
    """Voice activity detection configuration."""

    silence_threshold: float = 0.3
    min_speech_duration: float = 0.5
    speech_threshold: float = Field(ge=0.0, le=1.0, default=0.5)


class TurnDetectionConfig(BaseModel):
    """Smart Turn detection configuration."""

    threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    max_wait: float = 3.0
    model_path: str = "smart_turn_v3.onnx"


class STTConfig(BaseModel):
    """Speech-to-text configuration."""

    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    language: Optional[str] = "en"
    vad_filter: bool = False


class RelevanceConfig(BaseModel):
    """Relevance filter configuration."""

    default_sensitivity: str = "medium"
    thresholds: Dict[str, float] = {
        "low": 1.0,
        "medium": 0.75,
        "high": 0.5,
    }
    classifier: str = "openclaw"
    timeout: float = 2.0
    enable_cache: bool = True
    cache_ttl: int = 300


class TranscriptConfig(BaseModel):
    """Transcript management configuration."""

    window_duration: int = 90
    max_turns: int = 20
    timezone: str = "America/Los_Angeles"


class CoquiTTSConfig(BaseModel):
    """Coqui TTS specific configuration."""

    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    language: str = "en"
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 5.0
    top_k: int = 50
    top_p: float = 0.85


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    engine: str = "coqui"
    device: str = "cuda"
    streaming: bool = True
    chunk_duration: float = 0.5
    coqui: CoquiTTSConfig


class AudioConfig(BaseModel):
    """Audio buffering configuration."""

    buffer_duration: float = 10.0
    processing_sample_rate: int = 16000
    discord_sample_rate: int = 48000


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    vad: VADConfig
    turn_detection: TurnDetectionConfig
    stt: STTConfig
    relevance: RelevanceConfig
    transcript: TranscriptConfig
    tts: TTSConfig
    audio: AudioConfig


class CORSConfig(BaseModel):
    """CORS configuration."""

    enabled: bool = True
    allowed_origins: list[str] = ["*"]
    allowed_methods: list[str] = ["*"]
    allowed_headers: list[str] = ["*"]


class ServerConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8880
    enable_tts: bool = True
    enable_stt: bool = True
    api_key: Optional[str] = None
    cors: CORSConfig

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Get API key from environment if not set."""
        if v is None or v.strip() == "":
            return os.getenv("SERVER_API_KEY")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    track_latency: bool = True
    modules: Dict[str, str] = {}
    file: Optional[str] = None
    rotation: Dict[str, Any] = {}


class Config(BaseModel):
    """Main configuration."""

    discord: DiscordConfig
    agents: AgentsConfig
    openclaw: OpenClawConfig
    pipeline: PipelineConfig
    server: ServerConfig
    logging: LoggingConfig


def apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config dictionary.

    Environment variables use format: SECTION__SUBSECTION__KEY
    Example: PIPELINE__STT__MODEL_SIZE=large-v3
    """
    for key, value in os.environ.items():
        if "__" not in key:
            continue

        parts = key.lower().split("__")
        current = config_dict

        # Navigate to the nested location
        for part in parts[:-1]:
            if part not in current:
                break
            current = current[part]
        else:
            # Set the value
            final_key = parts[-1]
            if final_key in current:
                # Try to preserve type
                original_type = type(current[final_key])
                try:
                    if original_type == bool:
                        current[final_key] = value.lower() in ("true", "1", "yes")
                    elif original_type == int:
                        current[final_key] = int(value)
                    elif original_type == float:
                        current[final_key] = float(value)
                    else:
                        current[final_key] = value
                except (ValueError, TypeError):
                    current[final_key] = value

    return config_dict


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to config.yaml (default: ./config.yaml)

    Returns:
        Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    # Load .env file if it exists
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    # Determine config file path
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Apply environment variable overrides
    config_dict = apply_env_overrides(config_dict)

    # Validate and return
    return Config(**config_dict)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """Get the models directory."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_voices_dir() -> Path:
    """Get the voices directory."""
    voices_dir = get_project_root() / "server" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    return voices_dir
