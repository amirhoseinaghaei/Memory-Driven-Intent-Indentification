import json
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError


# Avoid proxy issues when accessing internal network
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",mlops.huawei.com"


class Settings(BaseSettings):

    # -----------------------------
    # request / streaming timeouts
    # -----------------------------
    REQUEST_TIMEOUT: int = 300
    STREAM_CONNECTION_TIMEOUT: int = 10
    STREAM_READ_TIMEOUT: int = 30

    # -----------------------------
    # session
    # -----------------------------
    SESSION_EXPIRE_TIME: int = 3 * 60 * 60
    SESSION_CLEANUP_CYCLE: int = 60

    # -----------------------------
    # LLM
    # -----------------------------
    API_KEY: str
    API_BASE: str
    MODEL_NAME: str = "qwen3-32b"
    NO_THINK_MODEL_NAME: str = "qwen3-235b"
    TEMPERATURE: float = 0.4
    MAX_OUTPUT_TOKEN: int = 2048
    RETRY_TIMES: int = 3

    # -----------------------------
    # embeddings
    # -----------------------------
    EMBEDDING_API_BASE: str
    EMBEDDING_MODEL: str

    # -----------------------------
    # neo4j
    # -----------------------------
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # -----------------------------
    # Pydantic config
    # -----------------------------
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def load_settings_from_json(config_path: str | Path) -> Settings:

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    try:
        settings = Settings(**data)

    except ValidationError as e:
        raise ValueError(f"Config validation failed:\n{e}")

    print(f"[CONFIG] Loaded config from {config_path}")
    print(f"[CONFIG] LLM model: {settings.MODEL_NAME}")
    print(f"[CONFIG] Embedding model: {settings.EMBEDDING_MODEL}")

    return settings


# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"
settings = load_settings_from_json(CONFIG_PATH)