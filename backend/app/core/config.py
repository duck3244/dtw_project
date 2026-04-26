from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DTW Speech Recognition API"
    api_prefix: str = "/api"

    sample_rate: int = 16000
    n_mfcc: int = 13
    dtw_backend: str = "auto"
    dtw_band: int | None = None
    score_aggregation: str = "min"
    knn_k: int = 3
    normalize: bool = True

    max_audio_bytes: int = 10 * 1024 * 1024
    max_evaluate_batch_bytes: int = 50 * 1024 * 1024
    max_evaluate_files: int = 200
    allowed_audio_suffixes: tuple[str, ...] = (".wav", ".flac", ".ogg", ".webm", ".mp3", ".m4a")

    # Persistence paths
    templates_path: Path = BACKEND_ROOT / "data" / "templates.pkl"  # legacy pickle (read-only after migration)
    store_db_path: Path = BACKEND_ROOT / "data" / "store.db"
    templates_dir: Path = BACKEND_ROOT / "data" / "templates"
    backup_dir: Path = BACKEND_ROOT / "data" / "backups"

    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v):
        # Accept comma-separated env value: "http://a,http://b"
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


settings = Settings()
