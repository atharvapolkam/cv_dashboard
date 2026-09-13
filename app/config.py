"""Application settings.

All tunables live here so deployment changes never require code edits.
Override any field with an environment variable prefixed ``CVD_``,
e.g. ``CVD_MAX_UPLOAD_MB=64``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CVD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---
    app_name: str = "CV Dashboard"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # --- paths ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    static_dir: Path = BASE_DIR / "static"
    templates_dir: Path = BASE_DIR / "templates"

    # --- uploads ---
    max_upload_mb: int = 32
    allowed_upload_ext: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm",
    )
    # Images larger than this on the long side are downscaled on ingest.
    # Set to 0 to disable. Protects both RAM and round-trip latency.
    ingest_max_side: int = 4096

    # --- image store ---
    # In-memory decoded-array cache. Disk is the source of truth.
    mem_cache_mb: int = 512
    session_ttl_minutes: int = 240
    max_images_per_session: int = 400
    max_history_per_session: int = 300

    # --- render / preview ---
    render_format: str = "webp"          # webp | jpeg | png
    render_quality: int = 88
    render_cache_entries: int = 256
    # Live-preview downscale ceiling used when ?preview=1
    preview_max_side: int = 900

    # --- execution guards ---
    op_timeout_seconds: float = 20.0
    max_contours_returned: int = 3000
    max_hough_returned: int = 2000

    # --- camera / future RTSP source ---
    enable_rtsp: bool = False
    rtsp_read_timeout_ms: int = 5000

    cors_origins: tuple[str, ...] = Field(default=("*",))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def mem_cache_bytes(self) -> int:
        return self.mem_cache_mb * 1024 * 1024

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
