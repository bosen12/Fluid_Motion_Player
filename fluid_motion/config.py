from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fluid_motion.paths import config_path, default_mpv_root


PROFILES = ("2x", "3x", "60", "120", "144", "display")
RIFE_MODELS = (426, 425, 46, 4251)  # 4251 = RIFE 4.25 lite (faster, lower quality)


@dataclass
class Settings:
    enabled: bool = False
    profile: str = "2x"
    scene_threshold: float = 0.10
    trt_streams: int = 1
    fp16: bool = True
    cuda_graph: bool = False
    start_hidden: bool = False
    autostart: bool = False
    mpv_root: str = field(default_factory=lambda: str(default_mpv_root()))
    rife_model: int = 426  # vsmlrt RIFEModel.v4_26 — 4.6 paints a waffle grid
    force_accel: bool = False  # user override: skip the RTX 50 flicker-safe gate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        settings = cls(**known)
        if settings.profile not in PROFILES:
            settings.profile = "2x"
        try:
            settings.rife_model = int(settings.rife_model)
        except (TypeError, ValueError):
            settings.rife_model = 426
        if settings.rife_model not in RIFE_MODELS:
            settings.rife_model = 426
        settings.scene_threshold = min(0.30, max(0.02, float(settings.scene_threshold)))
        settings.trt_streams = min(4, max(1, int(settings.trt_streams)))
        return settings


def load_settings(path: Path | None = None) -> Settings:
    target = path or config_path()
    if not target.is_file():
        settings = Settings()
        save_settings(settings, target)
        return settings
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return Settings()
        return Settings.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(settings.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
