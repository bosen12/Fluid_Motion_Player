from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fluid_motion.paths import config_path, default_mpv_root


PROFILES = ("2x", "3x", "60", "120", "144", "display")
RIFE_MODELS = (426, 425, 46)
# 4251 (4.25 lite) is not offered: vsmlrt TensorRT on RTX 50 copies frame n
# instead of interpolating, which reads as a flashing green/wrong picture.


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
        # Pinned, not clamped. Every extra TensorRT stream builds its own
        # execution context when the VapourSynth script loads, and mpv reloads
        # that script on every seek -- so the cost is paid per seek, forever.
        # Nothing can use those contexts: RIFE is temporal, so both
        # core.num_threads (vs_script) and mpv's concurrent-frames (inject)
        # are held at 1 for correctness, leaving exactly one frame request in
        # flight. Measured on an RTX 5070 Ti at 1080p, 4 streams cost +0.23s
        # of stall after each seek and +1.1 GB of VRAM (+4.5 GB at 4K) for
        # identical output fps.
        #
        # The slider that used to set this is gone from the UI, so this also
        # has to migrate whatever an older version left in the config file --
        # otherwise anyone who had raised it stays stuck there with no way back.
        settings.trt_streams = 1
        # Pinned for the same reason, one step further on. The flicker gate
        # this used to lift controls exactly two things -- num_streams and
        # use_cuda_graph -- and both are now fixed: streams above, and
        # cuda_graph has no way to be set and defaults off. Measured, turning
        # this on changes nothing in the generated .vpy but one comment line,
        # while the checkbox promised a risky speedup that does not exist.
        # (cuda_graph itself was measured too: no init or throughput
        # difference at this pipeline's one-frame-in-flight concurrency.)
        settings.force_accel = False
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
