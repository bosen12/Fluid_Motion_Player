from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fluid_motion.paths import engine_cache_dir


@dataclass
class CacheInfo:
    count: int = 0
    total_bytes: int = 0
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def info(path: Path | None = None) -> CacheInfo:
    target = path or engine_cache_dir()
    count = 0
    total = 0
    if target.is_dir():
        for entry in target.rglob("*"):
            if not entry.is_file():
                continue
            count += 1
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return CacheInfo(count=count, total_bytes=total, path=str(target))


def clear(path: Path | None = None) -> tuple[int, int]:
    """Delete every cached TensorRT engine file. Returns (deleted, locked)."""
    target = path or engine_cache_dir()
    deleted = 0
    locked = 0
    if not target.is_dir():
        return deleted, locked
    for entry in list(target.rglob("*")):
        if not entry.is_file():
            continue
        try:
            entry.unlink()
            deleted += 1
        except OSError:
            locked += 1
    for entry in sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            entry.rmdir()
        except OSError:
            pass
    return deleted, locked


def next_growth_deadline(current_bytes: int, previous_bytes: int, now: float, prior_deadline: float, grace: float) -> float:
    """Extend the "still compiling" deadline whenever the cache grew since the last sample.

    There's no way to know vsmlrt's engine filename for a given resolution/model
    ahead of time, so this treats "the cache directory is actively growing" as
    the signal that a TensorRT engine is being built right now.
    """
    if current_bytes > previous_bytes:
        return now + grace
    return prior_deadline
