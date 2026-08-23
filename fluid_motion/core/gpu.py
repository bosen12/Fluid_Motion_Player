from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from fluid_motion.core.proc import run_hidden


@dataclass
class GpuSnapshot:
    name: str = ""
    utilization: int = 0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    power_w: float = 0.0
    driver: str = ""
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CACHE: tuple[float, GpuSnapshot] | None = None
_CACHE_TTL = 1.5


def _run_smi(args: list[str]) -> str:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return ""
    try:
        result = run_hidden([smi, *args], timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def snapshot() -> GpuSnapshot:
    global _CACHE
    now = time.monotonic()
    if _CACHE is not None and now - _CACHE[0] < _CACHE_TTL:
        return _CACHE[1]
    raw = _run_smi(
        [
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        snap = GpuSnapshot()
        _CACHE = (now, snap)
        return snap
    line = raw.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        snap = GpuSnapshot()
        _CACHE = (now, snap)
        return snap

    def _int(value: str) -> int:
        try:
            return int(float(value))
        except ValueError:
            return 0

    def _float(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    snap = GpuSnapshot(
        name=parts[0],
        utilization=_int(parts[1]),
        memory_used_mb=_int(parts[2]),
        memory_total_mb=_int(parts[3]),
        power_w=_float(parts[4]),
        driver=parts[5],
        available=True,
    )
    _CACHE = (now, snap)
    return snap
