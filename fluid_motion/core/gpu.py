from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from fluid_motion.core.proc import run_hidden

_RTX_SERIES_RE = re.compile(r"RTX\s*(\d{4,5})", re.IGNORECASE)

NVIDIA = "nvidia"
AMD = "amd"
INTEL = "intel"
UNKNOWN = "unknown"

# Matched against the adapter name in order, so the first hit wins. Kept as
# substrings rather than a model table: a table of every SKU is wrong the month
# after it is written, and the vendor word is in the name on every real adapter
# Windows reports.
_VENDOR_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (NVIDIA, ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")),
    (AMD, ("amd", "radeon", "firepro", "vega")),
    (INTEL, ("intel", "arc ", "iris", "uhd graphics", "hd graphics")),
)

# The display-adapter class GUID. Every GPU driver registers a subkey here with
# a DriverDesc value, which is the same string Device Manager shows.
_ADAPTER_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

_ADAPTERS: list[str] | None = None


def vendor_of(name: str) -> str:
    """Infer the vendor from an adapter name. Pure, so it is directly testable."""
    text = (name or "").lower()
    for vendor, keys in _VENDOR_KEYS:
        if any(key in text for key in keys):
            return vendor
    return UNKNOWN


def detect_adapters(*, refresh: bool = False) -> list[str]:
    """Every display adapter Windows knows about, by name.

    Read from the driver registry rather than through WMI/CIM: this runs on the
    UI's path and spawning PowerShell for a value that cannot change while the
    machine is on costs more than the whole rest of the query. Cached for the
    process for the same reason.

    An empty list means "could not tell", not "no GPU" -- callers must not read
    it as grounds for switching a backend off.
    """
    global _ADAPTERS
    if _ADAPTERS is not None and not refresh:
        return list(_ADAPTERS)
    names: list[str] = []
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows-only app
        _ADAPTERS = names
        return list(names)
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _ADAPTER_CLASS_KEY) as root:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                # Only the numbered instance keys hold a driver; siblings like
                # "Properties" and "Configuration" do not.
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as node:
                        desc, _kind = winreg.QueryValueEx(node, "DriverDesc")
                except OSError:
                    continue
                if isinstance(desc, str) and desc.strip():
                    names.append(desc.strip())
    except OSError:
        pass
    _ADAPTERS = names
    return list(names)


def available_vendors(*, refresh: bool = False) -> set[str]:
    """Vendors with a display adapter present, nvidia-smi as a second opinion.

    nvidia-smi answering is proof of an NVIDIA GPU even when the registry read
    came back empty, which keeps a machine that today works on TensorRT working
    if the registry walk ever fails.
    """
    vendors = {vendor_of(name) for name in detect_adapters(refresh=refresh)}
    vendors.discard(UNKNOWN)
    if snapshot().available:
        vendors.add(NVIDIA)
    return vendors


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


def flicker_risk(name: str) -> bool:
    """RTX 50-series (Blackwell) tile-flickers with CUDA graphs / multiple TRT streams.

    Names we can't parse (unknown GPU, nvidia-smi unavailable) are treated as
    risky too — better to give up some throughput than ship visible corruption.
    """
    match = _RTX_SERIES_RE.search(name or "")
    if not match:
        return True
    series = int(match.group(1)[:2])
    return series == 50
