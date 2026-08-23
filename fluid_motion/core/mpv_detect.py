from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from fluid_motion.paths import default_mpv_root

PLAYER_NAMES = {"mpv.exe", "mpv.com", "mpvnet.exe", "mpv.net.exe"}


@dataclass
class PlayerProcess:
    pid: int
    name: str
    exe: str = ""
    title: str = ""
    pipe: str = ""
    connected: bool = False
    media: str = ""
    width: int = 0
    height: int = 0
    fps: str = ""
    estimated_vfps: str = ""
    interpolation: bool = False
    paused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_mpv_executable(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        candidate = path / "mpv.exe"
        if candidate.is_file():
            return candidate
    root = default_mpv_root()
    if (root / "mpv.exe").is_file():
        return root / "mpv.exe"
    which = shutil.which("mpv")
    if which:
        return Path(which)
    return None


def mpv_root_from(exe: Path | None = None, configured: str | None = None) -> Path:
    if configured:
        path = Path(configured)
        if path.is_dir():
            return path
        if path.is_file():
            return path.parent
    if exe:
        return exe.parent
    return default_mpv_root()


def iter_mpv_processes() -> list[PlayerProcess]:
    found: list[PlayerProcess] = []
    if psutil is None:
        return found
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name not in PLAYER_NAMES:
                continue
            exe = proc.info.get("exe") or ""
            found.append(PlayerProcess(pid=int(proc.info["pid"]), name=name, exe=exe))
        except (psutil.Error, TypeError, ValueError):
            continue
    found.sort(key=lambda item: item.pid)
    return found


def windows_temp_dir() -> Path:
    for key in ("TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return Path.home() / "AppData" / "Local" / "Temp"


def candidate_pipes(pid: int, extra: Iterable[str] = ()) -> list[str]:
    names = [
        rf"\\.\pipe\fluid-mpv-{pid}",
        rf"\\.\pipe\mpvpipe",
        rf"\\.\pipe\mpv-{pid}",
        rf"\\.\pipe\mpvsocket",
        str(windows_temp_dir() / "mpvSockets" / str(pid)),
    ]
    names.extend(extra)
    # preserve order, drop dups
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
