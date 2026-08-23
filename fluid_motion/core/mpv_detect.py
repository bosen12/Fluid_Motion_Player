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
    output_fps: str = ""
    target_fps: str = ""
    fps_ok: bool = True
    settling: bool = False
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


def _is_helper_mpv(proc: Any) -> bool:
    """thumbfast / screenshot helpers spawn a second mpv without scripts."""
    try:
        cmd = proc.info.get("cmdline") or proc.cmdline() or []
    except (psutil.Error, OSError, TypeError):
        return False
    text = " ".join(str(part) for part in cmd).lower()
    return "thumbfast" in text or "--load-scripts=no" in text


def iter_mpv_processes() -> list[PlayerProcess]:
    found: list[PlayerProcess] = []
    if psutil is None:
        return found
    # pid+name only: process_iter fetches every requested field for *every*
    # process on the box, and on Windows exe/cmdline each cost a handle open
    # plus a PEB read -- ~20ms for 600 processes, 3.3 times a second, forever.
    # Both are only ever needed for the handful that are actually mpv.
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name not in PLAYER_NAMES:
                continue
            if _is_helper_mpv(proc):
                continue
            try:
                exe = proc.exe() or ""
            except (psutil.Error, OSError):
                exe = ""
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


def list_win_pipes() -> list[str]:
    if os.name != "nt":
        return []
    try:
        return os.listdir(r"\\.\pipe\\")
    except OSError:
        return []


def candidate_pipes(pid: int, extra: Iterable[str] = ()) -> list[str]:
    socket_path = str(windows_temp_dir() / "mpvSockets" / str(pid))
    names = [
        rf"\\.\pipe\fluid-mpv-{pid}",
        rf"\\.\pipe\mpvpipe",
        rf"\\.\pipe\mpv-{pid}",
        rf"\\.\pipe\mpvsocket",
        rf"\\.\pipe\{socket_path}",
        socket_path,
    ]
    names.extend(extra)
    needle = str(pid)
    for raw in list_win_pipes():
        low = raw.lower()
        if needle not in raw and f"fluid-mpv-{pid}" not in low:
            continue
        if "thumbfast" in low:
            continue
        names.append(raw if raw.startswith("\\\\.\\pipe\\") else rf"\\.\pipe\{raw}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
