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


# What to call each player on screen. The process name is only what the OS
# knows it as, and for a host that embeds libmpv that is whatever wraps it --
# "python.exe" for AX Player run from source, "AXPlayer.exe" when packaged.
# The UI called every player "mpv" regardless, which is wrong for all of them
# except mpv, and leaves two open players looking identical.
_PLAYER_LABELS = {
    "mpv.exe": "mpv",
    "mpv.com": "mpv",
    "mpvnet.exe": "mpv.net",
    "mpv.net.exe": "mpv.net",
    "axplayer.exe": "AX Player",
}


def display_label(name: str, exe: str = "") -> str:
    known = _PLAYER_LABELS.get((name or "").lower())
    if known:
        return known
    stem = Path(exe).stem if exe else ""
    if stem and stem.lower() not in ("python", "pythonw"):
        return stem
    # An embedded host we cannot name from its process. Still better than
    # "mpv", which is the one thing it is definitely not.
    return "內嵌播放器"


@dataclass
class PlayerProcess:
    pid: int
    name: str
    exe: str = ""
    label: str = ""
    title: str = ""
    # The config dir this player itself reports over IPC -- the directory the
    # .vpy is written into and loaded from, which is not necessarily the one
    # settings.mpv_root points at. See Engine._player_root.
    config_dir: str = ""
    # Whether *that* directory has a usable TensorRT runtime. Defaults True so
    # a player we have not diagnosed yet is never blocked by this.
    ready: bool = True
    missing: str = ""
    # Set when the IPC script was installed into this player's config dir
    # while it was already running: mpv only loads scripts at launch.
    needs_restart: bool = False
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
    # Measured playback speed as a multiple of realtime, None until two
    # usable samples exist. Saturates at 1.0 -- see inject.realtime_ratio.
    realtime: float | None = None
    realtime_label: str = "—"
    # Share of target frames mpv discarded rather than present. Realtime can
    # be held by dropping frames, so this is the other half of "is it smooth".
    drop_rate: float | None = None
    playback_clean: bool | None = None

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


def _embedded_player_pids(seen: set[int]) -> set[int]:
    """Hosts that embed libmpv in-process (e.g. AX Player via python-mpv) never
    show up under mpv's own process name -- their OS process is python.exe or
    whatever wraps it.

    zz-fluid-ipc.lua is supposed to open a `fluid-mpv-<pid>` named pipe, but
    in practice this mpv config also ships mpvSockets.lua, and Lua scripts
    load alphabetically -- "zz-" guarantees zz-fluid-ipc.lua loads *last*.
    mpvSockets.lua claims `input-ipc-server` first (binding it to
    `%TEMP%\\mpvSockets\\<pid>`); mpv's IPC listener is only ever bound once,
    so zz-fluid-ipc.lua's later rebind to `fluid-mpv-<pid>` is silently a
    no-op and that pipe never actually exists. This is true for *every* mpv
    instance using this config, embedded or standalone -- confirmed by
    checking a live instance's pipes directly. mpvSockets.lua's own
    pid-named pipe is the one that's real, and it's already among
    candidate_pipes()'s fallback names, so just discover the pid from it.
    """
    pids: set[int] = set()
    for raw in list_win_pipes():
        low = raw.lower()
        if "thumbfast" in low:
            continue
        if low.startswith("fluid-mpv-"):
            pid_str = raw[len("fluid-mpv-") :]
        elif "mpvsockets" in low:
            pid_str = raw.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        else:
            continue
        if pid_str.isdigit():
            pid = int(pid_str)
            if pid not in seen:
                pids.add(pid)
    return pids


def iter_mpv_processes() -> list[PlayerProcess]:
    found: list[PlayerProcess] = []
    if psutil is None:
        return found
    seen: set[int] = set()
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
            pid = int(proc.info["pid"])
            found.append(PlayerProcess(pid=pid, name=name, exe=exe, label=display_label(name, exe)))
            seen.add(pid)
        except (psutil.Error, TypeError, ValueError):
            continue
    for pid in _embedded_player_pids(seen):
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            exe = proc.exe() or ""
        except psutil.NoSuchProcess:
            # The pipe named a pid that is not running -- it outlived its owner,
            # or the name was never a player's to begin with. Skipped rather
            # than described: the fallback below exists for a process that is
            # there but will not say what it is, and reusing it here invented a
            # player instead. That is not only a wrong card in the UI. Player
            # count drives allow_ambiguous in tick(), so one phantom entry
            # takes a genuinely single-player machine off the ambiguous pipe
            # names -- which for an mpv.conf that sets
            # input-ipc-server=mpvsocket are the only way in at all.
            continue
        except (psutil.Error, OSError):
            name, exe = "embedded-mpv", ""
        found.append(PlayerProcess(pid=pid, name=name, exe=exe, label=display_label(name, exe)))
        seen.add(pid)
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


def pipe_names_pid(raw: str, pid: int) -> bool:
    """True when this pipe name belongs to *this* pid, not merely contains it.

    A plain substring test matches the wrong process: pid 234 matches pid
    12345's `mpvSockets\12345`, and connecting to it hands one mpv's IPC to
    another mpv's PlayerProcess -- the filter is then applied twice to one
    player while the other is never touched, and the applied-settings
    bookkeeping is keyed to the wrong pid.
    """
    name = raw.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    needle = str(pid)
    return name == needle or name.lower() == f"fluid-mpv-{needle}" or name.endswith(f"-{needle}")


# Pipe names an mpv can be reached on that say nothing about *which* mpv it
# is. Someone whose mpv.conf sets input-ipc-server=mpvsocket is reachable
# only through one of these, so they are worth trying -- but with two players
# open they are actively wrong: both pids resolve to the same pipe, the filter
# is applied twice to one player and never to the other, and the
# applied-settings bookkeeping is keyed to whichever pid asked first. Same
# failure the pid-substring fix closed, through a door it left open.
_AMBIGUOUS_PIPES = (r"\\.\pipe\mpvpipe", r"\\.\pipe\mpvsocket")


def candidate_pipes(pid: int, extra: Iterable[str] = (), *, allow_ambiguous: bool = True) -> list[str]:
    """Pipes to try for this pid, best-identified first.

    allow_ambiguous=False drops the names that cannot identify a player. The
    watcher passes False whenever more than one player is present, which is
    exactly when guessing costs something and never when it is the only way in.
    """
    socket_path = str(windows_temp_dir() / "mpvSockets" / str(pid))
    names = [
        rf"\\.\pipe\fluid-mpv-{pid}",
        rf"\\.\pipe\mpv-{pid}",
        rf"\\.\pipe\{socket_path}",
        socket_path,
    ]
    if allow_ambiguous:
        names.extend(_AMBIGUOUS_PIPES)
    names.extend(extra)
    for raw in list_win_pipes():
        low = raw.lower()
        if not pipe_names_pid(raw, pid):
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
