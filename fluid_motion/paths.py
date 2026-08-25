from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def app_root() -> Path:
    if is_frozen():
        bundled = bundle_root() / "fluid_motion"
        if bundled.is_dir():
            return bundled
        return bundle_root()
    return Path(__file__).resolve().parent


def project_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return app_root().parent


def launch_command() -> str:
    if is_frozen():
        return f'"{Path(sys.executable)}"'
    return f'"{sys.executable}" -m fluid_motion'


def ui_dir() -> Path:
    return app_root() / "ui"


def resources_dir() -> Path:
    return app_root() / "resources"


def roaming_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = base / "FluidMotion"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return roaming_dir() / "config.json"


def engine_cache_dir() -> Path:
    path = roaming_dir() / "engines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_dir() -> Path:
    path = roaming_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def heartbeat_path() -> Path:
    return roaming_dir() / "alive"


def hotkey_path() -> Path:
    return roaming_dir() / "hotkey"


def seek_hold_path() -> Path:
    return roaming_dir() / "seek_hold"


def debug_log_path() -> Path:
    return roaming_dir() / "fluid_debug.log"


def mpv_root_candidates() -> list[Path]:
    """Places an mpv install plausibly lives, best guess first.

    Only a fallback: once a player is running its own config-dir is
    authoritative and gets used instead (Engine.install_root). This list
    covers the case where Fluid Motion is opened before any player has
    started. The previous three entries recognised only a hand-placed
    C:\\mpv -- every package-manager install, and AX Player's own bundled
    runtime, looked like "no mpv found".
    """
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    candidates = [
        Path(r"C:\mpv"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "mpv",
        Path.home() / "mpv",
        # AX Player's packaged build fetches its mpv here and runs out of it
        # in preference to anything else on the machine.
        local / "AXPlayer" / "mpv-runtime",
        Path.home() / "scoop" / "apps" / "mpv" / "current",
        Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "chocolatey" / "lib" / "mpv",
    ]
    on_path = shutil.which("mpv")
    if on_path:
        candidates.append(Path(on_path).parent)
    return candidates


def default_mpv_root() -> Path:
    for path in mpv_root_candidates():
        if (path / "mpv.exe").is_file():
            return path
    return Path(r"C:\mpv")


def script_output_path(mpv_root: Path) -> Path:
    shaders = mpv_root / "shaders"
    shaders.mkdir(parents=True, exist_ok=True)
    return shaders / "fluid_rife.vpy"


def lua_output_path(mpv_root: Path) -> Path:
    scripts = mpv_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    return scripts / "zz-fluid-ipc.lua"
