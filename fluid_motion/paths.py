from __future__ import annotations

import os
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


def default_mpv_root() -> Path:
    candidates = [
        Path(r"C:\mpv"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "mpv",
        Path.home() / "mpv",
    ]
    for path in candidates:
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
