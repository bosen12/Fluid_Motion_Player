from __future__ import annotations

from pathlib import Path
from typing import Any

from fluid_motion.config import Settings
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc
from fluid_motion.core.vs_script import RifeParams, parse_fps, write_vpy
from fluid_motion.paths import engine_cache_dir, script_output_path


FILTER_LABEL = "@fluid"


def _vf_arg(script: Path) -> str:
    posix = script.as_posix()
    return f"{FILTER_LABEL}:vapoursynth={posix}:8:4"


def current_filters(ipc: MpvIpc) -> str:
    try:
        value = ipc.get("vf")
    except IpcError:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "")


def interpolation_active(ipc: MpvIpc) -> bool:
    vf = current_filters(ipc)
    return FILTER_LABEL in vf or "vapoursynth" in vf.lower() and "fluid_rife" in vf


def snapshot_playback(ipc: MpvIpc) -> dict[str, Any]:
    def _get(name: str, default: Any = None) -> Any:
        try:
            return ipc.get(name)
        except IpcError:
            return default

    media = _get("media-title") or _get("filename") or ""
    width = int(_get("width") or 0)
    height = int(_get("height") or 0)
    fps = _get("container-fps") or _get("estimated-vfps") or ""
    if isinstance(fps, float):
        fps = f"{fps:.3f}".rstrip("0").rstrip(".")
    display = _get("display-fps")
    paused = bool(_get("pause") or False)
    return {
        "media": str(media),
        "width": width,
        "height": height,
        "fps": str(fps),
        "display_fps": float(display) if isinstance(display, (int, float)) else None,
        "paused": paused,
        "interpolation": interpolation_active(ipc),
        "vf": current_filters(ipc),
    }


def apply(ipc: MpvIpc, settings: Settings, mpv_root: Path) -> Path:
    info = snapshot_playback(ipc)
    source = parse_fps(info.get("fps"))
    display = info.get("display_fps")
    script = script_output_path(mpv_root)
    params = RifeParams(
        model=settings.rife_model,
        scene_threshold=settings.scene_threshold,
        trt_streams=settings.trt_streams,
        fp16=settings.fp16,
        cuda_graph=settings.cuda_graph,
        profile=settings.profile,
        engine_folder=str(engine_cache_dir()),
        mpv_root=str(mpv_root),
    )
    write_vpy(script, params, source_fps=source, display_fps=display)
    vf = current_filters(ipc)
    if FILTER_LABEL in vf:
        try:
            ipc.command("vf", "remove", FILTER_LABEL)
        except IpcError:
            pass
    ipc.command("vf", "add", _vf_arg(script))
    try:
        ipc.set("interpolation", False)
        ipc.set("video-sync", "display-resample")
        ipc.set("hr-seek-framedrop", False)
    except IpcError:
        pass
    ipc.command("show-text", "Fluid Motion  RIFE 4.6 TensorRT", 1800)
    return script


def remove(ipc: MpvIpc) -> None:
    vf = current_filters(ipc)
    if FILTER_LABEL in vf:
        ipc.command("vf", "remove", FILTER_LABEL)
    try:
        ipc.command("show-text", "Fluid Motion  off", 1200)
    except IpcError:
        pass


def fps_fraction_label(value: Any) -> str:
    frac = parse_fps(value)
    if frac is None:
        return "—"
    if frac.denominator in (1, 1000, 1001):
        approx = float(frac)
        if abs(approx - 23.976) < 0.02:
            return "23.976"
        if abs(approx - 29.97) < 0.02:
            return "29.97"
        if abs(approx - 59.94) < 0.02:
            return "59.94"
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{approx:.3f}".rstrip("0").rstrip(".")
    return f"{float(frac):.3f}".rstrip("0").rstrip(".")
