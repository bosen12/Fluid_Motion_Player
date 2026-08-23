from __future__ import annotations

from pathlib import Path
from typing import Any

from fluid_motion.config import Settings
from fluid_motion.core.gpu import snapshot as gpu_snapshot
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc
from fluid_motion.core.vs_script import RifeParams, parse_fps, rife_label, target_multi, write_vpy
from fluid_motion.paths import engine_cache_dir, script_output_path


FILTER_LABEL = "@fluid"


def _strip_other_vapoursynth(ipc: MpvIpc) -> None:
    """SVP4 and Fluid cannot share the same mpv graph; SVP wins and RIFE never runs."""
    try:
        vf = ipc.get("vf")
    except IpcError:
        return
    if not isinstance(vf, list):
        return
    for item in vf:
        if not isinstance(item, dict):
            continue
        if item.get("name") != "vapoursynth":
            continue
        label = item.get("label") or ""
        if label == "fluid":
            continue
        try:
            ipc.command("vf", "remove", f"@{label}" if label else "vapoursynth")
        except IpcError:
            continue


def _vf_arg(script: Path | None = None) -> str:
    # Use ~~ so the path has no drive colon; mpv splits vf args on ':'.
    # concurrent-frames must stay 1: RIFE is temporal; parallel requests tile-flicker.
    return f'{FILTER_LABEL}:vapoursynth="~~/shaders/fluid_rife.vpy":4:1'


def current_filters(ipc: MpvIpc) -> str:
    try:
        value = ipc.get("vf")
    except IpcError:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "")


def vf_is_fluid(value: Any) -> bool:
    text = str(value or "").lower()
    if "vapoursynth" not in text:
        return False
    return "fluid" in text or "rife" in text


def interpolation_active(ipc: MpvIpc) -> bool:
    try:
        return vf_is_fluid(ipc.get("vf"))
    except IpcError:
        return False


SHORTFALL_RATIO = 0.12
SEEK_HOLD_MAX_AGE = 2.0


def interpolation_held_off(
    seeking: bool,
    hold_age: float | None,
    max_age: float = SEEK_HOLD_MAX_AGE,
) -> bool:
    """True while mpv is seeking or lua still has the post-seek quiet file."""
    if seeking:
        return True
    if hold_age is None:
        return False
    return 0 <= hold_age < max_age


def as_fps(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    frac = parse_fps(value)
    return float(frac) if frac else None


def live_source_fps(container: Any, estimated: Any, interpolating: bool) -> float | None:
    """Decoder rate when it still looks like the file; otherwise container FPS.

    After RIFE, some mpv builds report estimated-vfps as the filtered output.
    That must not replace the source readout.
    """
    container_f = as_fps(container)
    estimated_f = as_fps(estimated)
    if interpolating and container_f and estimated_f and estimated_f > container_f * 1.3:
        return container_f
    return estimated_f or container_f


def measured_output_fps(
    display_fps: Any,
    vsync_ratio: Any,
    estimated_display: Any = None,
) -> float | None:
    """Presented video FPS: display refresh divided by mpv vsync-ratio."""
    display = as_fps(estimated_display) or as_fps(display_fps)
    vsync = as_fps(vsync_ratio)
    if display is None or vsync is None or vsync < 0.2:
        return None
    out = display / vsync
    if out < 1 or out > 500:
        return None
    return out


def live_fps_label(value: float | None) -> str:
    if value is None or value <= 0:
        return "—"
    for known, label in ((23.976, "23.976"), (29.97, "29.97"), (59.94, "59.94")):
        if abs(value - known) < 0.04:
            return label
    if abs(value - round(value)) < 0.08:
        return str(int(round(value)))
    return f"{value:.1f}"


def output_shortfall(
    actual: float | None,
    target: float | None,
    *,
    paused: bool,
    interpolating: bool,
) -> bool:
    if paused or not interpolating:
        return False
    if actual is None or target is None or target <= 1:
        return False
    return actual < target * (1.0 - SHORTFALL_RATIO)


def snapshot_playback(ipc: MpvIpc) -> dict[str, Any]:
    def _get(name: str, default: Any = None) -> Any:
        try:
            return ipc.get(name)
        except IpcError:
            return default

    media = _get("media-title") or _get("filename") or ""
    width = int(_get("width") or 0)
    height = int(_get("height") or 0)
    container = _get("container-fps")
    estimated = _get("estimated-vfps")
    display = _get("display-fps")
    est_display = _get("estimated-display-fps")
    vsync = _get("vsync-ratio")
    paused = bool(_get("pause") or False)
    seeking = bool(_get("seeking") or False)
    interpolating = interpolation_active(ipc)
    source = live_source_fps(container, estimated, interpolating)
    return {
        "media": str(media),
        "width": width,
        "height": height,
        "fps": source if source is not None else "",
        "container_fps": as_fps(container),
        "estimated_vfps": as_fps(estimated),
        "display_fps": as_fps(display),
        "estimated_display_fps": as_fps(est_display),
        "vsync_ratio": as_fps(vsync),
        "paused": paused,
        "seeking": seeking,
        "interpolation": interpolating,
        "vf": current_filters(ipc),
    }


def apply(ipc: MpvIpc, settings: Settings, mpv_root: Path, *, announce: bool = False) -> Path:
    info = snapshot_playback(ipc)
    source = parse_fps(info.get("container_fps") or info.get("fps"))
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
        gpu_name=gpu_snapshot().name,
    )
    write_vpy(script, params, source_fps=source, display_fps=display)
    multi, _ = target_multi(settings.profile, source, display)
    if float(multi) < 1.5:
        remove(ipc)
        return script
    _strip_other_vapoursynth(ipc)
    vf = current_filters(ipc)
    if FILTER_LABEL in vf or vf_is_fluid(vf):
        try:
            ipc.command("vf", "remove", FILTER_LABEL)
        except IpcError:
            pass
    try:
        ipc.command("vf", "add", _vf_arg(script), timeout=30)
    except IpcError as exc:
        if interpolation_active(ipc):
            return script
        raise IpcError(f"無法加入補幀濾鏡：{exc}") from exc
    try:
        ipc.set("interpolation", False)
        ipc.set("video-sync", "display-resample")
        ipc.set("hr-seek-framedrop", False)
        ipc.set("temporal-dither", False)
        # Do not switch gpu-api here: vulkan→d3d11 at runtime recreates vo and
        # leaves mpv running with no visible window.
        ipc.set("window-minimized", False)
    except IpcError:
        pass
    if announce:
        try:
            ipc.command("show-text", f"Fluid Motion  {rife_label(settings.rife_model)} TensorRT")
        except IpcError:
            pass
    return script


def remove(ipc: MpvIpc, *, announce: bool = False) -> None:
    try:
        ipc.command("vf", "remove", FILTER_LABEL)
    except IpcError:
        pass
    try:
        vf = ipc.get("vf")
    except IpcError:
        vf = []
    items = vf if isinstance(vf, list) else []
    for item in items:
        if not isinstance(item, dict) or item.get("name") != "vapoursynth":
            continue
        blob = str(item).lower()
        if "fluid" not in blob and "rife" not in blob:
            continue
        label = item.get("label") or ""
        try:
            ipc.command("vf", "remove", f"@{label}" if label else "vapoursynth")
        except IpcError:
            continue
    if announce:
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
