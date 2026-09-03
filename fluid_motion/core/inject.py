from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fluid_motion.config import Settings
from fluid_motion.core.gpu import available_vendors
from fluid_motion.core.gpu import snapshot as gpu_snapshot
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc
from fluid_motion.core.vs_script import (
    RifeParams,
    parse_fps,
    resolve_backend,
    rife_label,
    target_multi,
    write_vpy,
)
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


_PLAYBACK_PROPS = (
    ("interpolation", False),
    ("video-sync", "display-resample"),
    ("hr-seek-framedrop", False),
    ("temporal-dither", False),
)


def _sync_playback_props(ipc: MpvIpc) -> None:
    """Set only the playback props that actually differ. Never touch window-minimized:
    flipping it on every apply un-minimizes mpv for no reason.
    """
    for name, want in _PLAYBACK_PROPS:
        try:
            current = ipc.get(name)
        except IpcError:
            current = None
        if current == want:
            continue
        try:
            ipc.set(name, want)
        except IpcError:
            pass


def _vf_arg(script: Path | None = None) -> str:
    # `script` is deliberately unused (test_vf_arg_uses_label pins that):
    # the path is always the ~~/-relative one, and ~~ is *the player's own
    # config dir*. That is the constraint on the caller -- apply() has to be
    # handed the config dir of the mpv that will load this, or write_vpy()
    # puts the .vpy in one place while this points at another and mpv says
    # only "could not init VS". player_config_dir(ipc) is that value; the
    # settings.mpv_root fallback is right only when the two agree.
    # Use ~~ so the path has no drive colon; mpv splits vf args on ':'.
    # concurrent-frames must stay 1: RIFE is temporal; parallel requests tile-flicker.
    return f'{FILTER_LABEL}:vapoursynth="~~/shaders/fluid_rife.vpy":4:1'


def format_filters(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "")


def current_filters(ipc: MpvIpc) -> str:
    try:
        return format_filters(ipc.get("vf"))
    except IpcError:
        return ""


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


def fluid_filter_loaded(ipc: MpvIpc) -> bool:
    """True only when *our* labelled filter is on the graph.

    interpolation_active() accepts any vapoursynth filter whose description
    merely mentions rife/fluid. That is the right question for "is there
    something here to tear down", but too loose for confirming an apply()
    landed: a leftover from SVP or an older build answers yes, and apply()
    would then report success while mpv runs somebody else's filter.
    """
    try:
        vf = ipc.get("vf")
    except IpcError:
        return False
    if isinstance(vf, list):
        return any(isinstance(item, dict) and item.get("label") == "fluid" for item in vf)
    return FILTER_LABEL in str(vf or "")


SHORTFALL_RATIO = 0.12
SEEK_HOLD_MAX_AGE = 2.0
SETTLE_SECONDS = 5.0


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


# How much a new realtime sample is allowed to move the running figure. The
# tick rate is 0.3s and time-pos advances in whole decoded frames, so a raw
# sample is coarse enough to swing several percent on its own.
REALTIME_SMOOTHING = 0.35
# Below this the pipeline is not delivering realtime playback.
REALTIME_SHORTFALL = 0.97
# Share of the target frames that may be thrown away before playback stops
# looking smooth. Measured on real content: a comfortable setting discards
# 0.0-0.3% and looks clean, while one that visibly stutters discards ~10%.
DROP_LIMIT = 0.01


def realtime_ratio(
    delta_pos: float,
    delta_wall: float,
    speed: float = 1.0,
    previous: float | None = None,
    smoothing: float = REALTIME_SMOOTHING,
) -> float | None:
    """How fast playback is actually advancing, as a multiple of realtime.

    1.0 means the pipeline is keeping up. Lower means it cannot: the 4K
    measurements that made this worth showing sat at 0.32, and nothing in the
    UI said so -- the output fps readout is clamped to the presentation rate,
    so it reads a healthy 48 whether there is 4x of headroom or none at all.

    Deliberately has no meaning above 1.0: mpv presents frames at the target
    rate and no faster, so a pipeline with room to spare is indistinguishable
    from one that is exactly keeping up. Headroom cannot be measured from
    playback, only shortfall can.

    Returns None when the sample cannot say anything -- no elapsed time, a
    seek or file change moving the position backwards or by a jump, or a
    stopped clock.
    """
    if delta_wall <= 0 or speed <= 0:
        return None
    # Backwards or implausibly far: a seek, a loop, or a new file. Either way
    # the pair of samples does not describe playback speed.
    if delta_pos < 0 or delta_pos > delta_wall * speed * 4 + 1.0:
        return None
    ratio = delta_pos / (delta_wall * speed)
    if previous is None:
        return ratio
    return previous + (ratio - previous) * smoothing


def realtime_label(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    return f"{min(ratio, 1.0):.2f}×"


def drop_ratio(
    delta_drops: float,
    delta_wall: float,
    target_fps: float | None,
    previous: float | None = None,
    smoothing: float = REALTIME_SMOOTHING,
) -> float | None:
    """Share of the target frames mpv threw away rather than present.

    Realtime playback is not the same as smooth playback: when the filter
    cannot keep up, mpv can hold the clock by discarding frames instead of
    slowing down, so realtime_ratio reads a healthy 1.00 while the picture
    visibly stutters. Measured on real content at 120fps, that is exactly what
    happened -- 1.00x while discarding 9.6% of frames -- and the readout
    called it fine. This is the other half of that question.

    Returns None when the sample cannot say anything: no elapsed time, no
    known target, or a counter that went backwards (a new file resets it).
    """
    if delta_wall <= 0 or not target_fps or target_fps <= 0:
        return None
    if delta_drops < 0:
        return None
    ratio = (delta_drops / delta_wall) / target_fps
    if previous is None:
        return ratio
    return previous + (ratio - previous) * smoothing


def playback_is_clean(ratio: float | None, drops: float | None) -> bool | None:
    """Both halves have to hold: realtime *and* not paid for by dropping frames."""
    if ratio is None:
        return None
    if ratio < REALTIME_SHORTFALL:
        return False
    if drops is not None and drops >= DROP_LIMIT:
        return False
    return True


def as_fps(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    frac = parse_fps(value)
    return float(frac) if frac else None


def live_source_fps(container: Any, estimated: Any, interpolating: bool) -> float | None:
    """Decoder rate when it still looks like the file; otherwise container FPS.

    After RIFE, some mpv builds report estimated-vf-fps as the filtered output.
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
    """Presented video FPS: display refresh divided by mpv vsync-ratio.

    Fallback only -- see filtered_output_fps. Both inputs are live estimates,
    so dividing one by the other multiplies their jitter, and when the
    pipeline struggles vsync-ratio dips below 1 and the quotient explodes:
    at 165Hz a ratio of 0.41 reads as 402fps. Nothing can be presented faster
    than the panel refreshes, so anything above the refresh rate is the
    estimate breaking down rather than a real measurement.
    """
    display = as_fps(estimated_display) or as_fps(display_fps)
    vsync = as_fps(vsync_ratio)
    if display is None or vsync is None or vsync < 0.2:
        return None
    out = display / vsync
    refresh = max(filter(None, (as_fps(display_fps), as_fps(estimated_display))), default=None)
    if out < 1 or (refresh and out > refresh * 1.05):
        return None
    return out


def filtered_output_fps(
    estimated_vfps: Any,
    display_fps: Any,
    vsync_ratio: Any,
    estimated_display: Any = None,
    interpolating: bool = False,
) -> float | None:
    """What the filter chain is actually putting out.

    mpv reports this directly as estimated-vf-fps, which is both exact and
    steady: measured against a 72fps target it sat at 72.0 for every sample,
    while the display/vsync quotient wandered between 70.6 and 74.4 and, on a
    struggling pipeline, as far as 400. The quotient was only ever here
    because this property was being read under a name that does not exist, so
    it always came back empty -- with that fixed there is no reason to prefer
    the derived figure over the reported one.
    """
    if interpolating:
        direct = as_fps(estimated_vfps)
        if direct is not None:
            return direct
    return measured_output_fps(display_fps, vsync_ratio, estimated_display)


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


def is_settling(short: bool, now: float, settling_until: float) -> bool:
    """True while a just-applied filter change is still expected to be catching up.

    A vf remove+add tears down and rebuilds the whole VapourSynth/TensorRT
    pipeline, which visibly lags behind the new target for a few seconds —
    that isn't a real performance shortfall, so don't report it as one.
    """
    return short and now < settling_until


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
    # mpv's property is estimated-vf-fps. The name used here until now,
    # estimated-vfps, does not exist, so _get's IpcError guard swallowed the
    # "property not found" reply and returned None every single time -- the
    # fps readout has been silently falling back to container-fps.
    estimated = _get("estimated-vf-fps")
    display = _get("display-fps")
    est_display = _get("estimated-display-fps")
    vsync = _get("vsync-ratio")
    paused = bool(_get("pause") or False)
    seeking = bool(_get("seeking") or False)
    time_pos = _get("time-pos")
    speed = _get("speed")
    drops = _get("frame-drop-count")
    # One read of vf, not two. interpolation_active() and current_filters()
    # each fetched it, so every player paid an extra IPC round trip three
    # times a second for a value that had just been read -- and the docstring
    # on rate_snapshot puts a round trip at ~15ms while mpv is busy running
    # the filter.
    # Read separately from _get: this is the one property whose *failure*
    # changes a decision. vf_is_fluid(None) is False, which is
    # indistinguishable from a genuine "no filter loaded", so a busy mpv --
    # the one compiling a TensorRT engine, i.e. the one least able to cope --
    # used to read as "the filter fell off" and get another vf add every
    # APPLY_RETRY_BACKOFF. Report "could not ask" as its own state.
    vf_ok = True
    try:
        vf_raw = ipc.get("vf")
    except IpcError:
        vf_raw = None
        vf_ok = False
    interpolating = vf_is_fluid(vf_raw)
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
        "time_pos": float(time_pos) if isinstance(time_pos, (int, float)) else None,
        "speed": float(speed) if isinstance(speed, (int, float)) and speed > 0 else 1.0,
        "drops": float(drops) if isinstance(drops, (int, float)) else None,
        "vf": format_filters(vf_raw),
        # False means `interpolation` above is a guess, not a reading. Absent
        # from rate_snapshot on purpose: that one is only reached from a user
        # click, which applies unconditionally and never asks "is it missing".
        "vf_ok": vf_ok,
    }


def rate_snapshot(ipc: MpvIpc) -> dict[str, Any]:
    """Just the fields the multiplier depends on, and nothing else.

    snapshot_playback reads fourteen properties because the UI wants all of
    them, and each round trip costs real time -- around 15ms while mpv's main
    thread is busy running the filter, so the full set is roughly 230ms. Both
    set_enabled and apply() were paying that to work out one integer, twice
    per settings change, on top of the tick that follows.
    """
    def _get(name: str, default: Any = None) -> Any:
        try:
            return ipc.get(name)
        except IpcError:
            return default

    container = _get("container-fps")
    estimated = _get("estimated-vf-fps")
    interpolating = interpolation_active(ipc)
    source = live_source_fps(container, estimated, interpolating)
    return {
        "container_fps": as_fps(container),
        "fps": source if source is not None else "",
        "display_fps": as_fps(_get("display-fps")),
        "interpolation": interpolating,
    }


def resolve_multi(info: dict[str, Any], settings: Settings) -> int:
    """The integer multiplier apply() will actually use for this playback state.

    The watcher needs the exact same number apply() computes in order to tell
    whether the loaded filter still matches the current settings, so both sides
    read it from here rather than each deriving it from a slightly different
    fps field.
    """
    source = parse_fps(info.get("container_fps") or info.get("fps"))
    multi, _ = target_multi(settings.profile, source, info.get("display_fps"))
    return int(multi)


# hwdec modes whose frames already land in system memory. VapourSynth is a
# CPU-side filter, so a GPU-resident frame (plain "auto", "auto-safe",
# "d3d11va", "nvdec", ...) cannot be fed to it and the filter fails to load.
# Copy-back modes and software decoding are both fine.
# Keyed by pid. It used to be keyed by ipc.path, which is not an identity:
# an mpv that exits without a clean remove() leaves its entry behind, and the
# next player to land on the same pipe name -- easy when the name carries no
# pid at all -- would have that stale value restored onto it. Pids are not
# reused while the process is alive, and forget_hwdec() clears the entry when
# a player goes away.
_PREV_HWDEC: dict[int, str] = {}
_COPYBACK_HWDEC = "auto-copy"


def _hwdec_state_path() -> Path:
    from fluid_motion.paths import roaming_dir

    return roaming_dir() / "hwdec_state.json"


def _write_hwdec_state() -> None:
    """Mirror the saved modes to disk so a crash does not strand a player.

    The map only ever lived in memory, and nothing else knows the original
    value: the lua strips a stale @fluid when this app stops answering its
    heartbeat, but it has no idea what hwdec was before and never touches it.
    So a Fluid Motion that is killed while interpolating leaves the player on
    copy-back for the rest of its life -- paying the GPU->CPU transfer for a
    filter that is no longer there, with no symptom beyond "it got slower".
    Observed exactly that against a live player while testing.

    Process start time is stored alongside the pid because a pid on its own is
    not an identity across a restart of this app.
    """
    payload = {}
    for pid, value in _PREV_HWDEC.items():
        payload[str(pid)] = {"hwdec": value, "started": _process_started(pid)}
    try:
        path = _hwdec_state_path()
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _process_started(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 -- identity is a nicety, not a requirement
        return None


def stranded_hwdec() -> dict[int, str]:
    """Players a previous run left on copy-back, by pid.

    Only entries whose process is still the one we saved for: a pid alone can
    be reused, and restoring a stale mode onto somebody else's player would be
    this bug with the sign flipped.
    """
    try:
        raw = json.loads(_hwdec_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or not str(key).isdigit():
            continue
        pid = int(key)
        value = entry.get("hwdec")
        if not isinstance(value, str) or not value:
            continue
        started = entry.get("started")
        if isinstance(started, (int, float)):
            live = _process_started(pid)
            if live is None or abs(live - float(started)) > 1.0:
                continue
        out[pid] = value
    return out


def _hwdec_text(value: Any) -> str:
    """hwdec is a *priority list* option, so mpv hands it back as a list.

    str() on that yields "['auto-safe']", which is neither a valid value to
    set back nor something the suffix test below can read -- it has to be
    flattened to mpv's own comma-separated form first.
    """
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value or "")


def _hwdec_is_copyback(value: Any) -> bool:
    text = _hwdec_text(value).strip().lower()
    if text in ("", "no"):
        return True
    # Every entry has to be safe: mpv walks the list and uses the first one
    # that works, so a single GPU-resident mode in there can still be chosen.
    entries = [part.strip() for part in text.split(",") if part.strip()]
    return bool(entries) and all(
        entry == "no" or entry.endswith("-copy") or entry.endswith("-copy-safe")
        for entry in entries
    )


def ensure_copyback_hwdec(ipc: MpvIpc, pid: int | None = None) -> None:
    """Switch the player to copy-back decoding for as long as RIFE is loaded.

    Previously this was left entirely to the user's mpv.conf ("hwdec=auto-copy"
    in the README) and nothing checked it: a player configured for
    GPU-resident decoding -- AX Player ships hwdec=auto-safe, and it is the
    mpv default in many configs -- would install everything correctly and then
    silently fail to interpolate, with no message saying why.

    Doing it over IPC rather than rewriting someone's mpv.conf keeps the cost
    where it belongs: copy-back's GPU->CPU transfer is only paid while
    interpolation is actually on, and normal playback is left alone.
    """
    try:
        current = ipc.get("hwdec")
    except IpcError:
        return
    if _hwdec_is_copyback(current):
        return
    try:
        ipc.set("hwdec", _COPYBACK_HWDEC)
    except IpcError:
        return
    if pid is not None:
        _PREV_HWDEC[pid] = _hwdec_text(current)
        _write_hwdec_state()


def restore_hwdec(ipc: MpvIpc, pid: int | None = None) -> None:
    """Put back whatever decoding mode the player had before we changed it."""
    if pid is None:
        return
    previous = _PREV_HWDEC.pop(pid, None)
    if previous is None:
        return
    _write_hwdec_state()
    try:
        ipc.set("hwdec", previous)
    except IpcError:
        # The pop comes first so a success cannot leave a stale record behind
        # -- but a refusal is not a success. mpv is still running, still on
        # copy-back, and the mode it had before now exists nowhere: the
        # remove() the watcher retries finds an empty map and restores
        # nothing, and neither does the next run's stranded-hwdec recovery.
        # So the player would pay copy-back's GPU->CPU transfer for the rest
        # of its life over one timed-out command.
        #
        # No special case for a player that has actually gone away: tick()
        # sweeps hwdec_pids() against the processes it can see, so a record
        # put back for a dead pid is dropped on the next pass.
        _PREV_HWDEC.setdefault(pid, previous)
        _write_hwdec_state()


def flush_hwdec_state() -> None:
    """Make the file say exactly what the live map says.

    The map is the single writer: everything that should survive is in it, and
    anything dropped from it is meant to be gone. Startup recovery ends with
    one call to this rather than rewriting per pid, so a record it decided to
    keep and one it decided to restore cannot end up in different states.
    """
    _write_hwdec_state()


def adopt_hwdec(pid: int, previous: str) -> None:
    """Take over a saved mode from a previous run of this app.

    For a player that still has the filter loaded when we start up: it is
    legitimately on copy-back and must stay there, but the value to put back
    when the filter finally comes off only exists in the file the last run
    wrote. Without adopting it, remove() would find nothing to restore and the
    player would be stranded anyway -- just later.
    """
    _PREV_HWDEC.setdefault(pid, previous)
    _write_hwdec_state()


def hwdec_pids() -> list[int]:
    """Players whose original decoding mode is still being held for them."""
    return list(_PREV_HWDEC)


def forget_hwdec(pid: int) -> None:
    """Drop a departed player's saved mode without trying to restore it."""
    if _PREV_HWDEC.pop(pid, None) is not None:
        _write_hwdec_state()


def player_config_dir(ipc: MpvIpc) -> Path | None:
    """Where this specific player keeps its mpv config, straight from mpv.

    The .vpy has to land in the config dir of the player that will load it,
    and one configured mpv_root cannot answer that for everyone: an embedded
    host (AX Player) runs out of its own runtime folder, and a standalone mpv
    lives wherever a package manager put it. mpv answers for itself over IPC,
    so ask instead of guessing -- this works for any mpv-based player,
    embedded or not, with no knowledge of how it was installed.
    """
    try:
        value = ipc.get("config-dir")
    except IpcError:
        return None
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_dir() else None


def apply(
    ipc: MpvIpc,
    settings: Settings,
    mpv_root: Path,
    *,
    announce: bool = False,
    info: dict[str, Any] | None = None,
    pid: int | None = None,
    backend: str | None = None,
) -> Path:
    # The caller has usually just read these; re-reading them costs another
    # round of IPC for values that cannot have changed in between.
    info = info if info is not None else rate_snapshot(ipc)
    source = parse_fps(info.get("container_fps") or info.get("fps"))
    display = info.get("display_fps")
    script = script_output_path(mpv_root)
    # Supplied by the watcher, which has already resolved it for _filter_key
    # and for the readiness verdict. Resolving it again here would be a second
    # independent answer to the same question, equal to the first only by
    # coincidence -- available_vendors() reads caches with their own expiry.
    # Deciding once and carrying the answer is what keeps the key, the
    # readiness check and the file mpv actually loads describing one backend.
    if backend is None:
        backend = resolve_backend(settings.backend, available_vendors())
    params = RifeParams(
        model=settings.rife_model,
        scene_threshold=settings.scene_threshold,
        trt_streams=settings.trt_streams,
        fp16=settings.fp16,
        cuda_graph=settings.cuda_graph,
        profile=settings.profile,
        # TensorRT builds engines ahead of time and caches them here; ncnn
        # builds none, and render_vpy drops this rather than emitting a path
        # to a backend that has nowhere to put it.
        engine_folder=str(engine_cache_dir()),
        mpv_root=str(mpv_root),
        gpu_name=gpu_snapshot().name,
        force_accel=settings.force_accel,
        backend=backend,
    )
    write_vpy(script, params, source_fps=source, display_fps=display)
    multi = resolve_multi(info, settings)
    # multi == 1 is target_multi's own sentinel for "source already at/above
    # target" — nothing to gain there. Anything above that (even 1.2x) is a
    # real ask and should be applied, not silently dropped.
    if float(multi) <= 1.0:
        remove(ipc, pid=pid)
        return script
    # Before the filter goes on: VapourSynth cannot read GPU-resident frames,
    # so a player left on plain auto/auto-safe would reject it.
    ensure_copyback_hwdec(ipc, pid)
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
        # The reply can time out while mpv is still building the pipeline, so a
        # labelled filter that is genuinely on the graph still counts as applied.
        if fluid_filter_loaded(ipc):
            return script
        raise IpcError(f"無法加入補幀濾鏡：{exc}") from exc
    if not fluid_filter_loaded(ipc):
        raise IpcError("補幀濾鏡已送出，但沒有掛上 mpv 的 vf")
    _sync_playback_props(ipc)
    if announce:
        try:
            ipc.command("show-text", f"Fluid Motion  {rife_label(settings.rife_model)} TensorRT")
        except IpcError:
            pass
    return script


def remove(ipc: MpvIpc, *, announce: bool = False, pid: int | None = None) -> None:
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
    # The filter is gone, so copy-back's GPU->CPU transfer is pure cost now.
    restore_hwdec(ipc, pid)
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
