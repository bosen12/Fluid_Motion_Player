from pathlib import Path

import pytest

from fluid_motion.paths import ui_dir

from fluid_motion.config import Settings, load_settings, save_settings
from fluid_motion.core.inject import FILTER_LABEL, _vf_arg
from fluid_motion.core.mpv_detect import candidate_pipes
from fluid_motion.core.runtime import diagnose


def test_settings_roundtrip(tmp_path: Path):
    path = tmp_path / "config.json"
    settings = Settings(enabled=True, profile="120")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.enabled is True
    assert loaded.profile == "120"
    assert loaded.rife_model == 426


def test_trt_streams_is_pinned_to_one_on_load(tmp_path: Path):
    """Anything a previous version stored has to be migrated back down.

    The slider that could raise this is gone, so a config left at 4 would
    otherwise keep paying the per-seek rebuild forever with no way back.
    """
    path = tmp_path / "config.json"
    save_settings(Settings(trt_streams=4), path)
    assert load_settings(path).trt_streams == 1
    assert Settings.from_dict({"trt_streams": 999}).trt_streams == 1
    assert Settings.from_dict({"trt_streams": 0}).trt_streams == 1
    assert Settings().trt_streams == 1


class _FakeIpc:
    """Duck-types MpvIpc without touching a real pipe/socket."""

    def __init__(self, props: dict):
        self.props = dict(props)
        self.commands: list[tuple] = []
        self.sets: list[tuple] = []
        self.vf: list = []
        self.path = "fake-mpv-pipe"
        self.closed = False

    def close(self):
        self.closed = True

    def get(self, name: str):
        if name == "vf":
            return self.vf
        return self.props.get(name)

    def set(self, name: str, value):
        self.sets.append((name, value))
        self.props[name] = value

    def command(self, *args, **kwargs):
        self.commands.append(args)
        if len(args) >= 2 and args[0] == "vf" and args[1] == "add":
            self.vf = [{"name": "vapoursynth", "label": "fluid"}]
        elif len(args) >= 2 and args[0] == "vf" and args[1] == "remove":
            self.vf = []
        return None


def test_apply_applies_when_target_rounds_to_a_real_multiplier(tmp_path: Path, monkeypatch):
    from fluid_motion.core.inject import apply

    monkeypatch.setenv("APPDATA", str(tmp_path))
    ipc = _FakeIpc({"container-fps": 50, "estimated-vfps": 50, "vsync-ratio": 1})
    settings = Settings(profile="120", mpv_root=str(tmp_path))
    apply(ipc, settings, tmp_path)
    add_calls = [c for c in ipc.commands if c[0] == "vf" and c[1] == "add"]
    assert add_calls, "120/50 = 2.4x rounds to a real 2x multiplier and should apply"


def test_apply_does_not_unminimize_or_reset_matching_props(tmp_path: Path, monkeypatch):
    from fluid_motion.core.inject import apply

    monkeypatch.setenv("APPDATA", str(tmp_path))
    ipc = _FakeIpc(
        {
            "container-fps": 24,
            "estimated-vfps": 24,
            "interpolation": False,
            "video-sync": "display-resample",
            "hr-seek-framedrop": False,
            "temporal-dither": False,
            "window-minimized": True,
        }
    )
    apply(ipc, Settings(profile="2x", mpv_root=str(tmp_path)), tmp_path)
    names = [name for name, _ in ipc.sets]
    assert "window-minimized" not in names
    assert ipc.props["window-minimized"] is True
    assert "interpolation" not in names
    assert "video-sync" not in names


def test_apply_skips_when_nearest_integer_multiplier_is_only_one(tmp_path: Path, monkeypatch):
    from fluid_motion.core.inject import apply

    monkeypatch.setenv("APPDATA", str(tmp_path))
    ipc = _FakeIpc({"container-fps": 100, "estimated-vfps": 100, "vsync-ratio": 1})
    settings = Settings(profile="120", mpv_root=str(tmp_path))
    apply(ipc, settings, tmp_path)
    add_calls = [c for c in ipc.commands if c[0] == "vf" and c[1] == "add"]
    assert not add_calls, "120/100 = 1.2x rounds to 1 -- vsmlrt only supports integer multi, nothing achievable"


def test_apply_skips_only_when_source_already_at_target(tmp_path: Path, monkeypatch):
    from fluid_motion.core.inject import apply

    monkeypatch.setenv("APPDATA", str(tmp_path))
    ipc = _FakeIpc({"container-fps": 120, "estimated-vfps": 120, "vsync-ratio": 1})
    settings = Settings(profile="120", mpv_root=str(tmp_path))
    apply(ipc, settings, tmp_path)
    add_calls = [c for c in ipc.commands if c[0] == "vf" and c[1] == "add"]
    assert not add_calls, "source already at the target fps has nothing to interpolate"


def test_apply_lock_serializes_concurrent_apply_calls():
    import threading
    import time as time_mod

    from fluid_motion.core.watcher import Engine

    engine = Engine(Settings())
    active = [0]
    max_concurrent = [0]
    guard = threading.Lock()

    def critical_section():
        with engine._apply_lock:
            with guard:
                active[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active[0])
            time_mod.sleep(0.05)
            with guard:
                active[0] -= 1

    threads = [threading.Thread(target=critical_section) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_concurrent[0] == 1, "tick() and set_enabled() must never apply() concurrently"


def test_apply_lock_guards_every_apply_and_remove_call_site():
    import ast

    from fluid_motion.core import watcher as watcher_mod

    src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine")

    # Every call that touches mpv's vf graph must funnel through the two
    # lock-guarded helpers, or the background loop and a Bridge call can race
    # and stomp on each other's vf remove+add / .vpy write.
    callers = {}
    for node in engine.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id in {"apply", "remove"}:
                callers.setdefault(node.name, set()).add(call.func.id)

    assert callers == {"_apply_to": {"apply"}, "_remove_from": {"remove"}}, callers

    for name in ("_apply_to", "_remove_from"):
        body = ast.get_source_segment(src, next(n for n in engine.body if getattr(n, "name", None) == name))
        assert "self._apply_lock.acquire(" in body, f"{name} must take the apply lock"
        assert "self._apply_lock.release()" in body, f"{name} must always release it"


def test_apply_lock_is_released_even_when_mpv_refuses(tmp_path, monkeypatch):
    """A raising apply() must not strand the lock -- everything else would wedge."""
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_ipc import IpcError

    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path)))

    def boom(*args, **kwargs):
        raise IpcError("nope")

    monkeypatch.setattr(watcher_mod, "apply", boom)
    monkeypatch.setattr(watcher_mod, "remove", boom)
    assert engine._apply_to(1, _FakeIpc({}), 2) is False
    assert engine._remove_from(1, _FakeIpc({})) is False
    assert engine._apply_lock.acquire(timeout=0), "the lock was left held"
    engine._apply_lock.release()


def test_ui_thread_settings_change_does_not_block_on_a_slow_apply(tmp_path, monkeypatch):
    """set_enabled() runs on the pywebview bridge thread.

    Waiting out a multi-second apply() there freezes the window, so it hands the
    work to tick() instead -- which is only safe because the applied-settings
    snapshot guarantees tick will actually pick it up.
    """
    import threading
    import time as time_mod

    from fluid_motion.core import watcher as watcher_mod

    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="120", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    with engine._lock:
        engine._ipc = {4321: ipc}

    # The background tick is mid-apply and holds the lock.
    engine._apply_lock.acquire()
    try:
        started = time_mod.monotonic()
        done = threading.Event()
        threading.Thread(target=lambda: (engine.set_enabled(True), done.set()), daemon=True).start()
        assert done.wait(timeout=5), "set_enabled() never returned"
        elapsed = time_mod.monotonic() - started
    finally:
        engine._apply_lock.release()

    assert elapsed < watcher_mod.UI_APPLY_WAIT + 1.5, f"UI thread blocked for {elapsed:.1f}s"
    assert 4321 not in engine._applied, "the deferred change must stay pending for tick()"


def test_no_streams_control_survives_in_the_ui():
    """The slider is gone on purpose, so nothing may reintroduce a way to set it.

    Raising it can only ever cost: extra TensorRT contexts are rebuilt on every
    seek and can never be used, because RIFE is temporal and concurrency is
    pinned to one frame in flight for correctness.
    """
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="streams"' not in html
    assert "streams-val" not in html
    assert "trt_streams" not in js
    assert "set_streams" not in js


def test_force_accel_is_pinned_off_on_load(tmp_path: Path):
    """The gate it lifted no longer controls anything, so it is migrated off.

    Both things safe mode fixes -- num_streams and use_cuda_graph -- are now
    pinned regardless, which leaves force_accel changing one comment line in
    the generated .vpy while the UI promised a risky speedup.
    """
    path = tmp_path / "config.json"
    save_settings(Settings(force_accel=True), path)
    assert load_settings(path).force_accel is False
    assert Settings.from_dict({"force_accel": True}).force_accel is False
    assert Settings().force_accel is False


def test_invalid_profile_falls_back():
    settings = Settings.from_dict({"profile": "nope"})
    assert settings.profile == "2x"


def test_invalid_rife_model_falls_back():
    settings = Settings.from_dict({"rife_model": 999})
    assert settings.rife_model == 426


def test_rife_425_lite_model_is_rejected():
    # vsmlrt TRT 4.25 lite does not interpolate on this stack (mid-frame is
    # essentially a copy of frame n). Keep it out of the picker.
    settings = Settings.from_dict({"rife_model": 4251})
    assert settings.rife_model == 426


def test_diagnose_missing_tree(tmp_path: Path):
    status = diagnose(tmp_path)
    assert status.ready is False
    ids = {c.id: c.ok for c in status.checks}
    assert ids["mpv"] is False
    assert ids["tensorrt"] is False
    assert ids["rife46"] is False


def test_diagnose_ready_when_files_present(tmp_path: Path):
    (tmp_path / "mpv.exe").write_bytes(b"mz")
    (tmp_path / "vapoursynth.dll").write_bytes(b"dll")
    (tmp_path / "python.exe").write_bytes(b"py")
    (tmp_path / "python312.dll").write_bytes(b"dll")
    plugins = tmp_path / "vs-plugins"
    plugins.mkdir()
    (plugins / "MiscFilters.dll").write_bytes(b"dll")
    (plugins / "vstrt.dll").write_bytes(b"dll")
    (plugins / "nvinfer.dll").write_bytes(b"dll")
    (tmp_path / "vsmlrt.py").write_text("x", encoding="utf-8")
    models = plugins / "models" / "rife"
    models.mkdir(parents=True)
    (models / "rife_v4.6.onnx").write_bytes(b"onnx")
    status = diagnose(tmp_path)
    assert status.ready is True
    ids = {c.id: c.ok for c in status.checks}
    assert "rife425lite" not in ids





def test_vf_is_fluid_detects_labeled_and_unlabeled():
    from fluid_motion.core.inject import vf_is_fluid

    assert vf_is_fluid([{"name": "vapoursynth", "params": {"file": "~~/shaders/fluid_rife.vpy"}}])
    assert vf_is_fluid([{"name": "vapoursynth", "label": "fluid"}])
    assert not vf_is_fluid([])
    assert not vf_is_fluid([{"name": "lavfi"}])


def test_apply_does_not_switch_gpu_api():
    from fluid_motion.core import inject as inject_mod

    src = Path(inject_mod.__file__).read_text(encoding="utf-8")
    assert 'ipc.set("gpu-api"' not in src


def test_auto_apply_is_silent_user_toggle_announces():
    from fluid_motion.core import inject as inject_mod
    from fluid_motion.core import watcher as watcher_mod

    inject_src = Path(inject_mod.__file__).read_text(encoding="utf-8")
    watcher_src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    assert "announce: bool = False" in inject_src
    # Only the two user-driven set_enabled() paths announce; the background
    # tick's reconciling apply must stay silent or mpv shows an OSD toast
    # every time it quietly catches up on a change.
    assert "announce=True" in watcher_src
    assert watcher_src.count("announce=True") == 2, "only the two set_enabled paths announce"
    # The background tick's reconciling apply must stay silent, or mpv shows an
    # OSD toast every time it quietly catches up on a change.
    assert "self._apply_to(player.pid, ipc, want_multi, wait=apply_wait)" in watcher_src


def test_vf_arg_uses_label():
    arg = _vf_arg(Path("C:/mpv/shaders/fluid_rife.vpy"))
    assert arg.startswith(FILTER_LABEL)
    assert 'vapoursynth="~~/shaders/fluid_rife.vpy"' in arg
    assert arg.endswith(":4:1")
    assert "C:" not in arg


def test_ensure_input_binding_strips_old_f3_toggle(tmp_path: Path):
    from fluid_motion.core.bootstrap import ensure_input_binding

    conf = tmp_path / "input.conf"
    conf.write_text(
        'F3 vf toggle @fluid:vapoursynth="~~/shaders/fluid_rife.vpy":8:4; '
        'show-text "Fluid Motion RIFE 4.6 TensorRT" 2500\n',
        encoding="utf-8",
    )
    ensure_input_binding(tmp_path)
    text = conf.read_text(encoding="utf-8")
    assert ":8:4" not in text
    assert not text.lstrip().startswith("F3")
    assert "zz-fluid-ipc.lua" in text


def test_candidate_pipes_include_fluid_and_mpvpipe():
    pipes = candidate_pipes(4242)
    assert any("fluid-mpv-4242" in p for p in pipes)
    assert any(p.endswith("mpvpipe") or p.endswith("\\mpvpipe") for p in pipes)


def test_candidate_pipes_include_mpvsockets_prefixed_path():
    from fluid_motion.core.mpv_detect import windows_temp_dir
    from fluid_motion.core.mpv_ipc import as_win_pipe

    pid = 11588
    sock = str(windows_temp_dir() / "mpvSockets" / str(pid))
    pipes = candidate_pipes(pid)
    assert as_win_pipe(sock) in pipes


def test_as_win_pipe_keeps_colon_and_backslashes():
    from fluid_motion.core.mpv_ipc import as_win_pipe

    raw = r"C:\Users\boshe\AppData\Local\Temp\mpvSockets\11588"
    assert as_win_pipe(raw) == r"\\.\pipe\C:\Users\boshe\AppData\Local\Temp\mpvSockets\11588"
    assert as_win_pipe(r"\\.\pipe\mpvpipe") == r"\\.\pipe\mpvpipe"


def test_hidden_run_kwargs_sets_no_window_on_windows():
    import os

    from fluid_motion.core.proc import CREATE_NO_WINDOW, hidden_run_kwargs

    kwargs = hidden_run_kwargs()
    if os.name == "nt":
        assert kwargs.get("creationflags") == CREATE_NO_WINDOW
        assert "startupinfo" in kwargs
    else:
        assert kwargs == {}


def test_launch_command_uses_module_when_not_frozen():
    from fluid_motion.paths import is_frozen, launch_command

    assert is_frozen() is False
    assert "-m fluid_motion" in launch_command()


def test_notify_running_writes_hotkey(tmp_path: Path, monkeypatch):
    from fluid_motion import single
    from fluid_motion.paths import roaming_dir

    monkeypatch.setenv("APPDATA", str(tmp_path))
    single.notify_running("show")
    text = (roaming_dir() / "hotkey").read_text(encoding="utf-8")
    assert text == "show"


def test_lua_f3_does_not_inject_vf():
    from fluid_motion.paths import resources_dir

    lua = (resources_dir() / "zz-fluid-ipc.lua").read_text(encoding="utf-8")
    assert "請先開啟 Fluid Motion" in lua
    assert "vf add" not in lua
    assert "hotkey" in lua
    assert "alive" in lua


def test_interpolation_held_off_during_seek_and_hold_file():
    from fluid_motion.core.inject import interpolation_held_off

    assert interpolation_held_off(seeking=True, hold_age=None) is True
    assert interpolation_held_off(seeking=False, hold_age=0.1) is True
    assert interpolation_held_off(seeking=False, hold_age=0.0) is True
    assert interpolation_held_off(seeking=False, hold_age=3.0) is False
    assert interpolation_held_off(seeking=False, hold_age=None) is False
    assert interpolation_held_off(seeking=False, hold_age=-0.1) is False


def test_lua_strips_vf_on_seek_without_readding():
    from fluid_motion.paths import resources_dir

    lua = (resources_dir() / "zz-fluid-ipc.lua").read_text(encoding="utf-8")
    assert "seek_hold" in lua
    assert 'register_event("seek"' in lua
    assert 'register_event("playback-restart"' in lua
    assert 'observe_property("seeking"' in lua
    assert "vf add" not in lua
    assert 'vf", "remove", "@fluid"' in lua


def test_lua_drops_the_filter_on_every_seek():
    """Unconditional, not gated on the seek looking like a drag.

    Gating it was measured and reverted: total time to interpolating is the
    same either way, but time to a *picture* is not -- 0.14s when the filter
    is dropped first against 0.34s when it has to rebuild in the chain, which
    is what makes a gated seek feel sluggish.
    """
    from fluid_motion.paths import resources_dir

    lua = (resources_dir() / "zz-fluid-ipc.lua").read_text(encoding="utf-8")
    assert "SCRUB_WINDOW" not in lua
    assert "last_seek_at" not in lua    # the drag detector's state
    assert "local function on_seek" not in lua
    for hook in ('register_event("seek"', 'observe_property("seeking"'):
        assert hook in lua
    # Both paths reach the teardown with no condition of their own in between.
    seek_body = lua.split('register_event("seek"', 1)[1].split("end)", 1)[0]
    assert "begin_seek_hold()" in seek_body
    assert "if" not in seek_body


def test_watcher_loop_keeps_a_plain_cadence():
    """No early wake on the seek hold clearing -- it was measured, not assumed.

    Cutting the wait short sounded free, but against the plain cadence the
    median time for interpolation to return went 0.71s -> 0.84s with a wider
    spread: the tick it saves is small next to the variance in rebuilding the
    VapourSynth script. The extra cross-thread signal bought nothing.
    """
    from fluid_motion.core import watcher as watcher_mod

    src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    assert "_wake" not in src
    loop = src.split("def _loop", 1)[1].split("def ", 1)[0]
    assert "self._stop.wait(TICK_SECONDS)" in loop


def test_titlebar_has_pywebview_drag_region():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    assert "pywebview-drag-region" in html
    assert "window-btns" in html


def test_live_source_prefers_estimated_unless_post_filter():
    from fluid_motion.core.inject import live_source_fps

    assert live_source_fps(30, 29.8, False) == 29.8
    assert live_source_fps(30, 59.9, True) == 30
    assert live_source_fps(30, None, True) == 30
    assert live_source_fps(None, 24.0, False) == 24.0


def test_measured_output_from_vsync_ratio():
    from fluid_motion.core.inject import measured_output_fps

    assert abs(measured_output_fps(165, 2.75) - 60) < 0.1
    assert abs(measured_output_fps(165, 5.5) - 30) < 0.1
    assert measured_output_fps(165, None) is None
    assert abs(measured_output_fps(60, 1, estimated_display=165) - 165) < 0.1


def test_is_settling_true_within_grace_window_when_short():
    from fluid_motion.core.inject import is_settling

    assert is_settling(short=True, now=10.0, settling_until=12.0) is True


def test_is_settling_false_after_grace_window_expires():
    from fluid_motion.core.inject import is_settling

    assert is_settling(short=True, now=13.0, settling_until=12.0) is False


def test_is_settling_false_when_not_short():
    from fluid_motion.core.inject import is_settling

    assert is_settling(short=False, now=10.0, settling_until=12.0) is False


def test_settling_indicator_wired_in_ui():
    css = (ui_dir() / "styles.css").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "is-settling" in css
    assert "settling" in js
    assert "切換中" in js


def test_heartbeat_runs_on_its_own_thread_independent_of_tick():
    from fluid_motion.core import watcher as watcher_mod

    src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    assert "_heartbeat_loop" in src
    assert "_heartbeat_thread" in src
    assert "fluid-heartbeat" in src
    # tick() must not be the only writer, or a slow apply() call (mpv
    # rebuilding the VS/TensorRT pipeline) starves the heartbeat past the
    # Lua side's 4s alive window and F3 wrongly reports Fluid Motion as dead.
    assert src.count("self._write_heartbeat()") >= 2


def test_engine_compiling_indicator_wired_in_ui():
    css = (ui_dir() / "styles.css").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "is-compiling" in css
    assert "engine_compiling" in js
    assert "首次編譯" in js


def test_output_shortfall_flags_stuck_at_source():
    from fluid_motion.core.inject import output_shortfall

    assert output_shortfall(30, 60, paused=False, interpolating=True) is True
    assert output_shortfall(52, 60, paused=False, interpolating=True) is True
    assert output_shortfall(58, 60, paused=False, interpolating=True) is False
    assert output_shortfall(30, 60, paused=True, interpolating=True) is False
    assert output_shortfall(30, 60, paused=False, interpolating=False) is False
    assert output_shortfall(None, 60, paused=False, interpolating=True) is False


def test_live_fps_label_keeps_broadcast_rates():
    from fluid_motion.core.inject import live_fps_label

    assert live_fps_label(23.976) == "23.976"
    assert live_fps_label(60.02) == "60"
    assert live_fps_label(54.2) == "54.2"
    assert live_fps_label(None) == "—"


def test_fps_gate_marks_shortfall_in_ui():
    css = (ui_dir() / "styles.css").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "is-bad" in css
    assert "--color-danger" in css
    assert "fps_ok" in js
    assert "效能不足" in js


def test_scene_presets_wired_in_ui():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="scene-presets"' in html
    assert "SCENE_PRESETS" in js
    assert "data-scene-preset" in js
    assert "set_scene" in js


def test_engine_cache_panel_wired_in_ui():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="cache-clear"' in html
    assert 'id="cache-open"' in html
    assert "clear_engine_cache" in js
    assert "open_engine_cache" in js
    assert "renderCache" in js


def test_gpu_mode_badge_wired_in_ui():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="gpu-mode"' in html
    assert "gpu_safe_mode" in js


def test_no_force_accel_control_survives_in_the_ui():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="force-accel"' not in html
    assert "force_accel" not in js
    assert "set_force_accel" not in js


def test_realtime_ratio_measures_shortfall_and_refuses_nonsense():
    from fluid_motion.core.inject import realtime_ratio

    # Keeping up: position advances as fast as the clock.
    assert realtime_ratio(1.0, 1.0) == pytest.approx(1.0)
    # The 4K case this panel exists for.
    assert realtime_ratio(0.32, 1.0) == pytest.approx(0.32)
    # mpv's own speed setting must not read as a shortfall or a surplus.
    assert realtime_ratio(2.0, 1.0, speed=2.0) == pytest.approx(1.0)
    assert realtime_ratio(0.5, 1.0, speed=0.5) == pytest.approx(1.0)

    # No elapsed time, a stopped clock, or a backwards jump says nothing.
    assert realtime_ratio(1.0, 0.0) is None
    assert realtime_ratio(1.0, -1.0) is None
    assert realtime_ratio(-5.0, 1.0) is None
    assert realtime_ratio(1.0, 1.0, speed=0) is None
    # A forward jump is a seek, not 30x playback.
    assert realtime_ratio(30.0, 1.0) is None


def test_realtime_ratio_smooths_towards_new_samples():
    from fluid_motion.core.inject import realtime_ratio

    # A single 0.3s sample is coarse, so a new reading only moves the figure
    # part of the way -- but it must move, and in the right direction.
    moved = realtime_ratio(0.5, 1.0, previous=1.0, smoothing=0.5)
    assert moved == pytest.approx(0.75)
    assert realtime_ratio(1.0, 1.0, previous=0.5, smoothing=0.5) == pytest.approx(0.75)
    # With no history the first usable sample is taken as-is.
    assert realtime_ratio(0.4, 1.0, previous=None) == pytest.approx(0.4)


def test_realtime_label_never_claims_headroom():
    from fluid_motion.core.inject import realtime_label

    # Above 1.0 is unmeasurable -- mpv presents at the target rate and no
    # faster -- so a fast pipeline must not be reported as if it had surplus.
    assert realtime_label(1.8) == "1.00×"
    assert realtime_label(1.0) == "1.00×"
    assert realtime_label(0.324) == "0.32×"
    assert realtime_label(None) == "—"


def test_realtime_panel_took_that_slot():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert 'id="realtime-stat"' in html
    assert 'id="realtime-bar"' in html
    assert "function renderRealtime(" in js
    assert "renderRealtime(" in js.split("function renderRealtime(", 1)[1]  # and it is called


def test_rife_425_lite_is_not_offered_in_ui():
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "4251" not in js


def test_toast_hidden_when_no_player_is_connected():
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "state.error && connected > 0" in js


def test_idle_toggle_does_not_claim_interpolating():
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "待命" in js
    assert 'enabled ? "補幀中" : "未啟用"' not in js


def test_idle_fps_gate_is_empty_without_a_player():
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "connected === 0" in js
    assert "等待 mpv" in js


def _tick_engine(monkeypatch, settings, ipc, pid=4321):
    """An Engine wired to exactly one fake mpv, with tick() safe to call."""
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_detect import PlayerProcess

    engine = watcher_mod.Engine(settings)
    monkeypatch.setattr(
        watcher_mod, "iter_mpv_processes", lambda: [PlayerProcess(pid=pid, name="mpv.exe")]
    )
    monkeypatch.setattr(watcher_mod, "connect_pid", lambda p, extra=None: ipc)
    monkeypatch.setattr(watcher_mod, "diagnose", lambda root: engine._runtime)
    monkeypatch.setattr(watcher_mod, "engine_cache_info", lambda: engine._engine_cache)
    engine._runtime.ready = True
    return engine


def _applied_profiles(monkeypatch, engine):
    """Record the profile each apply() actually wrote, without touching mpv."""
    from fluid_motion.core import watcher as watcher_mod

    seen: list[str] = []

    def fake_apply(ipc, settings, mpv_root, *, announce=False):
        seen.append(settings.profile)
        ipc.command("vf", "add", "@fluid:vapoursynth")
        return Path(mpv_root) / "shaders" / "fluid_rife.vpy"

    monkeypatch.setattr(watcher_mod, "apply", fake_apply)
    return seen


def test_settings_change_skipped_by_seek_hold_is_reapplied_by_the_next_tick(monkeypatch, tmp_path):
    """A profile change during a seek hold must survive the hold.

    set_enabled() `continue`s past apply() while settings.profile has already
    moved on, so the change reaches the config file but never reaches mpv. It
    is the applied-settings snapshot -- not player.interpolation -- that has to
    carry the "still owed an apply" state across the hold.
    """
    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="2x", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    seen = _applied_profiles(monkeypatch, engine)

    engine.tick()
    assert seen == ["2x"]

    # The user picks 120 while the post-seek hold is still active.
    monkeypatch.setattr(engine, "_held_off", lambda seeking=False: True)
    engine.update_settings(profile="120")
    assert seen == ["2x"], "an apply during the hold-off is skipped, by design"

    # The hold expires. The old filter is still loaded, so the pre-fix tick()
    # saw interpolation=True and did nothing at all -- forever.
    monkeypatch.setattr(engine, "_held_off", lambda seeking=False: False)
    engine.tick()
    assert seen == ["2x", "120"], "the skipped change must be picked back up"


def test_new_file_with_a_different_source_fps_reapplies_the_right_multiplier(monkeypatch, tmp_path):
    """The case that stayed broken forever: filter loaded, settings underneath stale.

    mpv keeps the vf across file loads, so player.interpolation stays True when
    the next file starts -- but the multiplier baked into the .vpy was computed
    for the previous file's fps. Keying only off "is a filter loaded" meant the
    new file kept playing through the old file's multiplier with nothing left
    to notice, which is the same shape of desync a skipped apply leaves behind.
    """
    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="120", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    seen = _applied_profiles(monkeypatch, engine)

    engine.tick()
    assert seen == ["120"]
    assert engine._applied[4321][-1] == 5, "120 / 23.976 -> 5x"

    # Next file in the playlist is 59.94, and the fluid vf is still loaded.
    ipc.props["container-fps"] = 59.94
    ipc.props["estimated-vfps"] = 59.94
    assert ipc.vf, "mpv keeps the filter across the file change"

    engine.tick()
    assert seen == ["120", "120"], "the multiplier is stale and must be rebuilt"
    assert engine._applied[4321][-1] == 2, "120 / 59.94 -> 2x"


def test_settled_settings_are_not_reapplied_every_tick(monkeypatch, tmp_path):
    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="120", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    seen = _applied_profiles(monkeypatch, engine)

    for _ in range(5):
        engine.tick()
    assert seen == ["120"], "a filter that already matches must be left alone"


def test_no_op_multiplier_does_not_reapply_forever(monkeypatch, tmp_path):
    """multi <= 1 is a legitimately applied no-op, not a missing filter.

    apply() removes the filter when there is nothing to interpolate, which
    leaves player.interpolation False -- the old `if not player.interpolation`
    test read that as "needs applying" and looped apply/remove every 0.3s.
    """
    from fluid_motion.core import watcher as watcher_mod

    ipc = _FakeIpc({"container-fps": 120, "estimated-vfps": 120})
    settings = Settings(enabled=True, profile="120", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    calls: list[str] = []
    monkeypatch.setattr(
        watcher_mod,
        "apply",
        lambda ipc_, s_, root_, announce=False: calls.append(s_.profile),
    )

    for _ in range(5):
        engine.tick()
    assert len(calls) == 1, f"source already at 120 -- applied once, then left alone: {calls}"


def test_rapid_setting_changes_converge_on_the_last_one(monkeypatch, tmp_path):
    """Clicking through several profiles quickly must end on the last click."""
    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="2x", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)
    seen = _applied_profiles(monkeypatch, engine)

    # Every one of these fails at the IPC layer except the last.
    monkeypatch.setattr(engine, "_held_off", lambda seeking=False: True)
    for profile in ("3x", "60", "144"):
        engine.update_settings(profile=profile)
    assert seen == [], "all four applies were skipped"

    monkeypatch.setattr(engine, "_held_off", lambda seeking=False: False)
    engine.tick()
    assert seen == ["144"], "the last selection wins, and it is actually applied"
    assert engine.settings.profile == "144"


def test_failed_apply_is_retried_after_a_backoff(monkeypatch, tmp_path):
    """An IpcError set self._error and was then forgotten by everyone."""
    import time as time_mod

    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_ipc import IpcError

    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="120", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)

    attempts: list[int] = []

    def failing(ipc_, s_, root_, announce=False):
        attempts.append(1)
        raise IpcError("mpv said no")

    monkeypatch.setattr(watcher_mod, "apply", failing)
    engine.tick()

    assert attempts == [1]
    assert engine._error == "mpv said no"
    assert 4321 not in engine._applied, "a failed apply must not look settled"

    # Backed off: the 0.3s loop must not hammer an mpv that is refusing commands.
    engine.tick()
    assert attempts == [1], "retry is throttled, not immediate"

    # Once the backoff expires it does retry -- and succeeds.
    engine._retry_at[4321] = time_mod.monotonic() - 1
    seen = _applied_profiles(monkeypatch, engine)
    engine.tick()
    assert seen == ["120"]
    assert engine._error == ""


def test_pipe_closing_apply_is_not_surfaced_as_error(monkeypatch, tmp_path):
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_ipc import IpcError

    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="2x", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)

    def dying(ipc_, s_, root_, announce=False):
        raise IpcError("無法加入補幀濾鏡：(232, 'WriteFile', '管道正關閉中。')")

    monkeypatch.setattr(watcher_mod, "apply", dying)
    engine.tick()
    assert engine._error == ""
    assert engine.state()["error"] == ""
    assert 4321 not in engine._retry_at, "a dead pipe must not be retried"


def test_error_clears_when_last_player_leaves(monkeypatch, tmp_path):
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_ipc import IpcError

    ipc = _FakeIpc({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(enabled=True, profile="2x", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)

    def failing(ipc_, s_, root_, announce=False):
        raise IpcError("mpv said no")

    monkeypatch.setattr(watcher_mod, "apply", failing)
    engine.tick()
    assert engine._error == "mpv said no"

    monkeypatch.setattr(watcher_mod, "iter_mpv_processes", lambda: [])
    engine.tick()
    assert engine._error == ""
    assert engine.state()["error"] == ""


def test_filter_key_changes_with_every_setting_that_rewrites_the_vpy():
    from fluid_motion.core.gpu import GpuSnapshot
    from fluid_motion.core.watcher import Engine

    engine = Engine(Settings(profile="2x", rife_model=426, trt_streams=1))
    engine._gpu = GpuSnapshot(name="NVIDIA GeForce RTX 4090", available=True)
    base = engine._filter_key(2)
    for field, value in (
        ("profile", "3x"),
        ("rife_model", 425),
        ("scene_threshold", 0.25),
        ("trt_streams", 4),
        ("fp16", False),
        ("cuda_graph", True),
        ("force_accel", True),
    ):
        engine = Engine(Settings(profile="2x", rife_model=426, trt_streams=1))
        engine._gpu = GpuSnapshot(name="NVIDIA GeForce RTX 4090", available=True)
        setattr(engine.settings, field, value)
        assert engine._filter_key(2) != base, f"{field} must invalidate the applied snapshot"
    # A new file with a different source fps resolves to a different multi.
    engine = Engine(Settings(profile="2x", rife_model=426, trt_streams=1))
    engine._gpu = GpuSnapshot(name="NVIDIA GeForce RTX 4090", available=True)
    assert engine._filter_key(3) != base


def test_filter_key_ignores_streams_clamped_away_by_safe_mode():
    from fluid_motion.core.gpu import GpuSnapshot
    from fluid_motion.core.watcher import Engine

    engine = Engine(Settings(profile="2x", rife_model=426, trt_streams=1, force_accel=False))
    engine._gpu = GpuSnapshot(name="NVIDIA GeForce RTX 5070 Ti", available=True)
    base = engine._filter_key(2)
    engine.settings.trt_streams = 4
    engine.settings.cuda_graph = True
    assert engine._filter_key(2) == base, "safe mode already forces 1 stream / no graph"


def test_remove_disconnect_is_not_surfaced_as_error(monkeypatch, tmp_path):
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_ipc import IpcError

    ipc = _FakeIpc({"container-fps": 23.976})
    settings = Settings(enabled=False, profile="2x", mpv_root=str(tmp_path))
    engine = _tick_engine(monkeypatch, settings, ipc)

    def dying(ipc_, announce=False):
        raise IpcError("無法加入補幀濾鏡：(232, 'WriteFile', '管道正關閉中。')")

    monkeypatch.setattr(watcher_mod, "remove", dying)
    engine._applied[4321] = engine._filter_key(2)
    ok = engine._remove_from(4321, ipc)
    assert ok is False
    assert engine._error == ""


def test_tick_reapplies_on_stale_settings_not_only_on_a_missing_filter():
    from fluid_motion.core import watcher as watcher_mod

    src = Path(watcher_mod.__file__).read_text(encoding="utf-8")
    # The bug: `if not player.interpolation: apply(...)` -- a filter that is
    # loaded but built from superseded settings looked healthy forever.
    assert "stale = applied != self._filter_key(want_multi)" in src
    assert "missing = want_multi > 1 and not player.interpolation" in src
    assert "if (stale or missing)" in src


def test_chip_groups_are_not_rebuilt_on_every_poll():
    """The poll used to destroy the chip under the cursor mid-click.

    render() runs every 900ms. Replacing #profiles.innerHTML between mousedown
    and mouseup detaches the button, so the browser dispatches click on the
    container, closest("[data-profile]") returns null, and the profile change
    is silently dropped -- which is exactly how mpv ended up running a filter
    that did not match the UI.
    """
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    for group in ("PROFILES.map", "MODELS.map", "SCENE_PRESETS.map"):
        assert f"innerHTML = {group}" not in js, f"{group} must not be re-rendered into innerHTML"
    assert 'root.dataset.built === "true"' in js, "chip groups must be built once"
    assert "function markPressed" in js, "state changes must only flip aria-pressed"


def test_in_flight_poll_cannot_repaint_stale_settings_over_a_command():
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")
    assert "let commandEpoch = 0;" in js
    assert "if (epoch !== commandEpoch) return;" in js
    # Every settings mutation goes through command(), which bumps the epoch.
    for name in ("set_profile", "set_model", "set_scene", "set_enabled"):
        assert f'command("{name}"' in js, f"{name} must go through command()"


def test_build_bat_does_not_report_success_after_a_failed_build():
    bat = Path("build.bat").read_text(encoding="utf-8")
    built = bat.index("echo Built:")
    guard = bat.index("if not exist dist\\FluidMotion.exe goto :failed")
    assert guard < built, "the success message must be gated on the exe existing"
    assert "BUILD FAILED" in bat
    assert "exit /b 1" in bat


def test_named_pipe_reads_peek_before_blocking():
    """ReadFile on a synchronous pipe blocks until bytes arrive.

    That made command()'s deadline decorative on Windows -- and named pipes are
    the only transport actually used there. An mpv that stopped servicing its
    IPC (compiling a TensorRT engine, wedged vo) parked the caller in ReadFile
    forever, holding the apply lock and freezing the UI behind it.
    """
    from fluid_motion.core import mpv_ipc as ipc_mod

    src = Path(ipc_mod.__file__).read_text(encoding="utf-8")
    read_body = src[src.index("def _read(self"):]
    peek = read_body.index("PeekNamedPipe")
    readfile = read_body.index("win32file.ReadFile")
    assert peek < readfile, "must peek for available bytes before calling ReadFile"
    assert "if not avail:" in read_body, "no bytes available must return instead of blocking"


def test_command_gives_up_at_the_deadline():
    from fluid_motion.core.mpv_ipc import IpcError, MpvIpc
    import time as time_mod

    ipc = MpvIpc(handle=object(), kind="unix", path="x")
    ipc._write = lambda data: None
    ipc._read = lambda n: b""  # mpv never answers

    started = time_mod.monotonic()
    try:
        ipc.command("get_property", "pause", timeout=0.3)
    except IpcError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("a silent mpv must raise, not hang")
    elapsed = time_mod.monotonic() - started
    assert 0.25 < elapsed < 1.5, f"deadline not honoured: {elapsed:.2f}s"


def test_no_bytes_are_discarded_between_commands():
    """command() used to keep its read buffer in a local.

    Anything read past the matching reply -- mpv pushes events down the same
    connection continuously -- was thrown away on return, so the next command
    started mid-line. The stream does resynchronise (the fragment fails to
    parse and is skipped), so this is robustness rather than a live bug, but
    carrying the tail over costs one attribute and removes the whole class.
    """
    from fluid_motion.core.mpv_ipc import MpvIpc

    ipc = MpvIpc(handle=object(), kind="unix", path="x")
    ipc._write = lambda data: None
    chunks = [b'{"data":"a","request_id":1}\n{"data":"b","request_id":2}\n']
    ipc._read = lambda n: chunks.pop(0) if chunks else b""

    assert ipc.command("get_property", "x", timeout=1.0) == "a"
    # Already buffered from the first read; a local buffer would have dropped it
    # and this would sit until the deadline.
    assert ipc.command("get_property", "y", timeout=1.0) == "b"


def test_stale_reply_from_a_timed_out_command_is_not_mistaken_for_the_next_one():
    from fluid_motion.core.mpv_ipc import MpvIpc

    ipc = MpvIpc(handle=object(), kind="unix", path="x")
    ipc._write = lambda data: None
    ipc._req = 7
    # The late answer to request 6 is sitting in the pipe ahead of request 7's.
    chunks = [b'{"data":"stale","request_id":6}\n{"data":"fresh","request_id":7}\n']
    ipc._read = lambda n: chunks.pop(0) if chunks else b""
    assert ipc.command("get_property", "x", timeout=1.0) == "fresh"


def test_multiplier_is_capped():
    from fractions import Fraction

    from fluid_motion.core.vs_script import MAX_MULTI, target_multi

    # Every distinct multi compiles its own TensorRT engine, and RIFE cannot
    # keep up this far past real time anyway.
    assert target_multi("144", Fraction(12), None) == (MAX_MULTI, False)
    assert target_multi("display", Fraction(24000, 1001), 240.0) == (MAX_MULTI, False)
    # Ordinary targets are untouched.
    assert target_multi("144", Fraction(24000, 1001), None) == (6, False)
    assert target_multi("120", Fraction(24000, 1001), None) == (5, False)


def test_apply_fails_loudly_when_the_filter_does_not_land(tmp_path, monkeypatch):
    """apply() used to accept any vapoursynth filter mentioning rife/fluid.

    A leftover from SVP or an older build answered yes, so a vf add that never
    took effect was reported as success -- and the applied-settings snapshot
    would then record a filter that is not actually running.
    """
    from fluid_motion.core.inject import apply
    from fluid_motion.core.mpv_ipc import IpcError

    monkeypatch.setenv("APPDATA", str(tmp_path))

    class SilentlyIgnoresAdd(_FakeIpc):
        def command(self, *args, **kwargs):
            self.commands.append(args)  # accepts "vf add" but never loads it

    ipc = SilentlyIgnoresAdd({"container-fps": 23.976, "estimated-vfps": 23.976})
    settings = Settings(profile="120", mpv_root=str(tmp_path))
    try:
        apply(ipc, settings, tmp_path)
    except IpcError as exc:
        assert "沒有掛上" in str(exc)
    else:
        raise AssertionError("apply() must not report success when nothing loaded")


def test_apply_does_not_accept_someone_elses_vapoursynth_filter(tmp_path, monkeypatch):
    from fluid_motion.core.inject import fluid_filter_loaded

    monkeypatch.setenv("APPDATA", str(tmp_path))
    ipc = _FakeIpc({})
    ipc.vf = [{"name": "vapoursynth", "label": "svp_rife"}]
    assert not fluid_filter_loaded(ipc), "another tool's rife filter is not ours"
    ipc.vf = [{"name": "vapoursynth", "label": "fluid"}]
    assert fluid_filter_loaded(ipc)


def test_process_scan_does_not_fetch_exe_and_cmdline_for_every_process():
    """~20ms per call across 600 processes, 3.3 times a second, forever.

    process_iter materialises the requested fields for every process on the
    box, and on Windows exe/cmdline each cost a handle open plus a PEB read.
    Only processes already matched by name need them.
    """
    from fluid_motion.core import mpv_detect

    src = Path(mpv_detect.__file__).read_text(encoding="utf-8")
    body = src[src.index("def iter_mpv_processes"):]
    start = body.index("psutil.process_iter(")
    scan = body[start:body.index(")", start) + 1]
    assert "exe" not in scan and "cmdline" not in scan, f"still eagerly fetched: {scan}"


def test_settings_change_writes_the_config_once(tmp_path, monkeypatch):
    import fluid_motion.core.watcher as watcher_mod

    engine = watcher_mod.Engine(Settings(enabled=True, mpv_root=str(tmp_path)))
    writes = []
    monkeypatch.setattr(watcher_mod, "save_settings", lambda s: writes.append(s.profile))
    monkeypatch.setattr(engine, "tick", lambda **kw: None)
    engine.update_settings(profile="120")
    assert writes == ["120"], f"config.json written {len(writes)}x for one change"


def test_hotkey_dispatch_does_not_re_enter_itself(tmp_path, monkeypatch):
    """set_enabled() ends with tick(), which calls back into _consume_hotkey."""
    import fluid_motion.core.watcher as watcher_mod

    hotkey = tmp_path / "hotkey"
    hotkey.write_text("toggle", encoding="utf-8")
    monkeypatch.setattr(watcher_mod, "hotkey_path", lambda: hotkey)

    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path)))
    depth = {"now": 0, "max": 0}
    real = engine._consume_hotkey

    def counting():
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        try:
            real()
        finally:
            depth["now"] -= 1

    def set_enabled(enabled):
        # Stands in for set_enabled()'s trailing tick(), and races a fresh
        # hotkey landing in the same window.
        hotkey.write_text("toggle", encoding="utf-8")
        engine._consume_hotkey()

    monkeypatch.setattr(engine, "_consume_hotkey", counting)
    monkeypatch.setattr(engine, "set_enabled", set_enabled)
    try:
        engine._consume_hotkey()
    except RecursionError:
        raise AssertionError("hotkey dispatch recursed without bound") from None
    assert depth["max"] <= 2, f"hotkey dispatch nested {depth['max']} deep"
