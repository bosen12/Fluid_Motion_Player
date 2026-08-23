from pathlib import Path

from fluid_motion.paths import ui_dir

from fluid_motion.config import Settings, load_settings, save_settings
from fluid_motion.core.inject import FILTER_LABEL, _vf_arg
from fluid_motion.core.mpv_detect import candidate_pipes
from fluid_motion.core.runtime import diagnose


def test_settings_roundtrip(tmp_path: Path):
    path = tmp_path / "config.json"
    settings = Settings(enabled=True, profile="120", trt_streams=3)
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.enabled is True
    assert loaded.profile == "120"
    assert loaded.trt_streams == 3
    assert loaded.rife_model == 426


def test_invalid_profile_falls_back():
    settings = Settings.from_dict({"profile": "nope"})
    assert settings.profile == "2x"


def test_invalid_rife_model_falls_back():
    settings = Settings.from_dict({"rife_model": 999})
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
    assert "apply(ipc, self.settings, Path(self.settings.mpv_root), announce=True)" in watcher_src
    assert "remove(ipc, announce=True)" in watcher_src
    assert watcher_src.count("announce=True") == 2


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
