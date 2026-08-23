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
    assert loaded.rife_model == 46


def test_invalid_profile_falls_back():
    settings = Settings.from_dict({"profile": "nope"})
    assert settings.profile == "2x"


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


def test_vf_arg_uses_label():
    arg = _vf_arg(Path("C:/mpv/shaders/fluid_rife.vpy"))
    assert arg.startswith(FILTER_LABEL)
    assert "vapoursynth=" in arg
    assert "fluid_rife.vpy" in arg


def test_candidate_pipes_include_fluid_and_mpvpipe():
    pipes = candidate_pipes(4242)
    assert any("fluid-mpv-4242" in p for p in pipes)
    assert any(p.endswith("mpvpipe") or p.endswith("\\mpvpipe") for p in pipes)


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


def test_titlebar_has_pywebview_drag_region():
    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    assert "pywebview-drag-region" in html
    assert "window-btns" in html
