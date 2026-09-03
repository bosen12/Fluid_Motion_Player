"""What "ready" has to mean once there is more than one backend.

Nobody working on this repo has AMD hardware, so the AMD path cannot be proved
to interpolate. What it *can* be held to is the property that matters when it
is wrong: a tree without the ncnn plugin must report itself not ready, rather
than reporting ready and leaving mpv to say only "could not init VS".
"""
from __future__ import annotations

from pathlib import Path

from fluid_motion.core.runtime import diagnose, missing_labels


def _tree(root: Path, *, accel: str | None) -> Path:
    """Everything both backends need, plus whichever accelerator is asked for."""
    (root / "mpv.exe").write_bytes(b"mz")
    (root / "vapoursynth.dll").write_bytes(b"dll")
    (root / "python.exe").write_bytes(b"py")
    (root / "python312.dll").write_bytes(b"dll")
    (root / "vsmlrt.py").write_text("x", encoding="utf-8")
    plugins = root / "vs-plugins"
    plugins.mkdir(exist_ok=True)
    (plugins / "MiscFilters.dll").write_bytes(b"dll")
    models = plugins / "models" / "rife"
    models.mkdir(parents=True, exist_ok=True)
    (models / "rife_v4.6.onnx").write_bytes(b"onnx")
    if accel == "trt":
        (plugins / "vstrt.dll").write_bytes(b"dll")
        (plugins / "nvinfer.dll").write_bytes(b"dll")
    elif accel == "ncnn":
        (plugins / "vsncnn.dll").write_bytes(b"dll")
    return root


def test_an_amd_tree_without_the_ncnn_plugin_is_not_ready(tmp_path):
    """The whole safety argument for shipping an untested backend rests here."""
    status = diagnose(_tree(tmp_path, accel=None), backend="ncnn")

    assert status.ready is False
    ids = {c.id: c.ok for c in status.checks}
    assert ids["ncnn"] is False
    assert "ncnn / Vulkan" in missing_labels(status), "the panel has to name what is missing"


def test_an_amd_tree_with_the_ncnn_plugin_is_ready(tmp_path):
    status = diagnose(_tree(tmp_path, accel="ncnn"), backend="ncnn")

    assert status.ready is True
    assert {c.id for c in status.checks} >= {"ncnn"}
    assert "tensorrt" not in {c.id for c in status.checks}, (
        "an AMD tree was judged against a CUDA-only requirement"
    )


def test_a_tensorrt_tree_is_not_accepted_as_an_amd_one(tmp_path):
    """The failure that would be silent: vstrt.dll is present, so a check that
    did not follow the backend would pass, and the generated script would then
    ask for Backend.NCNN_VK with nothing to load it."""
    status = diagnose(_tree(tmp_path, accel="trt"), backend="ncnn")

    assert status.ready is False


def test_an_amd_tree_is_not_accepted_as_a_tensorrt_one(tmp_path):
    status = diagnose(_tree(tmp_path, accel="ncnn"), backend="trt")

    assert status.ready is False
    assert "TensorRT + CUDA" in missing_labels(status)


def test_the_nvidia_verdict_is_exactly_what_it_was(tmp_path):
    """Unchanged behaviour is the point: this is the arrangement that has always
    reported ready, and adding a second backend must not have moved it."""
    status = diagnose(_tree(tmp_path, accel="trt"), backend="trt")

    assert status.ready is True
    ids = {c.id for c in status.checks}
    assert "tensorrt" in ids
    assert "ncnn" not in ids


# -- what the installer actually fetches -----------------------------------
def _record_downloads(monkeypatch, tmp_path):
    """Every URL install_runtime asks for, with nothing actually fetched."""
    from fluid_motion.core import bootstrap

    import types

    urls: list[str] = []
    monkeypatch.setattr(bootstrap, "install_vapoursynth", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_sevenzr", lambda cache: tmp_path / "7zr.exe")
    monkeypatch.setattr(bootstrap, "_extract", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "download_dir", lambda: tmp_path / "cache")
    # The models step shells out to 7z to list the archive before extracting a
    # single member from it; nothing is on disk here for it to read.
    monkeypatch.setattr(
        bootstrap,
        "run_hidden",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    (tmp_path / "cache").mkdir(exist_ok=True)

    def fake_download(url, dest, cb, label, span):
        urls.append(url)

    monkeypatch.setattr(bootstrap, "_download", fake_download)
    return urls


def test_an_amd_install_does_not_pull_the_cuda_runtime(monkeypatch, tmp_path):
    """2.6 GB of TensorRT and CUDA cannot load on an AMD GPU, so fetching it
    would cost the whole download and still leave the tree unusable. The ncnn
    plugin is 2.7 MB and is the only accelerator this path needs -- Vulkan
    itself comes from the display driver."""
    from fluid_motion.core.bootstrap import install_runtime

    urls = _record_downloads(monkeypatch, tmp_path)
    install_runtime(tmp_path / "mpv", backend="ncnn")

    joined = " ".join(urls)
    assert "VSNCNN-Windows-x64" in joined, "the ncnn plugin was never fetched"
    assert "tensorrt" not in joined.lower(), "an AMD install downloaded the CUDA runtime"
    assert "vsmlrt-cuda" not in joined
    # Still needs the parts both backends share.
    assert "models." in joined, "the RIFE models are not backend-specific"


def test_an_nvidia_install_still_pulls_tensorrt(monkeypatch, tmp_path):
    from fluid_motion.core.bootstrap import install_runtime

    urls = _record_downloads(monkeypatch, tmp_path)
    install_runtime(tmp_path / "mpv", backend="trt")

    joined = " ".join(urls)
    assert "vsmlrt-windows-x64-tensorrt" in joined
    assert ".7z.001" in joined and ".7z.002" in joined, "both parts are required"
    assert "VSNCNN" not in joined


# -- the UI stops claiming CUDA on a machine that has none ------------------
def test_the_engine_line_names_the_backend_actually_in_use():
    """It read "TensorRT · CUDA" from a literal, so an AMD machine would have
    been told it was running CUDA. The label now follows state.backend."""
    from fluid_motion.paths import ui_dir

    js = (ui_dir() / "app.js").read_text(encoding="utf-8")

    assert "TensorRT · CUDA · 等待 mpv" not in js, "the hardcoded CUDA label is back"
    assert "function backendLabel" in js
    assert "backendLabel(state.backend)" in js


def test_the_backend_selector_is_wired_end_to_end():
    """Three separate places have to agree or the control silently does
    nothing: the container in the HTML, the render call, and the bridge
    method."""
    from fluid_motion.paths import ui_dir

    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = (ui_dir() / "app.js").read_text(encoding="utf-8")

    assert 'id="backends"' in html
    assert "renderBackends(state.settings.backend, state.backend, state.adapters)" in js
    assert 'command("set_backend"' in js
    assert hasattr(__import__("fluid_motion.api", fromlist=["Bridge"]).Bridge, "set_backend")
