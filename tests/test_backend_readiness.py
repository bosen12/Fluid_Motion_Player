"""What "ready" has to mean once there is more than one backend.

Nobody working on this repo has AMD hardware, so the AMD path cannot be proved
to interpolate. What it *can* be held to is the property that matters when it
is wrong: a tree without the ncnn plugin must report itself not ready, rather
than reporting ready and leaving mpv to say only "could not init VS".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fluid_motion.core.runtime import diagnose, missing_labels


def _code_only(js: str) -> str:
    """The file with whole-line // comments dropped.

    Only whole-line ones: a naive split on "//" would cut a URL in half. That
    is enough here, because what these assertions guard against is a string
    being *used*, and the comments that mention the old strings are all on
    their own lines.
    """
    return "\n".join(
        line for line in js.splitlines() if not line.lstrip().startswith("//")
    )


@pytest.fixture(autouse=True)
def vulkan_present(monkeypatch):
    """Vulkan is a property of the machine running the suite, not of tmp_path.

    Left alone, every ncnn assertion here would quietly depend on whether the
    box has a display driver installed -- green on a workstation, red on a CI
    runner, for reasons having nothing to do with the code. Pinned on, and the
    one test that cares turns it off explicitly.
    """
    from fluid_motion.core import runtime

    monkeypatch.setattr(runtime, "vulkan_available", lambda: True)


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


def test_the_ncnn_plugin_without_vulkan_is_not_ready(monkeypatch, tmp_path):
    """The plugin is installable; the Vulkan loader is not -- it comes from the
    display driver. Checking only the DLL would call this machine ready and let
    the filter construct and then fail, with mpv saying nothing more useful than
    "could not init VS"."""
    from fluid_motion.core import runtime

    monkeypatch.setattr(runtime, "vulkan_available", lambda: False)
    status = diagnose(_tree(tmp_path, accel="ncnn"), backend="ncnn")

    assert status.ready is False
    detail = next(c.detail for c in status.checks if c.id == "ncnn")
    assert "Vulkan" in detail, "the message has to name Vulkan, not the plugin"
    assert "vsncnn.dll" not in detail, "the plugin is present -- do not blame it"


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


# -- the seam between detection and the file mpv loads ----------------------
def test_apply_writes_the_backend_the_hardware_resolved_to(monkeypatch, tmp_path):
    """resolve_backend() being right is worth nothing if apply() does not carry
    the answer into RifeParams. Everything either side of this seam is tested;
    without this, a wiring mistake here would leave an AMD machine writing a
    TensorRT script and nothing would fail."""
    from fluid_motion.config import Settings
    from fluid_motion.core import gpu, inject

    captured = {}
    monkeypatch.setattr(
        inject, "write_vpy", lambda path, params, **kw: captured.update(backend=params.backend) or path
    )
    monkeypatch.setattr(inject, "available_vendors", lambda: {gpu.AMD})
    monkeypatch.setattr(inject, "resolve_multi", lambda info, settings: 1)  # stop before any IPC

    inject.apply(
        _FakeIpc(),
        Settings(backend="auto"),
        tmp_path,
        info={"container_fps": 24, "display_fps": 60},
    )

    assert captured["backend"] == "ncnn", "an AMD machine wrote a TensorRT script"


def test_apply_still_writes_tensorrt_for_an_nvidia_machine(monkeypatch, tmp_path):
    from fluid_motion.config import Settings
    from fluid_motion.core import gpu, inject

    captured = {}
    monkeypatch.setattr(
        inject, "write_vpy", lambda path, params, **kw: captured.update(backend=params.backend) or path
    )
    monkeypatch.setattr(inject, "available_vendors", lambda: {gpu.NVIDIA, gpu.AMD})
    monkeypatch.setattr(inject, "resolve_multi", lambda info, settings: 1)

    inject.apply(
        _FakeIpc(),
        Settings(backend="auto"),
        tmp_path,
        info={"container_fps": 24, "display_fps": 60},
    )

    assert captured["backend"] == "trt"


class _FakeIpc:
    """Enough of MpvIpc for apply() and for a full tick() over one player."""

    def __init__(self):
        self.commands = []
        self.path = "fake-pipe"
        self.vf: list = []
        self.props = {
            "hwdec": "auto-copy",
            "container-fps": 24,
            "estimated-vf-fps": 24,
            "display-fps": 60,
        }

    def get(self, name, *, timeout=None):
        if name == "vf":
            return self.vf
        return self.props.get(name)

    def set(self, name, value):
        self.props[name] = value

    def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[:2] == ("vf", "add"):
            self.vf = [{"name": "vapoursynth", "label": "fluid"}]
        elif args[:2] == ("vf", "remove"):
            self.vf = []

    def close(self):
        pass


# -- a backend change has to reach mpv --------------------------------------
def _engine_with_one_player(monkeypatch, vendors):
    """A connected, ready player, with the vendor set under the test's control."""
    from fluid_motion.config import Settings
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.mpv_detect import PlayerProcess
    from fluid_motion.core.runtime import Check, RuntimeStatus

    def _ready(root):
        ids = ("mpv", "vapoursynth", "scdetect", "vsmlrt", "tensorrt", "python")
        checks = [Check(n, n.title(), True) for n in ids]
        checks += [Check(f"rife{n}", f"RIFE {n}", True) for n in (46, 425, 426)]
        return RuntimeStatus(mpv_root=str(root), ready=True, checks=checks)

    engine = watcher_mod.Engine(Settings(enabled=True, profile="2x", mpv_root="Z:/cfg"))
    ipc = _FakeIpc()
    told: list[str | None] = []

    def fake_apply(ipc_, settings, mpv_root, *, announce=False, info=None, pid=None, backend=None):
        told.append(backend)
        ipc_.command("vf", "add", "@fluid:vapoursynth")
        return Path(mpv_root) / "fluid_rife.vpy"

    monkeypatch.setattr(
        watcher_mod, "iter_mpv_processes",
        lambda: [PlayerProcess(pid=99, name="mpv.exe", label="mpv")],
    )
    monkeypatch.setattr(watcher_mod, "connect_pid", lambda p, extra=None, **kw: ipc)
    monkeypatch.setattr(watcher_mod, "player_config_dir", lambda _i: Path("Z:/player"))
    monkeypatch.setattr(watcher_mod, "diagnose", lambda root, **_kw: _ready(root))
    monkeypatch.setattr(watcher_mod, "engine_cache_info", lambda: engine._engine_cache)
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: r)
    monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda r: None)
    monkeypatch.setattr(watcher_mod, "apply", fake_apply)
    monkeypatch.setattr(watcher_mod, "available_vendors", lambda **_kw: vendors[0])
    engine._runtime = _ready("Z:/cfg")
    return engine, told


def test_a_changed_backend_is_re_applied_rather_than_left_running(monkeypatch):
    """_filter_key is "everything that changes the generated .vpy", and the
    backend is the largest such difference there is -- it decides which
    Backend() the script constructs.

    Leaving it out looked harmless because a UI switch re-applies anyway
    (set_enabled invalidates first). A change arriving any other way -- the
    "auto" answer moving because the adapter set did -- was silently kept:
    mpv went on running the old backend's filter while the panel reported the
    new one, with nothing left to notice.
    """
    from fluid_motion.core import gpu

    vendors = [{gpu.NVIDIA}]
    engine, told = _engine_with_one_player(monkeypatch, vendors)

    engine.tick()
    assert told == ["trt"]

    engine.tick()
    assert told == ["trt"], "an unchanged backend must not re-apply every tick"

    vendors[0] = {gpu.AMD}
    engine.tick()
    assert told == ["trt", "ncnn"], "the filter kept running on the old backend"


def test_the_key_and_the_script_cannot_disagree_about_the_backend(monkeypatch):
    """_apply_to used to resolve the backend twice, either side of the apply
    lock, and that lock blocks for as long as another apply takes -- seconds,
    while mpv rebuilds its pipeline. A vendor set that moved in between (the
    nvidia-smi snapshot expires every 1.5s) recorded one backend in the key and
    wrote the other into the .vpy. Once the vendors settled back, the key
    matched again and nothing ever re-applied: mpv ran the wrong backend for
    the rest of the session.
    """
    from fluid_motion.config import Settings
    from fluid_motion.core import watcher as watcher_mod

    engine = watcher_mod.Engine(Settings(enabled=True, mpv_root="Z:/cfg"))
    # A backend that answers differently on each call is what the race looks
    # like from inside _apply_to.
    answers = iter(["trt", "ncnn", "ncnn", "ncnn"])
    monkeypatch.setattr(engine, "_backend", lambda: next(answers))

    told: dict[str, str | None] = {}
    monkeypatch.setattr(
        watcher_mod, "apply",
        lambda ipc, s, root, *, announce=False, info=None, pid=None, backend=None: (
            told.update(backend=backend) or Path("x.vpy")
        ),
    )
    monkeypatch.setattr(watcher_mod, "player_config_dir", lambda _i: Path("Z:/p"))

    engine._apply_to(99, _FakeIpc(), 2, info={"container_fps": 24, "display_fps": 60})

    assert engine._applied[99][-2] == told["backend"], (
        "the key describes a different backend from the script that was written"
    )


def test_apply_is_told_the_backend_rather_than_deriving_its_own(monkeypatch):
    """One decision, used everywhere. The watcher already resolves the backend
    for _filter_key and for the readiness verdict; apply() deriving a third
    answer independently would be equal to those only by coincidence, since
    available_vendors() reads caches with their own expiry."""
    from fluid_motion.core import gpu

    engine, told = _engine_with_one_player(monkeypatch, [{gpu.AMD}])

    engine.tick()

    assert told == ["ncnn"], "apply() was not handed the resolved backend"
    assert None not in told, "apply() was left to work it out for itself"


def test_ncnn_does_not_scan_a_tensorrt_engine_cache(monkeypatch):
    """ncnn compiles no engines, so there is nothing to size and nothing that
    can grow -- and the panel that shows it is hidden on that backend.

    Not free to leave in: a machine that switched over from TensorRT still has
    the old engines on disk, and that scan measured 6.4 ms, i.e. 382 ms/minute
    on the active housekeeping cadence.
    """
    from fluid_motion.core import gpu
    from fluid_motion.core import watcher as watcher_mod

    engine, _ = _engine_with_one_player(monkeypatch, [{gpu.AMD}])
    scans = []
    monkeypatch.setattr(
        watcher_mod, "engine_cache_info",
        lambda: scans.append(1) or engine._engine_cache,
    )
    engine._housekeeping_at = 0.0  # force the housekeeping half to run

    engine.tick()

    assert scans == [], "an ncnn machine walked the TensorRT engine cache"
    assert engine._engine_growing_until == 0.0, "ncnn cannot be compiling an engine"


def test_tensorrt_still_scans_its_engine_cache(monkeypatch):
    from fluid_motion.core import gpu
    from fluid_motion.core import watcher as watcher_mod

    engine, _ = _engine_with_one_player(monkeypatch, [{gpu.NVIDIA}])
    scans = []
    monkeypatch.setattr(
        watcher_mod, "engine_cache_info",
        lambda: scans.append(1) or engine._engine_cache,
    )
    engine._housekeeping_at = 0.0

    engine.tick()

    assert scans, "the engine cache reading the UI shows was never taken"


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


def test_no_runtime_message_tells_an_amd_user_to_install_tensorrt():
    """Three sentences named TensorRT unconditionally -- the readiness line, the
    setup button and the not-ready prompt. On an AMD machine each one named a
    runtime that is neither installed nor wanted, and one of them was an
    instruction to go and install it."""
    from fluid_motion.paths import ui_dir

    js = _code_only((ui_dir() / "app.js").read_text(encoding="utf-8"))

    assert "function backendName" in js
    assert '"安裝 TensorRT 執行環境"' not in js
    assert '"還缺 TensorRT 執行環境。安裝後請重新開啟 mpv。"' not in js
    assert "backendName(state.backend)" in js


def test_no_player_runtime_error_names_the_resolved_ncnn_backend(monkeypatch, tmp_path):
    from fluid_motion.config import Settings
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.vs_script import BACKEND_NCNN

    engine = watcher_mod.Engine(Settings(backend="amd", mpv_root=str(tmp_path)))
    engine._runtime.ready = False
    monkeypatch.setattr(engine, "_backend", lambda: BACKEND_NCNN)
    monkeypatch.setattr(watcher_mod, "iter_mpv_processes", lambda: [])

    engine.set_enabled(True)

    assert "ncnn / Vulkan" in engine._error
    assert "TensorRT" not in engine._error


def test_backend_neutral_app_shell_is_updated_from_resolved_state():
    from fluid_motion.paths import project_root, ui_dir

    html = (ui_dir() / "index.html").read_text(encoding="utf-8")
    js = _code_only((ui_dir() / "app.js").read_text(encoding="utf-8"))
    app_py = (project_root() / "fluid_motion" / "app.py").read_text(encoding="utf-8")

    assert 'id="brand-backend"' in html
    assert "RIFE · TensorRT" not in html
    assert '$("brand-backend").textContent' in js
    assert "RIFE TensorRT 即時補幀" not in app_py


def test_the_gpu_panel_does_not_deny_a_gpu_it_can_name():
    """state.gpu comes from nvidia-smi, so an AMD machine had gpu.available
    false and was told "未偵測到 NVIDIA GPU" -- while state.adapters held its
    actual adapter name and the ncnn backend was set to run on it."""
    from fluid_motion.paths import ui_dir

    # Comments stripped first: the fix's own comment quotes the string it
    # replaced, and matching that would be the test failing for the wrong
    # reason -- the property is "this is not what gets shown", not "this text
    # appears nowhere in the file".
    js = _code_only((ui_dir() / "app.js").read_text(encoding="utf-8"))

    assert '"未偵測到 NVIDIA GPU"' not in js, "the NVIDIA-only message is back"
    assert "state.adapters || [])[0]" in js, "the detected adapter name is not used"
    # The safe/accelerated badge is the RTX 50 CUDA-graph gate; ncnn has no
    # CUDA graphs, so reporting on it there would describe a feature not in play.
    assert 'state.backend === "ncnn"' in js


def test_the_ui_poll_does_not_respawn_nvidia_smi(monkeypatch):
    """state() runs every 900ms for as long as the app is running -- app.js
    polls unconditionally, tray or not -- and it asks _backend() every time.

    _backend() calls available_vendors(), which called gpu.snapshot(), which
    shells out to nvidia-smi behind a 1.5s TTL. A 900ms caller misses that TTL
    every other call: measured at 46.7ms a spawn, 1.6 seconds of subprocess per
    minute, forever.

    watcher's HOUSEKEEPING_ACTIVE/IDLE constants exist to stop exactly this --
    their comment says "spawning nvidia-smi every 1.5s (its own cache TTL),
    forever" -- and _backend() reintroduced it one layer up, through a path
    those constants do not gate.

    The clock has to move. The first version of this test called _backend() in
    a tight loop and asserted on the spawn count: it passed with the cache
    removed, because sixty iterations take microseconds and snapshot()'s own
    TTL covered every one of them. It was measuring nothing. Advancing a fake
    clock by the real poll interval is what makes the TTL expire the way it
    does in the app.

    Counted at _run_smi rather than at available_vendors, because that is where
    the process is actually created; a cache that merely moved the call would
    still pass a check on the wrapper.
    """
    import types

    from fluid_motion.config import Settings
    from fluid_motion.core import gpu
    from fluid_motion.core import watcher as watcher_mod

    clock = [0.0]
    monkeypatch.setattr(gpu, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    spawns = []
    monkeypatch.setattr(gpu, "_run_smi", lambda args: spawns.append(args) or "")
    monkeypatch.setattr(gpu, "_CACHE", None)
    monkeypatch.setattr(gpu, "detect_adapters", lambda **_kw: ["NVIDIA GeForce RTX 5070 Ti"])

    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))
    baseline = len(spawns)

    for _ in range(60):
        clock[0] += 0.9  # the UI's poll interval
        engine._backend()

    assert len(spawns) == baseline, (
        f"{len(spawns) - baseline} nvidia-smi spawns across 54 seconds of "
        "polling. At 46.7ms each that is subprocess churn for the whole life "
        "of the app, on a path HOUSEKEEPING_IDLE cannot gate."
    )


def test_a_forced_refresh_still_reaches_the_hardware(monkeypatch):
    """The cache above must not make the answer permanent. available_vendors
    keeps its refresh flag precisely so a caller that knows something changed
    can get past it -- without this, the mutation that drops `not refresh`
    from the guard survives, and the flag becomes a lie the signature tells."""
    import types

    from fluid_motion.core import gpu

    clock = [0.0]
    monkeypatch.setattr(gpu, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(gpu, "_CACHE", None)
    # No nvidia-smi: this machine's real one answers, and it would add NVIDIA
    # to every set below regardless of what detect_adapters is told to say.
    monkeypatch.setattr(gpu, "_run_smi", lambda args: "")
    monkeypatch.setattr(gpu, "detect_adapters", lambda **_kw: ["AMD Radeon"])

    assert gpu.available_vendors() == {gpu.AMD}

    monkeypatch.setattr(gpu, "detect_adapters", lambda **_kw: ["NVIDIA GeForce RTX 5070 Ti"])
    assert gpu.available_vendors() == {gpu.AMD}, "the cache is not holding"
    assert gpu.available_vendors(refresh=True) == {gpu.NVIDIA}, (
        "refresh=True did not reach past the cache"
    )
