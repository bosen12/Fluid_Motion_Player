"""TensorRT timing-cache seeding in the generated script.

The block runs inside mpv's VapourSynth, where nothing here can reach it, so
the tests cut it out of a real render_vpy() result and run it against a fake
vsmlrt. What ships is what is tested: the text is taken from the rendered
script, not from the constant it is built from.

The fake is a real .py file imported into sys.modules, and every "run" execs
the script into a fresh namespace that is cleared afterwards -- which is what
mpv does on every seek, in one long-lived interpreter. 1.6.13 broke exactly
there (HANDOFF §9.63): its tests ran the script once per process, the one
shape that cannot show a wrapper outliving its run.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pytest

from fluid_motion.core.vs_script import BACKEND_NCNN, RifeParams, render_vpy

START = "from vsmlrt import RIFE, Backend\n"
END = "\n# RIFE is temporal."

FAKE_VSMLRT = '''
TARGET = None
FAIL_NEXT = 0
CALLS = []


def get_engine_path(target):
    # Pure in its arguments, like the real one -- which is what makes taking
    # it from a fresh copy of the file a faithful repair.
    return target


def RIFE(clip, multi=2, backend=None):
    global FAIL_NEXT
    CALLS.append(backend)
    # Resolved through the module global, as vsmlrt.trtexec() does.
    get_engine_path(TARGET)
    backend.force_fp16 = False
    backend.custom_args.append("--precisionConstraints=obey")
    if FAIL_NEXT:
        FAIL_NEXT -= 1
        raise RuntimeError("trtexec execution fails")
    return "interpolated"
'''


def _render(engine_folder, **kw):
    return render_vpy(RifeParams(model=426, mpv_root="C:/mpv", engine_folder=engine_folder,
                                 gpu_name="NVIDIA GeForce RTX 5070 Ti", **kw))


def _seed_block(text):
    return text[text.index(START) + len(START):text.index(END)]


class FakeBackend:
    def __init__(self):
        self.force_fp16 = True
        self.custom_args = []


class Uncopyable:
    """Stands in for a VideoNode, which refuses to be (deep)copied."""

    def __deepcopy__(self, memo):
        raise TypeError("cannot be converted to a Python object for pickling")


@pytest.fixture
def mpv(tmp_path, monkeypatch):
    """One long-lived 'mpv' interpreter: vsmlrt imported once, script run many times."""
    engines = tmp_path / "engines"
    engines.mkdir()
    src = tmp_path / "vsmlrt.py"
    src.write_text(FAKE_VSMLRT, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("vsmlrt", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vsmlrt", module)
    module.TARGET = str(engines / "new.engine")
    genuine = module.get_engine_path
    block = _seed_block(_render(str(engines)))

    def run(backend=None, clip=None):
        """One evaluation of the generated script, then mpv's teardown."""
        ns = {}
        exec(compile("from vsmlrt import RIFE\n" + block, "fluid_rife.vpy", "exec"), ns)
        try:
            return ns["RIFE"](clip if clip is not None else Uncopyable(), multi=2,
                              backend=backend or FakeBackend())
        finally:
            ns.clear()

    return types.SimpleNamespace(run=run, vsmlrt=module, engines=engines, genuine=genuine, src=src)


def _cache(path: Path, data: bytes, age: float) -> Path:
    path.write_bytes(data)
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def _is_genuine(mpv):
    return mpv.vsmlrt.get_engine_path.__code__.co_filename == str(mpv.src)


def test_new_engine_is_seeded_from_the_newest_cache(mpv):
    _cache(mpv.engines / "a.engine.cache", b"old", age=100)
    _cache(mpv.engines / "b.engine.cache", b"newest", age=10)
    _cache(mpv.engines / "c.engine.cache", b"", age=1)  # a failed build's leftover
    assert mpv.run() == "interpolated"
    assert (mpv.engines / "new.engine.cache").read_bytes() == b"newest"


def test_runs_again_and_again_in_one_interpreter(mpv):
    # The 1.6.13 failure: the second run (a seek) died with NameError.
    for _ in range(5):
        assert mpv.run() == "interpolated"
        assert _is_genuine(mpv), "the wrapper outlived its run"
    assert len(mpv.vsmlrt.CALLS) == 5


def test_module_is_restored_even_when_rife_fails(mpv):
    mpv.vsmlrt.FAIL_NEXT = 1
    with pytest.raises(RuntimeError):
        mpv.run()
    assert _is_genuine(mpv)
    assert mpv.run() == "interpolated"


def test_player_poisoned_by_1_6_13_is_repaired(mpv):
    # What 1.6.13 left behind: a wrapper on the module whose globals are gone.
    dead = {}
    exec(compile("def get_engine_path(*a, **k):\n    return _fm_get_engine_path(*a, **k)\n",
                 "C:/mpv/shaders/fluid_rife.vpy", "exec"), dead)
    mpv.vsmlrt.get_engine_path = dead["get_engine_path"]
    dead.clear()
    with pytest.raises(NameError):
        mpv.vsmlrt.get_engine_path()

    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    assert mpv.run() == "interpolated"
    assert mpv.vsmlrt.get_engine_path("x") == "x"
    assert (mpv.engines / "new.engine.cache").read_bytes() == b"seed"
    assert mpv.run() == "interpolated"


def test_built_engine_is_left_alone(mpv):
    # mpv re-runs the script on every seek; with the engine there, nothing moves.
    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    (mpv.engines / "new.engine").write_bytes(b"x" * 2048)
    mpv.run()
    assert not (mpv.engines / "new.engine.cache").exists()


def test_failed_tiny_engine_still_gets_a_seed(mpv):
    # vsmlrt rebuilds an engine under 1024 bytes (a 0-byte one is a real
    # leftover in the owner's cache), so that rebuild is worth seeding too.
    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    (mpv.engines / "new.engine").write_bytes(b"")
    mpv.run()
    assert (mpv.engines / "new.engine.cache").read_bytes() == b"seed"


def test_own_cache_is_never_overwritten(mpv):
    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    _cache(mpv.engines / "new.engine.cache", b"mine", age=50)
    mpv.run()
    assert (mpv.engines / "new.engine.cache").read_bytes() == b"mine"


def test_engine_outside_the_folder_is_not_seeded(mpv, tmp_path):
    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # A cache there too, so only the folder check stands between it and a copy:
    # vsmlrt's own engines beside the models are not this app's to touch.
    _cache(elsewhere / "theirs.engine.cache", b"not ours", age=5)
    mpv.vsmlrt.TARGET = str(elsewhere / "new.engine")
    mpv.run()
    assert not (elsewhere / "new.engine.cache").exists()


def test_filesystem_errors_never_break_the_build(mpv):
    mpv.engines.rmdir()  # listdir now raises
    assert mpv.run() == "interpolated"


def test_failed_seeded_build_is_retried_without_the_seed(mpv):
    _cache(mpv.engines / "b.engine.cache", b"seed", age=10)
    mpv.vsmlrt.FAIL_NEXT = 1
    backend = FakeBackend()

    assert mpv.run(backend=backend) == "interpolated"

    calls = mpv.vsmlrt.CALLS
    assert len(calls) == 2
    assert not (mpv.engines / "new.engine.cache").exists(), "the seed must be gone"
    # The retry gets the backend as the script built it, not as RIFE() left it.
    assert calls[1] is not backend
    assert calls[1].custom_args == ["--precisionConstraints=obey"]
    assert _is_genuine(mpv)


def test_failure_without_a_seed_is_not_retried(mpv):
    mpv.vsmlrt.FAIL_NEXT = 1
    with pytest.raises(RuntimeError, match="trtexec"):
        mpv.run()
    assert len(mpv.vsmlrt.CALLS) == 1


def test_clip_is_never_copied(mpv):
    # Regression: the first draft deep-copied every argument, and a VideoNode
    # cannot be copied -- RIFE() never ran, on every call.
    assert mpv.run(clip=Uncopyable()) == "interpolated"


def test_only_tensorrt_with_an_engine_folder_gets_the_block():
    trt = _render("C:/cache")
    assert "_fm_seeded_rife" in trt
    ast.parse(trt)
    assert "_fm_" not in _render("")
    assert "_fm_" not in render_vpy(RifeParams(model=426, mpv_root="C:/mpv", engine_folder="C:/cache",
                                               backend=BACKEND_NCNN))
