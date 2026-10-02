"""TensorRT timing-cache seeding in the generated script.

The block runs inside mpv's VapourSynth, where nothing here can reach it, so
the tests cut it out of a real render_vpy() result and run it against a fake
vsmlrt. What ships is what is tested: the text is taken from the rendered
script, not from the constant it is built from.
"""
from __future__ import annotations

import ast
import os
import sys
import time
import types
from pathlib import Path

import pytest

from fluid_motion.core.vs_script import BACKEND_NCNN, RifeParams, render_vpy

START = "from vsmlrt import RIFE, Backend\n"
END = "\n# RIFE is temporal."


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
def harness(tmp_path, monkeypatch):
    """Run the shipped seed block against a fake vsmlrt in tmp_path/engines."""
    engines = tmp_path / "engines"
    engines.mkdir()
    fake = types.ModuleType("vsmlrt")
    fake.target = engines / "new.engine"

    def get_engine_path(*args, **kwargs):
        return str(fake.target)

    fake.get_engine_path = get_engine_path
    monkeypatch.setitem(sys.modules, "vsmlrt", fake)

    calls = []

    def rife(clip, multi=2, backend=None):
        calls.append(backend)
        # Like vsmlrt: the engine path is resolved through the module global,
        # and the backend is mutated on the way.
        fake.get_engine_path()
        backend.force_fp16 = False
        backend.custom_args.append("--precisionConstraints=obey")
        if fake.fail_next:
            fake.fail_next -= 1
            raise RuntimeError("trtexec execution fails")
        return "interpolated"

    fake.fail_next = 0
    ns = {"os": os, "RIFE": rife}
    exec(compile(_seed_block(_render(str(engines))), "seed", "exec"), ns)
    return types.SimpleNamespace(ns=ns, fake=fake, engines=engines, calls=calls, original=get_engine_path)


def _cache(path: Path, data: bytes, age: float) -> Path:
    path.write_bytes(data)
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def test_new_engine_is_seeded_from_the_newest_cache(harness):
    _cache(harness.engines / "a.engine.cache", b"old", age=100)
    _cache(harness.engines / "b.engine.cache", b"newest", age=10)
    _cache(harness.engines / "c.engine.cache", b"", age=1)  # a failed build's leftover

    path = harness.fake.get_engine_path()

    assert path == str(harness.fake.target)
    assert (harness.engines / "new.engine.cache").read_bytes() == b"newest"
    assert harness.ns["_fm_seeded"] == [path + ".cache"]


def test_built_engine_is_left_alone(harness):
    # mpv re-runs the script on every seek; with the engine there, nothing moves.
    _cache(harness.engines / "b.engine.cache", b"seed", age=10)
    harness.fake.target.write_bytes(b"x" * 2048)
    harness.fake.get_engine_path()
    assert not (harness.engines / "new.engine.cache").exists()
    assert harness.ns["_fm_seeded"] == []


def test_failed_tiny_engine_still_gets_a_seed(harness):
    # vsmlrt rebuilds an engine under 1024 bytes (a 0-byte one is a real
    # leftover in the owner's cache), so that rebuild is worth seeding too.
    _cache(harness.engines / "b.engine.cache", b"seed", age=10)
    harness.fake.target.write_bytes(b"")
    harness.fake.get_engine_path()
    assert (harness.engines / "new.engine.cache").read_bytes() == b"seed"


def test_own_cache_is_never_overwritten(harness):
    _cache(harness.engines / "b.engine.cache", b"seed", age=10)
    _cache(harness.engines / "new.engine.cache", b"mine", age=50)
    harness.fake.get_engine_path()
    assert (harness.engines / "new.engine.cache").read_bytes() == b"mine"
    assert harness.ns["_fm_seeded"] == []


def test_engine_outside_the_folder_is_not_seeded(harness, tmp_path):
    _cache(harness.engines / "b.engine.cache", b"seed", age=10)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # A cache there too, so only the folder check stands between it and a copy:
    # vsmlrt's own engines beside the models are not this app's to touch.
    _cache(elsewhere / "theirs.engine.cache", b"not ours", age=5)
    harness.fake.target = elsewhere / "new.engine"
    harness.fake.get_engine_path()
    assert not (elsewhere / "new.engine.cache").exists()


def test_filesystem_errors_never_break_the_lookup(harness):
    harness.engines.rmdir()  # listdir now raises
    assert harness.fake.get_engine_path() == str(harness.fake.target)


def test_failed_seeded_build_is_retried_without_the_seed(harness):
    _cache(harness.engines / "b.engine.cache", b"seed", age=10)
    harness.fake.fail_next = 1
    backend = FakeBackend()

    out = harness.ns["RIFE"](Uncopyable(), multi=2, backend=backend)

    assert out == "interpolated"
    assert len(harness.calls) == 2
    assert not (harness.engines / "new.engine.cache").exists(), "the seed must be gone"
    assert harness.fake.get_engine_path is harness.original, "no seeding on the retry"
    # The retry gets the backend as the script built it, not as RIFE() left it.
    assert harness.calls[1] is not backend
    assert harness.calls[1].force_fp16 is False  # mutated by the retry itself
    assert harness.calls[1].custom_args == ["--precisionConstraints=obey"]


def test_failure_without_a_seed_is_not_retried(harness):
    harness.fake.fail_next = 1
    with pytest.raises(RuntimeError, match="trtexec"):
        harness.ns["RIFE"](Uncopyable(), multi=2, backend=FakeBackend())
    assert len(harness.calls) == 1


def test_clip_is_never_copied(harness):
    # Regression: the first version deep-copied every argument, and a
    # VideoNode cannot be copied -- RIFE() never ran, on every call, which
    # would have switched interpolation off for everyone on NVIDIA.
    assert harness.ns["RIFE"](Uncopyable(), multi=2, backend=FakeBackend()) == "interpolated"


def test_only_tensorrt_with_an_engine_folder_gets_the_block():
    trt = _render("C:/cache")
    assert "_fm_seeding_get_engine_path" in trt
    ast.parse(trt)
    assert "_fm_" not in _render("")
    assert "_fm_" not in render_vpy(RifeParams(model=426, mpv_root="C:/mpv", engine_folder="C:/cache",
                                               backend=BACKEND_NCNN))
