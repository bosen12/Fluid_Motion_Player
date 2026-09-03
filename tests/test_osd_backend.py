"""The on-screen line has to name the backend that actually ran.

It said "TensorRT" unconditionally, which was true for as long as TensorRT
was the only path. The ncnn/Vulkan backend exists precisely for machines
without it, and those are the machines the line was wrong on.
"""
import pytest

from fluid_motion.config import Settings
from fluid_motion.core import inject
from fluid_motion.core.vs_script import BACKEND_NCNN, BACKEND_TRT


class _Ipc:
    def __init__(self):
        self.texts: list[str] = []
        self.vf = [{"name": "vapoursynth", "label": "fluid"}]

    def get(self, name):
        if name == "vf":
            return self.vf
        if name == "hwdec":
            return "auto-copy"
        return None

    def set(self, name, value):
        pass

    def command(self, *args, **kwargs):
        if args and args[0] == "show-text":
            self.texts.append(args[1])


@pytest.fixture()
def announce(monkeypatch, tmp_path):
    monkeypatch.setattr(inject, "write_vpy", lambda *a, **kw: None)
    monkeypatch.setattr(inject, "script_output_path", lambda root: tmp_path / "f.vpy")

    def _run(backend: str) -> str:
        ipc = _Ipc()
        inject.apply(
            ipc,
            Settings(profile="2x", rife_model=426),
            tmp_path,
            announce=True,
            # 24fps source against a 60Hz display: a real 2x ask, so apply()
            # gets past its multi <= 1 early return and reaches the OSD line.
            info={"container_fps": 24.0, "fps": 24.0, "display_fps": 60.0,
                  "interpolation": False},
            pid=None,
            backend=backend,
        )
        assert ipc.texts, "apply() announced nothing"
        return ipc.texts[-1]

    return _run


def test_the_tensorrt_line_still_says_tensorrt(announce):
    assert "TensorRT" in announce(BACKEND_TRT)


def test_an_ncnn_machine_is_not_told_it_is_running_tensorrt(announce):
    line = announce(BACKEND_NCNN)
    assert "TensorRT" not in line, f"an ncnn player was shown {line!r}"
    assert "ncnn" in line
