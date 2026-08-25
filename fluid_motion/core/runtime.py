from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fluid_motion.core.vs_script import rife_label, rife_onnx_name
from fluid_motion.paths import default_mpv_root, engine_cache_dir


@dataclass
class Check:
    id: str
    label: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeStatus:
    mpv_root: str
    ready: bool
    checks: list[Check] = field(default_factory=list)
    rife_onnx: str = ""
    vsmlrt: str = ""
    vstrt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mpv_root": self.mpv_root,
            "ready": self.ready,
            "checks": [c.to_dict() for c in self.checks],
            "rife_onnx": self.rife_onnx,
            "vsmlrt": self.vsmlrt,
            "vstrt": self.vstrt,
            "engine_cache": str(engine_cache_dir()),
        }


def _exists(path: Path) -> bool:
    return path.is_file()


def diagnose(mpv_root: str | Path | None = None) -> RuntimeStatus:
    root = Path(mpv_root) if mpv_root else default_mpv_root()
    checks: list[Check] = []

    mpv_exe = root / "mpv.exe"
    checks.append(
        Check("mpv", "mpv", mpv_exe.is_file(), str(mpv_exe) if mpv_exe.is_file() else "找不到 mpv.exe")
    )

    # Two layouts exist in the wild. Repackaged mpv+VapourSynth bundles drop a
    # root vapoursynth.dll next to mpv.exe; the official portable install
    # (what install_vapoursynth builds) pip-installs the Python module and
    # leaves only VSScript.dll at the root. mpv's bridge links against
    # VSScript in both cases -- verified by running --vf=vapoursynth against
    # an official mpv build on an official-recipe tree -- so testing only for
    # vapoursynth.dll reports a perfectly good install as missing.
    vs_dll = root / "vapoursynth.dll"
    vs_script = root / "VSScript.dll"
    vs_ok = vs_dll.is_file() or vs_script.is_file()
    checks.append(
        Check(
            "vapoursynth",
            "VapourSynth",
            vs_ok,
            "已就緒" if vs_ok else "mpv 未附帶 VapourSynth（可自動安裝）",
        )
    )

    misc = root / "vs-plugins" / "MiscFilters.dll"
    checks.append(
        Check("scdetect", "場景偵測 (misc.SCDetect)", misc.is_file(), str(misc) if misc.is_file() else "缺少 MiscFilters.dll")
    )

    vsmlrt = None
    for candidate in (root / "vsmlrt.py", root / "vs-plugins" / "vsmlrt.py"):
        if candidate.is_file():
            vsmlrt = candidate
            break
    checks.append(
        Check("vsmlrt", "vs-mlrt", vsmlrt is not None, str(vsmlrt) if vsmlrt else "尚未安裝 vsmlrt.py")
    )

    vstrt = None
    for name in ("vstrt.dll", "vsmlrt.dll"):
        candidate = root / "vs-plugins" / name
        if candidate.is_file():
            vstrt = candidate
            break
    nvinfer = list((root / "vs-plugins").glob("nvinfer*.dll")) if (root / "vs-plugins").is_dir() else []
    cuda_dir = root / "vs-plugins" / "vsmlrt-cuda"
    trt_ok = vstrt is not None and (bool(nvinfer) or cuda_dir.is_dir())
    trt_detail = "TensorRT 插件就緒" if trt_ok else "需要 vstrt.dll 與 TensorRT/CUDA runtime"
    checks.append(Check("tensorrt", "TensorRT + CUDA", trt_ok, trt_detail))

    def _find_onnx(model: int) -> Path | None:
        name = rife_onnx_name(model)
        for folder in ("rife_v2", "rife"):
            for base in (root / "vs-plugins" / "models", root / "models"):
                candidate = base / folder / name
                if candidate.is_file():
                    return candidate
        return None

    onnx46 = _find_onnx(46)
    onnx425 = _find_onnx(425)
    onnx426 = _find_onnx(426)
    onnx = onnx426 or onnx425 or onnx46
    checks.append(
        Check("rife46", rife_label(46), onnx46 is not None, str(onnx46) if onnx46 else "尚未下載 rife_v4.6.onnx")
    )
    checks.append(
        Check("rife425", rife_label(425), onnx425 is not None, str(onnx425) if onnx425 else "尚未下載 rife_v4.25.onnx")
    )
    checks.append(
        Check("rife426", rife_label(426), onnx426 is not None, str(onnx426) if onnx426 else "尚未下載 rife_v4.26.onnx")
    )

    python_ok = (root / "python.exe").is_file() and (root / "python312.dll").is_file()
    checks.append(
        Check("python", "mpv 內嵌 Python 3.12", python_ok, str(root / "python.exe") if python_ok else "缺少內嵌 Python")
    )

    core_ok = all(c.ok for c in checks if c.id not in {"rife46", "rife425", "rife426"})
    ready = core_ok and (onnx46 is not None or onnx425 is not None or onnx426 is not None)
    return RuntimeStatus(
        mpv_root=str(root),
        ready=ready,
        checks=checks,
        rife_onnx=str(onnx or ""),
        vsmlrt=str(vsmlrt or ""),
        vstrt=str(vstrt or ""),
    )
