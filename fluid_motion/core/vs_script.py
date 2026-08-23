from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


RIFE_46 = 46
MODULUS = 32  # RIFE 4.6 requires multiples of 32


@dataclass
class RifeParams:
    model: int = RIFE_46
    scene_threshold: float = 0.10
    trt_streams: int = 2
    fp16: bool = True
    cuda_graph: bool = True
    profile: str = "2x"
    engine_folder: str = ""
    mpv_root: str = ""
    cache_size_mb: int = 8192


def parse_fps(value: Any) -> Fraction | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return Fraction(value).limit_denominator(1001)
    text = str(value).strip()
    if not text or text in {"nan", "unavailable"}:
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            frac = Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError):
            return None
        return frac if frac > 0 else None
    try:
        frac = Fraction(text).limit_denominator(1001)
    except (ValueError, ZeroDivisionError):
        return None
    return frac if frac > 0 else None


def target_multi(profile: str, source_fps: Fraction | None, display_fps: float | None) -> tuple[int | Fraction, bool]:
    """Return (multi, video_player). Integer multi is preferred for realtime."""
    if profile == "3x":
        return 3, False
    if profile == "2x":
        return 2, False
    if profile == "display":
        target = display_fps or 120.0
    else:
        try:
            target = float(profile)
        except ValueError:
            return 2, False
    src = source_fps or Fraction(24000, 1001)
    if src <= 0:
        return 2, False
    multi = Fraction(target).limit_denominator(120) / src
    if multi < 1:
        return 2, False
    if multi.denominator == 1:
        return int(multi), False
    return multi, True


def _py_multi(multi: int | Fraction) -> str:
    if isinstance(multi, Fraction):
        return f"Fraction({multi.numerator}, {multi.denominator})"
    return str(int(multi))


def render_vpy(params: RifeParams, source_fps: Fraction | None = None, display_fps: float | None = None) -> str:
    multi, video_player = target_multi(params.profile, source_fps, display_fps)
    engine = Path(params.engine_folder).as_posix() if params.engine_folder else ""
    video_player_arg = ",\n    video_player=True" if video_player else ""
    engine_arg = f',\n        engine_folder=r"{engine}"' if engine else ""

    return f'''# Fluid Motion — RIFE 4.6 TensorRT (generated, do not edit)
import os
import sys
from fractions import Fraction

import vapoursynth as vs
from vapoursynth import core

MPV_ROOT = r"{params.mpv_root}"
if MPV_ROOT:
    sys.path.insert(0, MPV_ROOT)
    sys.path.insert(0, os.path.join(MPV_ROOT, "vs-plugins"))

from vsmlrt import RIFE, Backend

core.max_cache_size = {int(params.cache_size_mb)}

SC_THRESHOLD = {float(params.scene_threshold):.4f}
RIFE_MODEL = {int(params.model)}
RIFE_FORMAT = vs.RGBH
TRT_FP16 = {bool(params.fp16)}
TRT_STREAMS = {int(params.trt_streams)}
TRT_CUDA_GRAPH = {bool(params.cuda_graph)}
MOD = {MODULUS}

if "video_in" not in globals():
    raise RuntimeError("Fluid Motion expects mpv video_in")

clip = video_in
fps = Fraction(container_fps).limit_denominator(1000)
fps_num, fps_den = fps.numerator, fps.denominator

is_hd = clip.width > 1024 or clip.height >= 600
est_matrix = 1 if is_hd else 5

def ensure_matrix(node):
    matrix = node.get_frame(0).props.get("_Matrix")
    if matrix is None or matrix == 2:
        node = core.std.SetFrameProps(node, _Matrix=est_matrix)
    return node

clip = ensure_matrix(clip)

if clip.format.color_family in (vs.YUV, vs.GRAY):
    sc_clip = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
else:
    sc_clip = core.resize.Bilinear(clip, format=vs.GRAY8, matrix=est_matrix)
sc_clip = sc_clip.misc.SCDetect(SC_THRESHOLD)

pad_w = (-clip.width) % MOD
pad_h = (-clip.height) % MOD
if pad_w or pad_h:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

clip = core.resize.Bicubic(clip, format=RIFE_FORMAT)
clip = clip.std.CopyFrameProps(sc_clip, ["_SceneChangePrev", "_SceneChangeNext"])

models_root = os.path.join(MPV_ROOT, "vs-plugins", "models") if MPV_ROOT else ""
v2 = os.path.join(models_root, "rife_v2", "rife_v4.6.onnx")
impl = 2 if os.path.isfile(v2) else 1

clip = RIFE(
    clip,
    multi={_py_multi(multi)},
    model=RIFE_MODEL,
    _implementation=impl{video_player_arg},
    backend=Backend.TRT(
        fp16=TRT_FP16,
        use_cuda_graph=TRT_CUDA_GRAPH,
        static_shape=True,
        num_streams=TRT_STREAMS,
        output_format=1{engine_arg},
    ),
)

if pad_w or pad_h:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix=est_matrix)
clip.set_output()
'''


def write_vpy(path: Path, params: RifeParams, source_fps: Fraction | None = None, display_fps: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_vpy(params, source_fps, display_fps), encoding="utf-8")
    return path
