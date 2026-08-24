from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from fluid_motion.core.gpu import flicker_risk


RIFE_46 = 46
RIFE_425 = 425
RIFE_426 = 426
# Past this RIFE cannot keep up in real time anyway, and every distinct multi
# compiles and caches its own TensorRT engine. Without a cap, a 12fps clip on
# the 144 profile asks for 12x and a 240Hz panel on "display" asks for 10x.
MAX_MULTI = 8


def rife_onnx_name(model: int) -> str:
    text = str(int(model))
    major = int(text[0])
    tail = text[1:]
    suffix = ""
    if len(text) >= 4 and text[-1] in "12":
        suffix = "_lite" if text[-1] == "1" else "_heavy"
        tail = text[1:-1]
    minor = int(tail) if tail else 0
    return f"rife_v{major}.{minor}{suffix}.onnx"


def rife_label(model: int) -> str:
    name = rife_onnx_name(model).removeprefix("rife_").removesuffix(".onnx")
    return "RIFE " + name.replace("v", "").replace("_", " ")


def rife_modulus(model: int) -> int:
    """vsmlrt RIFE spatial alignment for impl 1.

    4.25 lite downsamples two extra octaves (128). 4.26 and the 4.25/4.26
    heavy variants need 64. Everything else, including 4.6 and 4.25, is 32.
    Impl 2 pads inside the ONNX; this value is still written into the script
    so the v1 fallback is not stuck on a hardcoded 32.
    """
    text = str(int(model))
    major = int(text[0])
    tail = text[1:]
    kind = ""
    if len(text) >= 4 and text[-1] in "12":
        kind = "lite" if text[-1] == "1" else "heavy"
        tail = text[1:-1]
    minor = int(tail) if tail else 0
    if (major, minor) >= (4, 26):
        return 64
    if (major, minor) == (4, 25) and kind == "lite":
        return 128
    if (major, minor) == (4, 25) and kind == "heavy":
        return 64
    return 32


@dataclass
class RifeParams:
    model: int = RIFE_426
    scene_threshold: float = 0.10
    trt_streams: int = 1
    fp16: bool = True
    cuda_graph: bool = False
    profile: str = "2x"
    engine_folder: str = ""
    mpv_root: str = ""
    cache_size_mb: int = 8192
    gpu_name: str = ""
    force_accel: bool = False


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


def target_multi(profile: str, source_fps: Fraction | None, display_fps: float | None) -> tuple[int, bool]:
    """Return (multi, video_player).

    vsmlrt's RIFE call only produces extra frames for an integer multi — a
    Fraction (e.g. 2.5) is accepted without error but silently generates no
    speedup at all — so any target is always rounded to the nearest usable
    integer instead of passed through as-is.
    """
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
    # Round to the nearest achievable integer. If that's 1x, the source is
    # already close enough to the target that there's nothing to gain.
    snapped = round(float(target) / float(src))
    if snapped <= 1:
        return 1, False
    return min(snapped, MAX_MULTI), False


def _py_multi(multi: int | Fraction) -> str:
    if isinstance(multi, Fraction):
        return f"Fraction({multi.numerator}, {multi.denominator})"
    return str(int(multi))


def render_vpy(params: RifeParams, source_fps: Fraction | None = None, display_fps: float | None = None) -> str:
    multi, _video_player = target_multi(params.profile, source_fps, display_fps)
    engine = Path(params.engine_folder).as_posix() if params.engine_folder else ""
    engine_arg = f',\n        engine_folder=r"{engine}"' if engine else ""
    # CUDA graphs + parallel streams tile-flicker on RTX 50 (Blackwell); unknown
    # GPUs fail toward the same safe path instead of assuming they're fine.
    # force_accel is a user-requested escape hatch to test past that gate.
    risky = flicker_risk(params.gpu_name)
    safe_mode = risky and not params.force_accel
    cuda_graph = False if safe_mode else bool(params.cuda_graph)
    streams = 1 if safe_mode else max(1, int(params.trt_streams))
    gpu_note = params.gpu_name or "unknown"
    if safe_mode:
        mode_note = "safe mode (no CUDA graph / single stream)"
    elif risky:
        mode_note = "accelerated mode — user override of flicker-risk GPU, may tile-flicker"
    else:
        mode_note = "accelerated mode"

    return f'''# Fluid Motion — RIFE TensorRT (generated, do not edit)
# GPU: {gpu_note} — {mode_note}
import os
import sys
from fractions import Fraction

import vapoursynth as vs
from vapoursynth import core

MPV_ROOT = r"{params.mpv_root}"
if MPV_ROOT:
    sys.path.insert(0, MPV_ROOT)
    sys.path.insert(0, os.path.join(MPV_ROOT, "vs-plugins"))
    _cuda = os.path.join(MPV_ROOT, "vs-plugins", "vsmlrt-cuda")
    if os.path.isdir(_cuda):
        os.environ["PATH"] = _cuda + os.pathsep + os.environ.get("PATH", "")

from vsmlrt import RIFE, Backend

# RIFE is temporal. Default VS thread count races get_frame and shreds the picture.
core.num_threads = 1
core.max_cache_size = {int(params.cache_size_mb)}

SC_THRESHOLD = {float(params.scene_threshold):.4f}
RIFE_MODEL = {int(params.model)}
RIFE_ONNX = "{rife_onnx_name(params.model)}"
RIFE_FORMAT = vs.RGBH
TRT_FP16 = {bool(params.fp16)}
TRT_STREAMS = {int(streams)}
TRT_CUDA_GRAPH = {bool(cuda_graph)}
MOD = {rife_modulus(params.model)}
MULTI = {_py_multi(multi)}

if "video_in" not in globals():
    raise RuntimeError("Fluid Motion expects mpv video_in")

clip = video_in
fps = Fraction(container_fps).limit_denominator(1000)

is_hd = clip.width > 1024 or clip.height >= 600
est_matrix = 1 if is_hd else 5
clip = core.std.SetFrameProps(clip, _Matrix=est_matrix)

if clip.format.color_family in (vs.YUV, vs.GRAY):
    sc_clip = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
else:
    sc_clip = core.resize.Bilinear(clip, format=vs.GRAY8, matrix=est_matrix)
sc_clip = sc_clip.misc.SCDetect(SC_THRESHOLD)

if clip.format.color_family != vs.RGB:
    clip = core.resize.Bicubic(clip, format=RIFE_FORMAT, matrix_in=est_matrix)
else:
    clip = core.resize.Bicubic(clip, format=RIFE_FORMAT)

# mpv video_in reports a fake huge length; Trim so RIFE can shift n / n+1.
MAX_LEN = 2000000
if clip.num_frames <= 0 or clip.num_frames > MAX_LEN:
    clip = clip.std.Trim(length=MAX_LEN)
    sc_clip = sc_clip.std.Trim(length=MAX_LEN)

clip = clip.std.CopyFrameProps(sc_clip, ["_SceneChangePrev", "_SceneChangeNext"])

models_root = os.path.join(MPV_ROOT, "vs-plugins", "models") if MPV_ROOT else ""
v2 = os.path.join(models_root, "rife_v2", RIFE_ONNX)
impl = 2 if os.path.isfile(v2) else 1

# Implementation 2 pads internally (SVP path). v1 needs the model's vsmlrt
# modulus with black borders — extra Stack/Flip padding caused a flashing lattice.
pad_w = pad_h = 0
if impl == 1:
    pad_w = (-clip.width) % MOD
    pad_h = (-clip.height) % MOD
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

clip = RIFE(
    clip,
    multi=MULTI,
    model=RIFE_MODEL,
    _implementation=impl,
    backend=Backend.TRT(
        fp16=TRT_FP16,
        force_fp16=True,
        tf32=True,
        use_cuda_graph=TRT_CUDA_GRAPH,
        static_shape=True,
        num_streams=TRT_STREAMS,
        use_jit_convolutions=False,
        tiling_optimization_level=0,
        output_format=1{engine_arg},
    ),
)

if pad_w or pad_h:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

_out_fps = (fps if fps > 0 else Fraction(24000, 1001)) * MULTI
clip = core.std.AssumeFPS(clip, fpsnum=_out_fps.numerator, fpsden=_out_fps.denominator)
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix=est_matrix)
clip.set_output()
'''


def write_vpy(path: Path, params: RifeParams, source_fps: Fraction | None = None, display_fps: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_vpy(params, source_fps, display_fps), encoding="utf-8")
    return path
