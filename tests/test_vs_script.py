from fractions import Fraction
from pathlib import Path

from fluid_motion.core.vs_script import RifeParams, parse_fps, render_vpy, target_multi, write_vpy


def test_rife_onnx_name_and_label():
    from fluid_motion.core.vs_script import rife_label, rife_onnx_name

    assert rife_onnx_name(46) == "rife_v4.6.onnx"
    assert rife_onnx_name(425) == "rife_v4.25.onnx"
    assert rife_onnx_name(426) == "rife_v4.26.onnx"
    assert "4.25" in rife_label(425)
    assert "4.26" in rife_label(426)
    assert "4.6" in rife_label(46)
    assert rife_onnx_name(4251) == "rife_v4.25_lite.onnx"
    assert "4.25" in rife_label(4251)
    assert "lite" in rife_label(4251)


def test_rife_modulus_matches_vsmlrt():
    from fluid_motion.core.vs_script import rife_modulus

    assert rife_modulus(46) == 32
    assert rife_modulus(425) == 32
    assert rife_modulus(426) == 64
    assert rife_modulus(4251) == 128
    assert rife_modulus(4252) == 64
    assert rife_modulus(4262) == 64


def test_render_vpy_modulus_follows_model():
    assert "MOD = 32" in render_vpy(RifeParams(model=425, mpv_root=r"C:\mpv"))
    assert "MOD = 64" in render_vpy(RifeParams(model=426, mpv_root=r"C:\mpv"))
    assert "MOD = 128" in render_vpy(RifeParams(model=4251, mpv_root=r"C:\mpv"))
    assert "MOD = 32" in render_vpy(RifeParams(model=46, mpv_root=r"C:\mpv"))


def test_parse_fps_fraction_string():
    assert parse_fps("24000/1001") == Fraction(24000, 1001)
    assert parse_fps("24") == Fraction(24, 1)
    assert parse_fps(0) is None
    assert parse_fps("unavailable") is None


def test_target_multi_2x_is_integer():
    multi, player = target_multi("2x", Fraction(24000, 1001), 144)
    assert multi == 2
    assert player is False


def test_target_multi_120_from_24_is_integer():
    multi, player = target_multi("120", Fraction(24, 1), None)
    assert multi == 5
    assert player is False


def test_target_multi_60_from_film_snaps_to_nearest_integer():
    # 60 / 23.976 = 2.5025x — vsmlrt's RIFE silently produces no extra frames
    # for a Fraction multi, so this must round to a usable integer (3), not
    # pass a Fraction through.
    multi, player = target_multi("60", Fraction(24000, 1001), None)
    assert player is False
    assert multi == 3
    assert isinstance(multi, int)


def test_target_multi_60fps_source_stays_integer_2x():
    multi, player = target_multi("120", Fraction(60000, 1001), None)
    assert multi == 2
    assert player is False


def test_target_multi_3x_on_60fps_stays_3x():
    multi, player = target_multi("3x", Fraction(60, 1), None)
    assert multi == 3
    assert player is False


def test_target_multi_120_from_60_is_2x():
    multi, player = target_multi("120", Fraction(60, 1), None)
    assert multi == 2
    assert player is False


def test_target_multi_60_on_60fps_is_passthrough():
    multi, player = target_multi("60", Fraction(60, 1), None)
    assert multi == 1


def test_target_multi_rounds_down_when_closer_to_lower_integer():
    # 100 / 23.976 = 4.171x -> rounds down to 4, not up.
    multi, player = target_multi("display", Fraction(24000, 1001), 100.0)
    assert multi == 4
    assert player is False


def test_target_multi_never_returns_a_fraction():
    for profile, src, display in [
        ("60", Fraction(24000, 1001), None),
        ("120", Fraction(24000, 1001), None),
        ("144", Fraction(30000, 1001), None),
        ("display", Fraction(24000, 1001), 100.0),
    ]:
        multi, _ = target_multi(profile, src, display)
        assert isinstance(multi, int)


def test_target_multi_3x_on_film_stays_3x():
    multi, player = target_multi("3x", Fraction(24, 1), None)
    assert multi == 3
    assert player is False


def test_render_contains_rife_46_and_trt():
    text = render_vpy(RifeParams(mpv_root=r"C:\mpv", engine_folder=r"C:\cache"))
    assert "RIFE_MODEL = 426" in text
    assert "rife_v4.26.onnx" in text
    assert "use_jit_convolutions=False" in text
    assert "rife_v4.6.onnx" in render_vpy(RifeParams(model=46, mpv_root=r"C:\mpv"))
    assert "Backend.TRT" in text
    assert "use_cuda_graph" in text
    assert "video_in" in text
    assert "SCDetect" in text
    assert "engine_folder" in text
    assert "matrix_in=est_matrix" in text
    assert "TRT_CUDA_GRAPH = False" in text
    assert "core.num_threads = 1" in text
    assert "AssumeFPS" in text
    assert "vsmlrt-cuda" in text
    assert "MAX_LEN" in text
    assert "std.Trim" in text
    assert "2 ** 31" not in text
    assert "clip = RIFE(" in text
    assert "RIFEMerge" not in text
    assert "force_fp16=True" in text
    assert "rife_v2" in text
    assert ".get_frame(" not in text


def test_render_blackwell_gpu_forces_safe_mode_even_if_requested_fast():
    text = render_vpy(
        RifeParams(
            mpv_root=r"C:\mpv",
            gpu_name="NVIDIA GeForce RTX 5070 Ti",
            cuda_graph=True,
            trt_streams=3,
        )
    )
    assert "TRT_CUDA_GRAPH = False" in text
    assert "TRT_STREAMS = 1" in text
    assert "safe mode" in text


def test_render_non_blackwell_gpu_honors_requested_settings():
    text = render_vpy(
        RifeParams(
            mpv_root=r"C:\mpv",
            gpu_name="NVIDIA GeForce RTX 4090",
            cuda_graph=True,
            trt_streams=3,
        )
    )
    assert "TRT_CUDA_GRAPH = True" in text
    assert "TRT_STREAMS = 3" in text
    assert "accelerated mode" in text


def test_render_unknown_gpu_defaults_to_safe_mode():
    text = render_vpy(RifeParams(mpv_root=r"C:\mpv", cuda_graph=True, trt_streams=3))
    assert "TRT_CUDA_GRAPH = False" in text
    assert "TRT_STREAMS = 1" in text


def test_render_force_accel_overrides_blackwell_safe_mode():
    text = render_vpy(
        RifeParams(
            mpv_root=r"C:\mpv",
            gpu_name="NVIDIA GeForce RTX 5070 Ti",
            cuda_graph=True,
            trt_streams=3,
            force_accel=True,
        )
    )
    assert "TRT_CUDA_GRAPH = True" in text
    assert "TRT_STREAMS = 3" in text
    assert "user override" in text


def test_effective_backend_clamps_blackwell_without_override():
    from fluid_motion.core.vs_script import effective_backend

    streams, graph, safe = effective_backend(
        "NVIDIA GeForce RTX 5070 Ti", trt_streams=4, cuda_graph=True, force_accel=False
    )
    assert safe is True
    assert streams == 1
    assert graph is False


def test_force_accel_has_no_effect_on_safe_gpu():
    text = render_vpy(
        RifeParams(
            mpv_root=r"C:\mpv",
            gpu_name="NVIDIA GeForce RTX 4090",
            cuda_graph=True,
            trt_streams=3,
            force_accel=True,
        )
    )
    assert "TRT_CUDA_GRAPH = True" in text
    assert "user override" not in text


def test_write_vpy(tmp_path: Path):
    path = tmp_path / "fluid_rife.vpy"
    write_vpy(path, RifeParams(profile="2x", mpv_root=r"C:\mpv"))
    assert path.is_file()
    assert "MULTI = 2" in path.read_text(encoding="utf-8")
