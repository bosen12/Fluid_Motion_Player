from fractions import Fraction
from pathlib import Path

from fluid_motion.core.vs_script import RifeParams, parse_fps, render_vpy, target_multi, write_vpy


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


def test_target_multi_60_from_film_is_fractional():
    multi, player = target_multi("60", Fraction(24000, 1001), None)
    assert player is True
    assert isinstance(multi, Fraction)
    assert float(multi) > 2


def test_render_contains_rife_46_and_trt():
    text = render_vpy(RifeParams(mpv_root=r"C:\mpv", engine_folder=r"C:\cache"))
    assert "RIFE_MODEL = 46" in text
    assert "Backend.TRT" in text
    assert "use_cuda_graph" in text
    assert "video_in" in text
    assert "SCDetect" in text
    assert "engine_folder" in text


def test_write_vpy(tmp_path: Path):
    path = tmp_path / "fluid_rife.vpy"
    write_vpy(path, RifeParams(profile="2x", mpv_root=r"C:\mpv"))
    assert path.is_file()
    assert "multi=2" in path.read_text(encoding="utf-8")
