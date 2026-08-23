from pathlib import Path

from fluid_motion.core.engine_cache import clear, info, next_growth_deadline


def test_info_on_empty_dir(tmp_path: Path):
    cache = info(tmp_path)
    assert cache.count == 0
    assert cache.total_bytes == 0
    assert cache.path == str(tmp_path)


def test_info_counts_files_recursively(tmp_path: Path):
    (tmp_path / "a.engine").write_bytes(b"x" * 10)
    nested = tmp_path / "rife_v2"
    nested.mkdir()
    (nested / "b.engine").write_bytes(b"y" * 20)
    cache = info(tmp_path)
    assert cache.count == 2
    assert cache.total_bytes == 30


def test_info_on_missing_dir(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    cache = info(missing)
    assert cache.count == 0
    assert cache.total_bytes == 0


def test_clear_removes_files_and_empty_subdirs(tmp_path: Path):
    nested = tmp_path / "rife_v2"
    nested.mkdir()
    (tmp_path / "a.engine").write_bytes(b"x")
    (nested / "b.engine").write_bytes(b"y")
    deleted, locked = clear(tmp_path)
    assert deleted == 2
    assert locked == 0
    assert info(tmp_path).count == 0
    assert not nested.exists()
    assert tmp_path.exists()


def test_clear_on_missing_dir_is_a_noop(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    deleted, locked = clear(missing)
    assert deleted == 0
    assert locked == 0


def test_next_growth_deadline_extends_on_growth():
    deadline = next_growth_deadline(current_bytes=200, previous_bytes=100, now=10.0, prior_deadline=5.0, grace=3.0)
    assert deadline == 13.0


def test_next_growth_deadline_holds_when_unchanged():
    deadline = next_growth_deadline(current_bytes=100, previous_bytes=100, now=10.0, prior_deadline=5.0, grace=3.0)
    assert deadline == 5.0


def test_next_growth_deadline_holds_when_shrinking():
    deadline = next_growth_deadline(current_bytes=50, previous_bytes=100, now=10.0, prior_deadline=5.0, grace=3.0)
    assert deadline == 5.0
