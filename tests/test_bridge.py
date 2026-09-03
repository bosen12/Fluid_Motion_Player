"""The JS-to-Python bridge, which had no tests at all.

It is the surface the web UI reaches, so it is also the surface a refactor can
quietly re-route. Every setter here is supposed to go through
Engine.update_settings, where the clamping and migration rules live -- nothing
pinned that, and a setter assigning straight onto the dataclass would have
shipped looking fine.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fluid_motion.api import Bridge
from fluid_motion.config import Settings
from fluid_motion.core import watcher as watcher_mod


@pytest.fixture()
def bridge(monkeypatch):
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: r)
    monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda r: None)
    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))
    return Bridge(engine, hide=lambda: None, quit_app=lambda: None)


def test_settings_go_through_validation_not_straight_onto_the_dataclass(bridge):
    """update_settings runs Settings.from_dict, which clamps. A setter that
    assigned directly would write an out-of-range value into the .vpy and then
    have it silently become something else on the next launch -- the exact
    failure update_settings' own comment records."""
    bridge.set_scene(0.9)

    assert bridge.engine.settings.scene_threshold == 0.30, "the clamp was bypassed"


def test_an_unknown_backend_falls_back_rather_than_pinning_the_machine(bridge):
    bridge.set_backend("nonsense")

    assert bridge.engine.settings.backend == "auto"


def test_a_known_backend_is_kept(bridge):
    bridge.set_backend("amd")

    assert bridge.engine.settings.backend == "amd"


# -- what clearing the cache is allowed to leave unsaid ---------------------
def test_engines_that_could_not_be_deleted_are_reported(bridge, monkeypatch, tmp_path):
    """clear() has always counted what it could not remove, and the caller
    threw it away. Locked is the ordinary case, not an edge one: a TensorRT
    engine is mapped while the filter runs, so clearing during playback
    half-works and the panel shows a number that simply did not drop."""
    from fluid_motion.core import engine_cache

    monkeypatch.setattr(
        "fluid_motion.core.engine_cache.clear", lambda path=None: (3, 2)
    )

    state = bridge.clear_engine_cache()

    assert "2" in state["error"], f"the locked count was not surfaced: {state['error']!r}"


def test_a_clean_clear_says_nothing(bridge, monkeypatch):
    """Empty clears the channel, so a successful retry takes the previous
    complaint down with it rather than leaving a stale warning up."""
    monkeypatch.setattr(
        "fluid_motion.core.engine_cache.clear", lambda path=None: (5, 0)
    )
    bridge.engine.report("something older")

    state = bridge.clear_engine_cache()

    assert state["error"] == ""


def test_clear_counts_a_held_engine_as_locked(tmp_path):
    """The behaviour the reporting depends on, against a real open handle
    rather than a stub."""
    from fluid_motion.core import engine_cache

    (tmp_path / "a.engine").write_bytes(b"x" * 10)
    (tmp_path / "b.engine").write_bytes(b"y" * 10)

    with open(tmp_path / "a.engine", "rb"):
        deleted, locked = engine_cache.clear(tmp_path)

    assert (deleted, locked) == (1, 1)
