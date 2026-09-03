"""What a tick still in flight is allowed to do once stop() has run.

stop() is not a barrier: it sets the event, drops the filter from every
connected player and returns, and app.py then tears down the tray and the
webview before os._exit(0). The three loop threads are daemons and nobody
joins them, so a tick that was already running gets to finish inside that
window -- and everything it does lands after the app has decided to quit.
"""
from pathlib import Path

import pytest

from fluid_motion.config import Settings
from fluid_motion.paths import heartbeat_path


@pytest.fixture()
def engine(monkeypatch):
    from fluid_motion.core import watcher as watcher_mod

    monkeypatch.setattr(watcher_mod, "iter_mpv_processes", lambda: [])
    return watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))


class _FakeIpc:
    path = "fake-pipe"

    def close(self):
        pass


def test_a_tick_after_stop_does_not_resurrect_the_heartbeat(engine):
    """The lua treats a fresh timestamp as "Fluid Motion is running".

    Its strip_stale() runs only at file load, never on a timer, so a file
    opened inside the four-second alive window would keep a filter that
    nothing is maintaining any more -- and hwdec stays on copy-back until
    this app runs again to undo it.
    """
    engine.tick()
    assert heartbeat_path().is_file()

    engine.stop()
    assert not heartbeat_path().is_file(), "stop() should remove it"

    engine.tick()
    assert not heartbeat_path().is_file(), "a racing tick put it back"


def test_an_apply_after_stop_does_not_re_add_the_filter(engine, monkeypatch):
    """stop() takes the filter off; a queued apply must not put it back.

    Both sides go through _apply_lock, so they never overlap -- but the lock
    says nothing about which runs first, and losing that race leaves an mpv
    interpolating under a .vpy owned by a process that has exited.
    """
    from fluid_motion.core import watcher as watcher_mod

    applied: list[int] = []
    monkeypatch.setattr(
        watcher_mod, "apply",
        lambda *a, **kw: applied.append(kw.get("pid")),
    )
    monkeypatch.setattr(watcher_mod, "player_config_dir", lambda ipc: Path("Z:/cfg"))

    ipc = _FakeIpc()
    assert engine._apply_to(4321, ipc, 2) is True
    assert applied == [4321], "control: a normal apply reaches mpv"

    engine.stop()
    applied.clear()
    assert engine._apply_to(4321, ipc, 2) is False
    assert applied == [], "an apply that won the lock after stop() reached mpv"
