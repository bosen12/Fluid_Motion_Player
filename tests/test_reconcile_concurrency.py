"""Slow IPC must not let an obsolete user request reach the player."""
import threading
from types import SimpleNamespace

import pytest

from fluid_motion.config import Settings
from fluid_motion.core import watcher


@pytest.fixture
def engine(monkeypatch, tmp_path):
    result = watcher.Engine(Settings(enabled=True, mpv_root=str(tmp_path)))
    monkeypatch.setattr(result, "_player_root", lambda *_: tmp_path)
    monkeypatch.setattr(watcher, "iter_mpv_processes", lambda: [])
    return result


@pytest.mark.parametrize("changes", [{"enabled": False}, {"profile": "4x"}])
def test_an_apply_superseded_while_waiting_does_not_touch_mpv(engine, monkeypatch, changes):
    calls = []
    monkeypatch.setattr(watcher, "apply", lambda *a, **kw: calls.append(a[1]))
    waiting = threading.Event()
    release = threading.Event()

    class WaitingLock:
        def acquire(self, **kw):
            waiting.set()
            assert release.wait(3)
            return True

        def release(self):
            pass

    monkeypatch.setattr(engine, "_apply_lock", WaitingLock())
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(engine._apply_to(42, object(), 2)))
    worker.start()
    try:
        assert waiting.wait(3)
        engine._store_settings(**changes)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert outcomes == [False]
    assert calls == [], "obsolete request rebuilt the filter after a newer setting"
    assert 42 not in engine._applied


def test_overlapping_ticks_skip_duplicate_discovery_without_waiting(engine, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def discover():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return []

    monkeypatch.setattr(watcher, "iter_mpv_processes", discover)
    errors = []

    def tick():
        try:
            engine.tick()
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=tick)
    second = threading.Thread(target=tick)
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        second.join(0.5)
        returned = not second.is_alive()
    finally:
        release.set()
        first.join(3)
        second.join(3)
    assert not errors
    assert returned, "a UI-triggered tick queued behind the blocked worker"
    assert calls == [1], "both ticks scanned the same players concurrently"
    engine.tick()
    assert calls == [1, 1], "the next scheduled tick must still run"


def test_tick_after_stop_does_not_discover_or_reconnect(engine, monkeypatch):
    calls = []
    monkeypatch.setattr(watcher, "iter_mpv_processes", lambda: calls.append(1) or [])
    engine.stop()
    engine.tick()
    assert calls == []


def test_an_apply_dispatched_after_disable_does_not_reenable_filter(engine, monkeypatch):
    engine._store_settings(enabled=False)
    monkeypatch.setattr(watcher, "apply", lambda *a, **kw: pytest.fail("applied while disabled"))
    assert engine._apply_to(42, object(), 2) is False


def test_stop_during_connect_closes_the_unpublished_pipe(engine, monkeypatch):
    closed = []
    ipc = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(watcher, "iter_mpv_processes", lambda: [SimpleNamespace(pid=42)])

    def connect(*args, **kwargs):
        engine.stop()
        return ipc

    monkeypatch.setattr(watcher, "connect_pid", connect)
    monkeypatch.setattr(watcher, "snapshot_playback", lambda _: pytest.fail("read after shutdown"))
    engine.tick()
    assert closed == [True]
    assert engine._ipc == {}


def test_stop_during_discovery_does_not_open_any_pipes(engine, monkeypatch):
    def discover():
        engine.stop()
        return [SimpleNamespace(pid=42)]

    monkeypatch.setattr(watcher, "iter_mpv_processes", discover)
    monkeypatch.setattr(watcher, "connect_pid", lambda *a, **kw: pytest.fail("connected after shutdown"))
    engine.tick()
    assert engine._ipc == {}


def test_stopped_tick_leaves_published_pipe_for_shutdown_removal(engine, monkeypatch):
    order = []
    removing = threading.Event()
    finish_removal = threading.Event()
    ipc = SimpleNamespace(close=lambda: order.append("close"))
    engine._ipc[42] = ipc
    monkeypatch.setattr(watcher, "iter_mpv_processes", lambda: [SimpleNamespace(pid=42)])

    def remove(*args, **kwargs):
        removing.set()
        assert finish_removal.wait(3)
        order.append("remove")

    monkeypatch.setattr(watcher, "remove", remove)
    shutdown = threading.Thread(target=engine.stop)

    def snapshot(_):
        shutdown.start()
        assert removing.wait(3)
        return {}

    monkeypatch.setattr(watcher, "snapshot_playback", snapshot)
    try:
        engine.tick()
        assert order == [], "tick closed a pipe still owned by shutdown"
    finally:
        finish_removal.set()
        shutdown.join(3)
    assert order == ["remove", "close"]
