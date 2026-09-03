"""One connection, several threads.

The watcher tick, the pywebview bridge (set_enabled), the hotkey loop and the
bootstrap thread all reach the same MpvIpc. `_apply_lock` in the Engine only
covers apply/remove -- every read path runs unguarded -- so the serialisation
has to live in the connection itself.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from fluid_motion.core.mpv_ipc import IpcError, MpvIpc


class _Handle:
    """A socket-shaped fake that answers every request it is given.

    It records how many commands had been written when the first read was
    served, which is what tells serialised apart from interleaved.
    """

    def __init__(self, second_caller_ready: threading.Event):
        self.writes: list[str] = []
        self.writes_at_first_read: int | None = None
        self._pending = b""
        self._ready = second_caller_ready
        self._guard = threading.Lock()

    # -- socket surface ---------------------------------------------------
    def settimeout(self, _timeout):
        pass

    def setblocking(self, _flag):
        pass

    def close(self):
        pass

    def sendall(self, data: bytes) -> None:
        msg = json.loads(data.decode("utf-8"))
        reply = json.dumps(
            {"request_id": msg["request_id"], "error": "success", "data": msg["command"][0]}
        )
        with self._guard:
            self.writes.append(msg["command"][0])
            self._pending += (reply + "\n").encode("utf-8")

    def recv(self, n: int) -> bytes:
        if self.writes_at_first_read is None:
            # Let the second caller get as far as it is going to get. With the
            # connection serialised it is parked on the lock and has written
            # nothing; without it, it has already put its command on the wire.
            self._ready.wait(timeout=2.0)
            time.sleep(0.1)
            with self._guard:
                self.writes_at_first_read = len(self.writes)
        with self._guard:
            out, self._pending = self._pending[:n], self._pending[n:]
        return out


def test_a_second_thread_cannot_put_a_command_on_the_wire_mid_exchange():
    """Interleaved writes are not the visible failure -- the stolen reply is.

    Both threads share one `_buf`, so whichever drains it first consumes every
    line in it, including the other's reply, and discards the ones whose
    request_id does not match. The loser then blocks to its own deadline and
    reports an IPC timeout that never happened on the wire. Worse, `_req` is
    incremented unguarded, so the two can be handed the same request_id and
    the reply genuinely cannot be routed.
    """
    ready = threading.Event()
    handle = _Handle(ready)
    ipc = MpvIpc(handle, "unix", "fake-pipe")
    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def first():
        try:
            results["alpha"] = ipc.command("alpha", timeout=5.0)
        except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
            errors["alpha"] = exc

    def second():
        ready.set()
        try:
            results["beta"] = ipc.command("beta", timeout=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors["beta"] = exc

    t1 = threading.Thread(target=first)
    t1.start()
    # Give the first caller the connection before the second one asks for it,
    # so the ordering under test is "second arrives mid-exchange".
    time.sleep(0.05)
    t2 = threading.Thread(target=second)
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"a caller failed: {errors}"
    assert handle.writes_at_first_read == 1, (
        "the second command reached the wire while the first exchange was open"
    )
    assert results == {"alpha": "alpha", "beta": "beta"}, (
        "a caller got the other's answer, or none at all"
    )


def test_a_caller_that_cannot_get_the_connection_fails_instead_of_corrupting_it():
    """The budget is the caller's own timeout: it already decided how long the
    command was worth waiting for. Refusing is safe only because
    snapshot_playback reports an unreadable vf as unknown rather than as
    "no filter loaded" -- see test_player_link."""
    handle = _Handle(threading.Event())
    ipc = MpvIpc(handle, "unix", "fake-pipe")
    ipc._lock.acquire()
    try:
        with pytest.raises(IpcError):
            ipc.command("get_property", "vf", timeout=0.1)
    finally:
        ipc._lock.release()
    assert handle.writes == [], "a refused command must not reach the wire"


class _SilentHandle:
    """Accepts every write and never replies. What a wedged mpv is on the wire.

    mpv services its IPC from one thread, so this is not a contrived state: an
    mpv compiling a TensorRT engine looks exactly like this, and so does one
    that has already exited (the peek in _read returns nothing either way).
    """

    def __init__(self):
        self.writes: list[str] = []

    def settimeout(self, _timeout):
        pass

    def close(self):
        pass

    def sendall(self, data: bytes) -> None:
        self.writes.append(json.loads(data.decode("utf-8"))["command"][1])

    def recv(self, _n: int) -> bytes:
        return b""


class _AnsweringHandle:
    """Answers every get_property from a dict, so a healthy read still reads."""

    def __init__(self, props: dict[str, object]):
        self.props = props
        self.writes: list[str] = []
        self._pending = b""

    def settimeout(self, _timeout):
        pass

    def close(self):
        pass

    def sendall(self, data: bytes) -> None:
        msg = json.loads(data.decode("utf-8"))
        name = msg["command"][1]
        self.writes.append(name)
        reply = json.dumps(
            {"request_id": msg["request_id"], "error": "success", "data": self.props.get(name)}
        )
        self._pending += (reply + "\n").encode("utf-8")

    def recv(self, n: int) -> bytes:
        out, self._pending = self._pending[:n], self._pending[n:]
        return out


def test_a_silent_mpv_costs_one_timeout_for_the_whole_snapshot_not_fifteen():
    """Measured before the budget existed: one get() 2.50s, the full snapshot
    37.6s -- and every field came back the same default the first failure had
    already fixed, so the other fourteen reads could not change the answer.
    tick() runs at TICK_SECONDS = 0.3 and walks players serially.
    """
    from fluid_motion.core.inject import snapshot_playback

    handle = _SilentHandle()
    ipc = MpvIpc(handle, "unix", "fake-pipe")

    budget = 0.3
    started = time.monotonic()
    info = snapshot_playback(ipc, budget=budget)
    elapsed = time.monotonic() - started

    assert len(handle.writes) == 1, (
        f"spent the budget {len(handle.writes)} times over: {handle.writes}"
    )
    assert elapsed < budget * 2, f"the batch cost {elapsed:.2f}s against a {budget}s budget"
    assert info["vf_ok"] is False, (
        "a snapshot nobody answered must read as 'could not ask', never as 'no filter'"
    )
    assert info["media"] == "" and info["fps"] == ""


def test_the_budget_does_not_cut_a_player_that_is_answering():
    """The other half: a working mpv must still be read in full. A budget that
    bought its speed by dropping properties would be a worse bug than the one
    it fixed -- the panel would just go blank instead of freezing.
    """
    from fluid_motion.core.inject import snapshot_playback

    handle = _AnsweringHandle(
        {"media-title": "ep01.mkv", "width": 1920, "height": 1080, "container-fps": 23.976}
    )
    ipc = MpvIpc(handle, "unix", "fake-pipe")

    info = snapshot_playback(ipc)

    assert "vf" in handle.writes, "the one property whose failure changes a decision went unasked"
    # Fourteen, not the fifteen a silent mpv provokes: `filename` is only read
    # as a fallback when `media-title` comes back empty.
    assert len(handle.writes) == 14, f"only {len(handle.writes)} of 14 properties were read"
    assert info["vf_ok"] is True
    assert info["media"] == "ep01.mkv"
    assert (info["width"], info["height"]) == (1920, 1080)


def test_rate_snapshot_shares_the_budget_too():
    """Three reads, not fifteen, but on the bridge thread -- so the 7.5s it used
    to cost against a silent mpv landed on the window, not on a background tick.
    """
    from fluid_motion.core.inject import rate_snapshot

    handle = _SilentHandle()
    ipc = MpvIpc(handle, "unix", "fake-pipe")

    budget = 0.3
    started = time.monotonic()
    rates = rate_snapshot(ipc, budget=budget)
    elapsed = time.monotonic() - started

    assert len(handle.writes) == 1, f"spent the budget {len(handle.writes)} times over"
    assert elapsed < budget * 2, f"the batch cost {elapsed:.2f}s against a {budget}s budget"
    assert rates["interpolation"] is False
