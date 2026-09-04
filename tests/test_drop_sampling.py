"""frame-drop-count is cumulative, so a missing reading is not a zero.

Every other unreadable property in a snapshot degrades to something inert: a
missing time-pos drops the sample, and vf carries an explicit vf_ok so "could
not ask" cannot be read as "no filter loaded". The drop counter was the one
place a failed read was spelled the same way as a measurement.
"""
import pytest

from fluid_motion.config import Settings
from fluid_motion.core.inject import DROP_LIMIT
from fluid_motion.core.mpv_ipc import IpcError


PID = 42
TARGET_FPS = 60.0
TICK = 0.3


@pytest.fixture()
def sampler(monkeypatch):
    """An engine fed one sample per call, on a clock the test controls."""
    from fluid_motion.core import watcher as watcher_mod

    clock = [1000.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: clock[0])
    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))
    state = {"pos": 0.0, "drops": 5000.0}

    def tick(*, readable: bool = True) -> float | None:
        clock[0] += TICK
        state["pos"] += TICK
        # 0.2 dropped frames a second -- 0.33% of a 60fps target. Comfortably
        # clean, and the figure the smoothing should stay parked on.
        state["drops"] += 0.06
        engine._update_realtime(
            PID,
            {
                "time_pos": state["pos"],
                "speed": 1.0,
                "drops": state["drops"] if readable else None,
            },
            TARGET_FPS,
        )
        return engine._drops.get(PID)

    tick.engine = engine
    return tick


def test_one_unreadable_drop_count_does_not_spike_the_rate(sampler):
    for _ in range(20):
        sampler()
    settled = sampler()
    assert settled is not None and settled < DROP_LIMIT, "control: clean reads read clean"

    sampler(readable=False)
    after = sampler()
    assert after is not None
    assert after < DROP_LIMIT, (
        f"one failed read left the drop rate at {after:.2%}. Differenced "
        "against a substituted zero, the next sample's delta is the whole "
        "cumulative counter, and the smoothing takes seconds to walk it back "
        "-- all of it spent calling healthy playback stuttering."
    )


def test_a_failed_read_is_recorded_as_absent_not_as_zero(sampler):
    """The spike is the symptom; what gets remembered is the cause.

    A tuple holding 0.0 is indistinguishable from a player that has genuinely
    dropped nothing, and it is that tuple the next tick differences against.
    """
    sampler()
    sampler(readable=False)
    _pos, drops, _at = sampler.engine._realtime_seen[PID]
    assert drops is None, f"an unreadable counter was recorded as {drops!r}"


# -- the same guarantee at the level that produces the value ---------------
class _PartlyReadableIpc:
    """Answers every property except the ones it is told to refuse."""

    def __init__(self, *, refuse: set[str]):
        self.path = "fake-pipe"
        self.refuse = refuse
        self.props = {
            "media-title": "ep01.mkv", "width": 1920, "height": 1080,
            "container-fps": 23.976, "estimated-vf-fps": 47.95, "display-fps": 60.0,
            "pause": False, "seeking": False, "time-pos": 12.0, "speed": 1.0,
            "frame-drop-count": 5000.0, "vf": [],
        }

    def get(self, name, *, timeout=None):
        if name in self.refuse:
            raise IpcError("mpv IPC timed out")
        return self.props.get(name)

    def close(self):
        pass


def test_the_snapshot_reports_an_unreadable_drop_count_as_absent():
    """The tests above drive the consumer: their fixture builds the info dict
    itself and hands in None, so they pin what _update_realtime does with it.
    Nothing pinned the half that *produces* it.

    Measured: changing snapshot_playback's `else None` back to `else 0.0` --
    which is the v9.7 bug exactly, a substituted zero differenced against the
    next cumulative reading -- left all 288 tests green. Same shape as the
    sidebar guard that sat one level below its bug (HANDOFF 9.37).
    """
    from fluid_motion.core.inject import snapshot_playback

    info = snapshot_playback(_PartlyReadableIpc(refuse={"frame-drop-count"}))

    assert info["drops"] is None, (
        f"an unreadable counter came back as {info['drops']!r}; the next tick "
        "differences against it and reports the whole counter as one tick's drops"
    )
    assert info["time_pos"] == 12.0, "the rest of the snapshot stopped being read"


def test_a_readable_drop_count_still_comes_through_as_a_number():
    """The other half: a producer that answered None for everything would pass
    the test above while making the drop rate permanently unmeasurable."""
    from fluid_motion.core.inject import snapshot_playback

    info = snapshot_playback(_PartlyReadableIpc(refuse=set()))

    assert info["drops"] == 5000.0
