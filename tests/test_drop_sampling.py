"""frame-drop-count is cumulative, so a missing reading is not a zero.

Every other unreadable property in a snapshot degrades to something inert: a
missing time-pos drops the sample, and vf carries an explicit vf_ok so "could
not ask" cannot be read as "no filter loaded". The drop counter was the one
place a failed read was spelled the same way as a measurement.
"""
import pytest

from fluid_motion.config import Settings
from fluid_motion.core.inject import DROP_LIMIT


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
