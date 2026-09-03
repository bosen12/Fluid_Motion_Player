"""Relationships between the tuning constants, not their digits.

A sweep that changed every module-level constant and re-ran the suite left 27
of 39 survivors here. Most are fine: vendor tokens, pinned upstream versions,
model ids -- nothing a test could judge without the network or the hardware.

These three are different. Each is a decision with a stated reason, and each
reason is a *relationship* between values rather than a number, so it can be
pinned without freezing a knob somebody may legitimately want to retune.
"""
from fluid_motion.core.inject import REALTIME_SMOOTHING
from fluid_motion.core.watcher import HOTKEY_POLL, HOUSEKEEPING_ACTIVE, HOUSEKEEPING_IDLE, TICK_SECONDS


# A fourth guard was written here and deleted: `SNAPSHOT_BUDGET ==
# COMMAND_TIMEOUT`, on the reasoning that a batch of reads is worth no more
# than the one stuck read it was already allowed to cost. It cannot fail --
# inject.py *defines* SNAPSHOT_BUDGET as COMMAND_TIMEOUT, so the two move
# together by construction and raising the timeout to 12.5 left the assertion
# green. A test that cannot fail is worse than a missing one, and it took a
# mutation to notice, which is the whole argument for running them.


def test_the_hotkey_poll_does_not_wait_behind_a_tick():
    """F3 is polled on its own thread precisely so a press is not queued behind
    tick()'s IPC round trips and process enumeration -- which used to put up to
    300ms in front of every toggle.

    A poll interval at or above the tick rate throws that away while looking
    like a harmless tidy-up: the toggle still works, just slowly, and nothing
    fails.
    """
    assert HOTKEY_POLL < TICK_SECONDS


def test_housekeeping_backs_off_when_nothing_is_playing():
    """The idle cadence exists because an app sitting in the tray with no
    player open was walking two directory trees three times a second and
    spawning nvidia-smi every 1.5s, forever. Idle has to be the slower of the
    two for that to mean anything.
    """
    assert HOUSEKEEPING_IDLE > HOUSEKEEPING_ACTIVE >= TICK_SECONDS


def test_the_realtime_smoothing_factor_is_a_weight_not_a_multiplier():
    """REALTIME_SMOOTHING is the share a new sample takes in the running
    figure. At 1.0 there is no smoothing left and a single coarse sample swings
    the readout; above 1.0 each sample overshoots and the value oscillates
    instead of converging; at or below 0 it never moves at all.

    Worth pinning because the number itself is tuned -- 0.35 recovers from a
    bad sample in 23 ticks -- so a future retune is expected, and only the
    range says which retunes are meaningful.
    """
    assert 0 < REALTIME_SMOOTHING < 1
