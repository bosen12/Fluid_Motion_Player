"""A removal that did not happen must not be reported as one that did.

_remove_from carries a full error path -- log the failure, set _error, return
False so the next tick retries. Nothing could reach it: every command inside
remove() swallowed its own IpcError, so an mpv refusing vf commands produced
"removed successfully".

Measured before the fix, against an mpv that answers reads and refuses every
vf command: both removes rejected, the filter still on the graph,
_remove_from returned True and _error was empty.

That matters beyond the missing log line. Everything after the removals
assumes the graph is clear -- restore_hwdec takes the player back off
copy-back, and VapourSynth cannot be fed a GPU-resident frame, so a leftover
filter plus a restored hwdec is a filter that cannot run.
"""
import pytest

from fluid_motion.core.inject import remove
from fluid_motion.core.mpv_ipc import IpcError

FLUID_VF = [{"name": "vapoursynth", "label": "fluid"}]


class _Ipc:
    """Answers reads; refuses vf commands when told to."""

    def __init__(self, *, vf, refuse_vf=True, vf_readable=True):
        self.path = "fake-pipe"
        self.vf = vf
        self.refuse_vf = refuse_vf
        self.vf_readable = vf_readable
        self.vf_reads = 0
        self.sets: list[tuple] = []
        self.commands: list[tuple] = []

    def get(self, name, *, timeout=None):
        if name == "vf":
            self.vf_reads += 1
            if not self.vf_readable:
                raise IpcError("mpv IPC timed out")
            return self.vf
        return None

    def set(self, name, value):
        self.sets.append((name, value))

    def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[:1] == ("vf",):
            if self.refuse_vf:
                raise IpcError("mpv IPC timed out")
            self.vf = []
        return None

    def close(self):
        pass


def test_a_refused_removal_with_the_filter_still_there_is_reported():
    ipc = _Ipc(vf=FLUID_VF)

    with pytest.raises(IpcError):
        remove(ipc, pid=4242)

    assert ipc.sets == [], (
        "hwdec was restored while the filter was still on the graph -- "
        "VapourSynth cannot be fed a GPU-resident frame"
    )


def test_a_refused_removal_with_nothing_left_is_not_invented():
    """mpv also errors when the label is already gone, which is the ordinary
    case for a second remove. Reporting that as a failure would have the tick
    retry forever against a player that is already clean."""
    ipc = _Ipc(vf=[], refuse_vf=True)

    remove(ipc, pid=4242)  # must not raise


def test_an_unreadable_vf_is_not_treated_as_a_leftover():
    """"Could not ask" is not "still there" -- the same distinction
    snapshot_playback draws with vf_ok. An mpv answering nothing keeps the
    behaviour this has always had rather than gaining a new failure mode."""
    ipc = _Ipc(vf=FLUID_VF, refuse_vf=True, vf_readable=False)

    remove(ipc, pid=4242)  # must not raise


def test_a_clean_removal_costs_no_extra_round_trip():
    """The leftover check runs only when something was refused. remove() is on
    the seek path -- the lua strips the filter on every seek and the watcher
    reconciles behind it -- so an extra vf read on the healthy path would be
    paid constantly for a case that is not happening."""
    ipc = _Ipc(vf=FLUID_VF, refuse_vf=False)

    remove(ipc, pid=4242)

    assert ipc.vf_reads == 1, f"the healthy path read vf {ipc.vf_reads} times"
