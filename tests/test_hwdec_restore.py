"""A refused hwdec restore must not throw the mode away.

Nothing else in the system knows what the player was decoding with before
interpolation switched it to copy-back. The lua strips a stale @fluid on its
own, but it has never touched hwdec; the only two copies of that value are
the in-memory map and the file mirroring it. Losing both leaves a running
mpv paying copy-back's GPU->CPU transfer with no filter to show for it.
"""
import pytest

from fluid_motion.core import inject
from fluid_motion.core.mpv_ipc import IpcError


PID = 7788


@pytest.fixture(autouse=True)
def clean_map():
    inject._PREV_HWDEC.clear()
    yield
    inject._PREV_HWDEC.clear()


class _Ipc:
    """An mpv that answers reads but refuses the write -- busy, not gone."""

    def __init__(self, *, refuse: bool):
        self.refuse = refuse
        self.sets: list[tuple] = []

    def get(self, name):
        return "auto-safe"

    def set(self, name, value):
        if self.refuse:
            raise IpcError("timed out waiting for reply")
        self.sets.append((name, value))


def test_a_successful_restore_clears_the_record():
    ipc = _Ipc(refuse=False)
    inject.ensure_copyback_hwdec(ipc, PID)
    assert inject.hwdec_pids() == [PID], "control: the mode was saved"

    inject.restore_hwdec(ipc, PID)
    assert ("hwdec", "auto-safe") in ipc.sets
    assert inject.hwdec_pids() == [], "a restored mode must not stay pending"
    assert PID not in inject.stranded_hwdec()


def test_a_refused_restore_keeps_the_record():
    saver = _Ipc(refuse=False)
    inject.ensure_copyback_hwdec(saver, PID)

    inject.restore_hwdec(_Ipc(refuse=True), PID)
    assert inject.hwdec_pids() == [PID], (
        "the only copy of the mode was dropped on a command mpv merely "
        "refused; the retry and the next run both find nothing to restore"
    )

    # And it is still on disk, which is what the next run actually reads.
    assert inject.stranded_hwdec().get(PID) == "auto-safe"
