"""The fps readout has to actually come from mpv's estimated-vf-fps.

The scar is in inject.py's own comment: the property was read under the name
"estimated-vfps", which mpv does not have, so _get's IpcError guard swallowed
the "property not found" reply and returned None every single time. The
readout silently fell back to container-fps for the whole life of that name,
and the fix shipped without a regression test -- reverting both call sites to
the wrong name still leaves the suite green.

Nothing here asserts on the spelling. A test that compares the string to
itself would pass for a name mpv does not have, which is the bug. Instead the
fake serves a container rate and a decoder rate that DIFFER, so the readout can
only produce the second one if the code asked mpv for something mpv answered.

Kept away from a real mpv, per this repo's testing rules: a wrong name is
observable from the fake side alone, because a name nothing serves reads as
absent.
"""
from typing import Any

import pytest

from fluid_motion.core.inject import rate_snapshot, snapshot_playback

CONTAINER = 23.976
# Higher than the container rate so the two are told apart, but under the 1.3x
# line live_source_fps() uses to spot a post-RIFE reading -- this is the plain
# "decoder is a little ahead of the container" case, not an interpolated one.
DECODER = 25.0


class _Ipc:
    """Serves exactly the property names mpv has, and nothing else.

    Duck-types MpvIpc. The absent-name behaviour is the point: get() answers
    None for anything unknown, which is what a wrong name looks like from here.
    """

    def __init__(self, props: dict[str, Any]):
        self.props = dict(props)
        self.asked: list[str] = []
        self.path = "fake-mpv-pipe"

    def get(self, name: str, *, timeout: float | None = None) -> Any:
        self.asked.append(name)
        if name == "vf":
            return []
        return self.props.get(name)

    def close(self) -> None:
        pass


def _ipc() -> _Ipc:
    return _Ipc({
        "container-fps": CONTAINER,
        "estimated-vf-fps": DECODER,
        "filename": "ep01.mkv",
        "width": 1920,
        "height": 1080,
    })


def test_the_two_rates_are_actually_distinguishable():
    """Guards the guard. If these ever collide, both checks below pass whether
    or not the property was read -- which is exactly how the original bug hid:
    every fake in the suite set the two to the same number."""
    assert DECODER != CONTAINER
    assert DECODER <= CONTAINER * 1.3, "above this, live_source_fps returns the container rate by design"


def test_snapshot_reports_the_decoder_rate_not_the_container_rate():
    snap = snapshot_playback(_ipc())
    assert snap["estimated_vfps"] == DECODER, (
        "mpv was never asked for a property it has -- the readout fell back to "
        "container-fps, which is what 'estimated-vfps' did for its whole life"
    )
    assert snap["fps"] == DECODER


def test_rate_snapshot_reports_the_decoder_rate_too():
    """The second call site. The name was wrong in both, and a fix to one is
    the kind of half-fix a single-site test would sign off on."""
    assert rate_snapshot(_ipc())["fps"] == DECODER


@pytest.mark.parametrize("snapshot", (snapshot_playback, rate_snapshot))
def test_the_container_rate_still_stands_in_when_mpv_cannot_estimate(snapshot):
    """The fallback is correct behaviour when the property is genuinely
    unavailable -- what was wrong was reaching it every time. Without this the
    checks above could be satisfied by deleting the fallback."""
    ipc = _Ipc({"container-fps": CONTAINER, "filename": "ep01.mkv"})
    assert snapshot(ipc)["fps"] == CONTAINER
