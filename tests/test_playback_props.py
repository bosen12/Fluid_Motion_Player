"""apply() changes four playback props; remove() has to give them back.

They used to be set and never restored. The owner's own mpv.conf sets
`interpolation` (tscale=oversample, 24fps content on a 165Hz panel), so one F3
left mpv's smoothmotion off for the rest of that player's life -- a quieter
playback than the one they configured, with nothing saying why. A default
config lost video-sync=audio the same way.

Restoring is cheap, which is what makes it the right default: measured against
a playing 1080p mpv, setting any of the four back costs no playback time
(within the +/-0.02s noise of a control), where an hwdec change stalls
0.2-0.4s.
"""
from pathlib import Path

import pytest

from fluid_motion.config import Settings
from fluid_motion.core import inject
from fluid_motion.core.inject import apply, remove
from fluid_motion.core.mpv_ipc import IpcError

PID = 7
INFO = {"container_fps": 24.0, "fps": 24.0, "display_fps": 165.0, "interpolation": False}

# What the owner's C:\mpv\mpv.conf actually sets, except video-sync, which is
# left at mpv's default so the test covers a prop a default config loses too.
OWNER = {
    "interpolation": True,
    "video-sync": "audio",
    "hr-seek-framedrop": False,
    "temporal-dither": False,
}


class _Mpv:
    """An mpv with a vf chain and props, that records every set()."""

    def __init__(self, **props):
        self.props = {"hwdec": "auto-copy", **props}
        self.vf: list = []
        self.sets: list[tuple[str, object]] = []
        self.refuse_sets: set[str] = set()
        self.unreadable: set[str] = set()
        self.path = "fake-pipe"

    def get(self, name, *, timeout=None):
        if name in self.unreadable:
            raise IpcError("mpv IPC timed out")
        if name == "vf":
            return list(self.vf)
        if name in self.props:
            return self.props[name]
        raise IpcError("property unavailable")

    def set(self, name, value):
        if name in self.refuse_sets:
            raise IpcError("error running command")
        self.sets.append((name, value))
        self.props[name] = value

    def command(self, *args, timeout=None):
        if args[:2] == ("vf", "add"):
            self.vf = [{"name": "vapoursynth", "label": "fluid"}]
        elif args[:2] == ("vf", "remove"):
            if not self.vf:
                # mpv errors on a label that is already gone.
                raise IpcError("error running command")
            self.vf = []


def _apply(mpv, tmp_path: Path) -> None:
    apply(mpv, Settings(profile="2x", mpv_root=str(tmp_path)), tmp_path, info=INFO, pid=PID)


def test_turning_the_filter_off_gives_the_player_its_own_settings_back(tmp_path):
    mpv = _Mpv(**OWNER)

    _apply(mpv, tmp_path)
    assert mpv.props["interpolation"] is False, "precondition: RIFE runs without mpv's own blending"
    assert mpv.props["video-sync"] == "display-resample"

    remove(mpv, pid=PID)

    assert mpv.props["interpolation"] is True, "mpv's smoothmotion stayed off after RIFE did"
    assert mpv.props["video-sync"] == "audio"
    assert PID not in inject._PREV_PROPS, "a finished restore left its record behind"


def test_a_filter_the_lua_took_off_for_a_seek_does_not_lose_the_original(tmp_path):
    """The seek path: zz-fluid-ipc.lua removes @fluid itself and the watcher
    re-applies, without remove() in between. The second apply finds our own
    values in place and must not record those as "what the player had"."""
    mpv = _Mpv(**OWNER)

    _apply(mpv, tmp_path)
    mpv.vf = []  # the lua's own `vf remove @fluid` on seek
    _apply(mpv, tmp_path)
    remove(mpv, pid=PID)

    assert mpv.props["interpolation"] is True
    assert mpv.props["video-sync"] == "audio"


def test_the_value_handed_back_is_the_latest_one_we_overrode(tmp_path):
    """The user switched video-sync while RIFE was on, and a re-apply after a
    seek overrode it again. Their switch is the value to give back, not the
    one the config file had before either of those happened."""
    mpv = _Mpv(**OWNER)

    _apply(mpv, tmp_path)
    mpv.props["video-sync"] = "display-vdrop"
    mpv.vf = []
    _apply(mpv, tmp_path)
    remove(mpv, pid=PID)

    assert mpv.props["video-sync"] == "display-vdrop"


def test_a_prop_changed_after_us_is_not_overwritten(tmp_path):
    """Only a prop still holding our value is ours to restore. One the user
    changed since -- no re-apply in between -- is a later choice than ours."""
    mpv = _Mpv(**OWNER)

    _apply(mpv, tmp_path)
    mpv.props["video-sync"] = "display-vdrop"
    remove(mpv, pid=PID)

    assert mpv.props["video-sync"] == "display-vdrop", "a later choice was undone"
    assert mpv.props["interpolation"] is True, "the untouched one still comes back"


def test_props_that_already_matched_are_left_alone_both_ways(tmp_path):
    mpv = _Mpv(**{name: want for name, want in inject._PLAYBACK_PROPS})

    _apply(mpv, tmp_path)
    remove(mpv, pid=PID)

    touched = [name for name, _ in mpv.sets if name in dict(inject._PLAYBACK_PROPS)]
    assert touched == [], f"set props nobody needed changed: {touched}"


def test_an_unreadable_prop_is_not_recorded_as_something_to_restore(tmp_path):
    """None is "could not ask". Restoring it would be inventing a setting."""
    mpv = _Mpv(**OWNER)
    mpv.unreadable.add("interpolation")

    _apply(mpv, tmp_path)
    applied_sets = len(mpv.sets)
    mpv.unreadable.clear()
    remove(mpv, pid=PID)

    restored = [name for name, _ in mpv.sets[applied_sets:]]
    assert "interpolation" not in restored, "restored a value that was never read"
    assert "video-sync" in restored, "precondition: the readable one is still restored"


def test_a_refused_restore_is_kept_for_the_next_remove(tmp_path):
    """Same rule as a refused hwdec restore: the record is the only copy of
    the player's own value, so a timed-out set must not throw it away."""
    mpv = _Mpv(**OWNER)
    _apply(mpv, tmp_path)

    mpv.refuse_sets.add("interpolation")
    remove(mpv, pid=PID)
    assert mpv.props["interpolation"] is False
    assert inject._PREV_PROPS[PID] == {"interpolation": True}
    assert mpv.props["video-sync"] == "audio", (
        "a refusal is an answer -- mpv is there, and the other prop still had to go back"
    )

    mpv.refuse_sets.clear()
    mpv.vf = [{"name": "vapoursynth", "label": "fluid"}]
    remove(mpv, pid=PID)
    assert mpv.props["interpolation"] is True


def test_an_mpv_that_stops_answering_costs_one_timeout_not_one_per_prop(tmp_path):
    """mpv serves IPC from one thread: no answer to the first read means none
    to the rest. Asking anyway spends a full command timeout per saved prop
    -- on the quit path, where stop() removes the filter from every player."""
    mpv = _Mpv(**OWNER)
    _apply(mpv, tmp_path)
    assert len(inject._PREV_PROPS[PID]) == 2, "precondition: two props to restore"

    asked: list[str] = []
    real_get = mpv.get

    def silent_get(name, *, timeout=None):
        if name in dict(inject._PLAYBACK_PROPS):
            asked.append(name)
            raise IpcError("mpv IPC timed out")
        return real_get(name, timeout=timeout)

    mpv.get = silent_get
    inject.restore_playback_props(mpv, pid=PID)

    assert len(asked) == 1, f"kept asking a player that had stopped answering: {asked}"
    assert inject._PREV_PROPS[PID] == {"interpolation": True, "video-sync": "audio"}, (
        "props that were never tried were dropped"
    )


def test_a_player_that_went_away_is_forgotten(monkeypatch, tmp_path):
    """Nothing to restore a departed player to, and pids are reused."""
    from fluid_motion.core import watcher as watcher_mod

    inject._PREV_PROPS[4321] = {"interpolation": True}
    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path)))
    monkeypatch.setattr(watcher_mod, "iter_mpv_processes", lambda: [])
    monkeypatch.setattr(watcher_mod, "engine_cache_info", lambda: engine._engine_cache)

    engine.tick()

    assert 4321 not in inject._PREV_PROPS


@pytest.mark.parametrize("name", [name for name, _ in inject._PLAYBACK_PROPS])
def test_every_forced_prop_is_one_remove_restores(tmp_path, name):
    """Derived from _PLAYBACK_PROPS, so a fifth forced prop is covered the day
    it is added rather than the day somebody notices it sticks."""
    want = dict(inject._PLAYBACK_PROPS)[name]
    original = "audio" if isinstance(want, str) else (not want)
    mpv = _Mpv(**{n: w for n, w in inject._PLAYBACK_PROPS})
    mpv.props[name] = original

    _apply(mpv, tmp_path)
    assert mpv.props[name] == want
    remove(mpv, pid=PID)

    assert mpv.props[name] == original
