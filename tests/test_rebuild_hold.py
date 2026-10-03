"""Pausing across a filter rebuild (inject.hold_for_rebuild + zz-fluid-ipc.lua).

The behaviour itself lives in mpv and was measured against a real one (HANDOFF
§9.67): a rebuild stops mpv's core, audio underran when playing, and a vf
change while paused used to let the script tear the filter straight off.
These pin the Python half's decisions and ordering, and the script's contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fluid_motion.config import Settings
from fluid_motion.core.inject import HOLD_API, apply, hold_for_rebuild
from fluid_motion.core.mpv_ipc import IpcError


class Ipc:
    """Records every call in order; vf behaves like mpv's."""

    def __init__(self, props=None, *, refuse=()):
        self.props = {"container-fps": 24, "estimated-vf-fps": 24, "vsync-ratio": 1, "display-fps": 60}
        self.props.update(props or {})
        self.refuse = set(refuse)
        self.calls: list[tuple] = []
        self.vf: list = []

    def get(self, name, *, timeout=None):
        self.calls.append(("get", name))
        if ("get", name) in self.refuse:
            raise IpcError("no answer")
        if name == "vf":
            return self.vf
        if name in self.props:
            return self.props[name]
        raise IpcError("property not found")

    def set(self, name, value):
        self.calls.append(("set", name, value))
        if ("set", name) in self.refuse:
            raise IpcError("refused")
        self.props[name] = value

    def command(self, *args, **kwargs):
        self.calls.append(("command",) + args)
        if args[:1] == ("script-message",) and ("command", "script-message") in self.refuse:
            raise IpcError("refused")
        if args[:2] == ("vf", "add"):
            self.vf = [{"name": "vapoursynth", "label": "fluid"}]
        elif args[:2] == ("vf", "remove"):
            self.vf = []
        return None

    def close(self):
        pass


def test_playing_player_is_paused_and_handed_to_the_script():
    ipc = Ipc({HOLD_API: 1, "pause": False})
    assert hold_for_rebuild(ipc) == "resume"
    assert ("set", "pause", True) in ipc.calls
    assert ipc.calls.index(("set", "pause", True)) < ipc.calls.index(("command", "script-message", "fluid-hold", "resume"))


def test_paused_player_stays_paused_but_is_shielded():
    # The user's pause is theirs: nothing set, but the script still has to
    # ignore its refresh seek or it tears the new filter straight off.
    ipc = Ipc({HOLD_API: 1, "pause": True})
    assert hold_for_rebuild(ipc) == "keep"
    assert not any(c[:2] == ("set", "pause") for c in ipc.calls)
    assert ("command", "script-message", "fluid-hold", "keep") in ipc.calls


@pytest.mark.parametrize("props", [{"pause": False}, {HOLD_API: 0, "pause": False}, {HOLD_API: "1", "pause": False}])
def test_old_script_is_never_paused(props):
    # Nothing in an older script would ever resume it.
    ipc = Ipc(props)
    assert hold_for_rebuild(ipc) == ""
    assert not any(c[0] == "set" for c in ipc.calls)
    assert not any(c[:2] == ("command", "script-message") for c in ipc.calls)


def test_unanswered_reads_mean_no_hold():
    ipc = Ipc({HOLD_API: 1, "pause": False}, refuse={("get", "pause")})
    assert hold_for_rebuild(ipc) == ""
    assert not any(c[0] == "set" for c in ipc.calls)


def test_refused_pause_means_no_hold_message():
    ipc = Ipc({HOLD_API: 1, "pause": False}, refuse={("set", "pause")})
    assert hold_for_rebuild(ipc) == ""
    assert not any(c[:2] == ("command", "script-message") for c in ipc.calls)


def test_undelivered_hold_undoes_the_pause():
    # Paused, then the script never heard about it: left alone the video would
    # sit paused for good. Put it back.
    ipc = Ipc({HOLD_API: 1, "pause": False}, refuse={("command", "script-message")})
    assert hold_for_rebuild(ipc) == ""
    assert ipc.props["pause"] is False
    assert [c for c in ipc.calls if c[:2] == ("set", "pause")] == [("set", "pause", True), ("set", "pause", False)]


def _apply(ipc, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    settings = Settings(profile="2x", mpv_root=str(tmp_path))
    (tmp_path / "shaders").mkdir(exist_ok=True)
    apply(ipc, settings, tmp_path, pid=4242, backend="trt")


def test_apply_holds_before_it_touches_vf(tmp_path: Path, monkeypatch):
    ipc = Ipc({HOLD_API: 1, "pause": False})
    ipc.vf = [{"name": "vapoursynth", "label": "fluid"}]  # stale filter: remove, then add
    _apply(ipc, tmp_path, monkeypatch)
    hold = ipc.calls.index(("command", "script-message", "fluid-hold", "resume"))
    vf_ops = [i for i, c in enumerate(ipc.calls) if c[:2] == ("command", "vf")]
    assert vf_ops and all(i > hold for i in vf_ops), ipc.calls
    assert ipc.vf == [{"name": "vapoursynth", "label": "fluid"}]


def test_apply_on_an_old_player_never_pauses(tmp_path: Path, monkeypatch):
    ipc = Ipc({"pause": False})
    _apply(ipc, tmp_path, monkeypatch)
    assert ipc.props["pause"] is False
    assert any(c[:3] == ("command", "vf", "add") for c in ipc.calls)


# -- the script's half of the contract ----------------------------------------

def _lua() -> str:
    from fluid_motion.paths import resources_dir

    return (resources_dir() / "zz-fluid-ipc.lua").read_text(encoding="utf-8")


def test_script_advertises_the_hold():
    assert 'set_property_native("user-data/fluid/hold-api", 1)' in _lua()
    assert HOLD_API == "user-data/fluid/hold-api"


def test_script_ends_the_hold_on_playback_restart_and_ignores_its_own_seeks():
    lua = _lua()
    restart = lua[lua.index('register_event("playback-restart"'):]
    assert "end_hold()" in restart[:200]
    seek_hold = lua[lua.index("local function begin_seek_hold()"):]
    assert seek_hold.splitlines()[1].strip() == "if hold then"
    assert 'register_script_message("fluid-hold"' in lua
    assert 'add_hook("on_preloaded"' in lua


def test_script_only_resumes_what_it_paused():
    lua = _lua()
    end = lua[lua.index("local function end_hold()"):lua.index("local function begin_hold(")]
    assert "if resume and" in end
    assert 'begin_hold(mode == "resume")' in lua


def test_stale_seeking_notification_does_not_drop_a_fresh_filter():
    # A queued seeking=true from a new file's load arrived 4 ms after the hold
    # resumed and tore the just-built filter off (measured, real mpv). The
    # observer re-reads the live value before acting; the "seek" event still
    # drops the filter unconditionally for real seeks.
    lua = _lua()
    observer = lua[lua.index('observe_property("seeking"'):]
    observer = observer[:observer.index("end)")]
    assert 'if mp.get_property_bool("seeking") then' in observer
    seek_event = lua[lua.index('register_event("seek"'):]
    assert seek_event.splitlines()[1].strip() == "begin_seek_hold()"
