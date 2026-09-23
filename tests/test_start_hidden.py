"""--start-hidden is about one launch, and must not end up in config.json.

It is how the autostart entry runs the app (bootstrap.set_autostart writes
`start "" <exe> --start-hidden`). main() used to write it onto the Settings
object it had just loaded -- the object Engine then persists whole on every
change -- so one F3, tray toggle or chip click during an autostarted session
saved start_hidden=true, and every launch by hand after that opened straight
into the tray with no window.

These drive the real main() with webview and pystray replaced by fakes, so the
path under test is the one the app runs; nothing here opens a window or a tray
icon, and --demo keeps the watcher threads from starting.
"""
import json
import sys
import types

import pytest

from fluid_motion.paths import config_path


class _Window:
    def __init__(self):
        self.hidden = 0
        self.events = types.SimpleNamespace(closing=_Handlers())

    def hide(self):
        self.hidden += 1

    def show(self):
        pass

    def restore(self):
        pass

    def destroy(self):
        pass


class _Handlers:
    def __iadd__(self, handler):
        return self


def _fake_modules(monkeypatch, on_start):
    """webview + pystray stand-ins; on_start(bridge) runs inside webview.start."""
    made = {}

    def create_window(*_args, js_api=None, **_kwargs):
        made["window"] = _Window()
        made["bridge"] = js_api
        return made["window"]

    def start(shown=None, **_kwargs):
        if shown is not None:
            shown()
        on_start(made["bridge"])

    webview = types.ModuleType("webview")
    webview.create_window = create_window
    webview.start = start

    class _Icon:
        def __init__(self, *args, **kwargs):
            pass

        def run_detached(self):
            pass

        def stop(self):
            pass

    class _Menu:
        SEPARATOR = object()

        def __init__(self, *items):
            pass

    pystray = types.ModuleType("pystray")
    pystray.Icon = _Icon
    pystray.Menu = _Menu
    pystray.MenuItem = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setitem(sys.modules, "pystray", pystray)
    return made


@pytest.fixture(autouse=True)
def no_players(monkeypatch):
    # A bridge call can end in a tick; this machine's real mpv processes are
    # not this test's business.
    from fluid_motion.core import watcher as watcher_mod

    monkeypatch.setattr(watcher_mod, "iter_mpv_processes", lambda: [])


def _saved() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))


def test_a_change_during_an_autostarted_session_does_not_save_the_flag(monkeypatch):
    from fluid_motion import app

    _fake_modules(monkeypatch, lambda bridge: bridge.set_profile("3x"))

    assert app.main(["--start-hidden", "--demo"]) == 0

    saved = _saved()
    assert saved["profile"] == "3x", "precondition: the change did reach the file"
    assert saved["start_hidden"] is False, (
        "a one-launch flag was persisted; the next launch by hand starts hidden"
    )


def test_the_flag_still_hides_the_window_it_was_given_for(monkeypatch):
    from fluid_motion import app

    made = _fake_modules(monkeypatch, lambda bridge: None)

    app.main(["--start-hidden", "--demo"])

    assert made["window"].hidden == 1


def test_a_launch_by_hand_shows_the_window(monkeypatch):
    from fluid_motion import app

    made = _fake_modules(monkeypatch, lambda bridge: None)

    app.main(["--demo"])

    assert made["window"].hidden == 0


def test_a_start_hidden_the_user_saved_is_still_honoured(monkeypatch):
    """The setting itself keeps working for anyone who put it in the file."""
    from fluid_motion import app
    from fluid_motion.config import Settings, save_settings

    save_settings(Settings(start_hidden=True))
    made = _fake_modules(monkeypatch, lambda bridge: None)

    app.main(["--demo"])

    assert made["window"].hidden == 1
