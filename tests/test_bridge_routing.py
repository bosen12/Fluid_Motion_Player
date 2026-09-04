"""Every Bridge method has to actually reach the engine.

test_bridge.py covers what set_backend *validates* and what clear_engine_cache
*reports*. A statement-deletion sweep found that nothing covers the rest: the
body of set_enabled, set_profile, set_model, set_autostart, start_setup, hide,
quit and open_engine_cache can each be emptied with the suite green, and so can
every `return self.get_state()`.

That surface is the whole API the web view has. A gutted setter is a control
that moves and does nothing -- the chip flips, the poll 900ms later reads the
old value back, and the only symptom is a toggle that will not stay put.

CLAUDE.md calls this file the security boundary; this is the other half of
that, which is that it is also the only boundary.
"""
import pytest

from fluid_motion.api import Bridge


class _SpyEngine:
    def __init__(self):
        self.calls: list[tuple] = []
        self.settings = type("S", (), {"enabled": False})()

    def state(self) -> dict:
        return {"marker": "from-engine", "calls": len(self.calls)}

    def set_enabled(self, enabled):
        self.calls.append(("set_enabled", enabled))

    def update_settings(self, **kwargs):
        self.calls.append(("update_settings", kwargs))

    def start_bootstrap(self):
        self.calls.append(("start_bootstrap",))

    def report(self, message):
        self.calls.append(("report", message))


@pytest.fixture()
def bridge():
    engine = _SpyEngine()
    hidden, quit_called = [], []
    api = Bridge(engine, lambda: hidden.append(1), lambda: quit_called.append(1))
    return api, engine, hidden, quit_called


@pytest.mark.parametrize(
    "method, argument, expected",
    [
        ("set_enabled", True, ("set_enabled", True)),
        ("set_enabled", False, ("set_enabled", False)),
        ("set_profile", "3x", ("update_settings", {"profile": "3x"})),
        ("set_model", 426, ("update_settings", {"rife_model": 426})),
        ("set_scene", 0.15, ("update_settings", {"scene_threshold": 0.15})),
        ("set_backend", "amd", ("update_settings", {"backend": "amd"})),
    ],
)
def test_every_setter_reaches_the_engine(bridge, method, argument, expected):
    api, engine, _, _ = bridge

    returned = getattr(api, method)(argument)

    assert engine.calls == [expected], f"{method} did not reach the engine"
    assert returned["marker"] == "from-engine", (
        f"{method} did not answer with the engine's own state"
    )


def test_the_setters_coerce_rather_than_trusting_the_web_view(bridge):
    """Everything here arrives from JS, where "426" and 426 are the same thing
    to the caller and not to Settings.from_dict."""
    api, engine, _, _ = bridge

    api.set_model("426")
    api.set_scene("0.2")
    api.set_enabled(1)

    assert engine.calls == [
        ("update_settings", {"rife_model": 426}),
        ("update_settings", {"scene_threshold": 0.2}),
        ("set_enabled", True),
    ]


def test_start_setup_starts_the_bootstrap(bridge):
    api, engine, _, _ = bridge

    returned = api.start_setup()

    assert engine.calls == [("start_bootstrap",)]
    assert returned["marker"] == "from-engine"


def test_hide_and_quit_reach_the_callables_they_were_handed(bridge):
    """The window's own close button and the tray's 結束 both end here. A no-op
    quit leaves a headless process with no tray icon and no window."""
    api, _, hidden, quit_called = bridge

    api.hide()
    api.quit()

    assert hidden == [1], "hide() did not reach the window"
    assert quit_called == [1], "quit() did not reach the shutdown path"


def test_autostart_writes_the_registry_entry_and_the_setting(bridge, monkeypatch):
    """Both halves, and they are separate: the entry is what Windows reads at
    login, the setting is what the checkbox reads back."""
    api, engine, _, _ = bridge
    written: list[tuple] = []
    monkeypatch.setattr(
        "fluid_motion.api.set_autostart", lambda on, cmd: written.append((on, cmd))
    )

    api.set_autostart(True)

    assert len(written) == 1 and written[0][0] is True, "no autostart entry was written"
    assert written[0][1], "the launch command handed over was empty"
    assert engine.calls == [("update_settings", {"autostart": True})]


def test_the_cache_folder_is_opened_and_nothing_else_is(bridge, monkeypatch):
    """os.startfile on a path built here -- pinned so it stays the engine cache
    directory and stays a single call."""
    from fluid_motion.paths import engine_cache_dir

    api, _, _, _ = bridge
    opened: list[str] = []
    monkeypatch.setattr("os.startfile", lambda path: opened.append(path), raising=False)

    api.open_engine_cache()

    assert opened == [str(engine_cache_dir())]


def test_autostart_file_names_the_real_entry():
    from fluid_motion.core.bootstrap import autostart_path

    api = Bridge(_SpyEngine(), lambda: None, lambda: None)

    assert api.autostart_file() == str(autostart_path())
