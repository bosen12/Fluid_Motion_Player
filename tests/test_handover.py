"""A second copy asking the running one to show itself.

single.handover_or_continue writes "show" into the hotkey file and exits; the
running instance's hotkey thread is what raises the window. The two ends are
wired at very different times -- start() launches that thread, while app.main()
only supplies the callback after `import webview`, create_window and the tray
handshake -- so the request can arrive before there is anything to serve it.
"""
from fluid_motion.config import Settings
from fluid_motion.core.watcher import Engine
from fluid_motion.paths import hotkey_path


def _engine() -> Engine:
    return Engine(Settings(enabled=False))


def test_a_show_that_arrives_before_the_window_exists_is_not_lost():
    """_consume_hotkey unlinks the file before dispatching, so a request the
    engine cannot serve yet used to be read, deleted, and dropped -- nothing
    retried it and the user's click simply did nothing.

    Measured window between start() returning and set_on_show() being called:
    72ms with webview already imported, and that is the floor -- the real path
    also pays a cold `import webview` and tray_ok.wait(timeout=1.5).
    """
    engine = _engine()
    hotkey_path().write_text("show", encoding="utf-8")

    engine._consume_hotkey()  # the hotkey thread, before app.main() got there

    assert not hotkey_path().is_file(), "the file is consumed either way"

    shown: list[int] = []
    engine.set_on_show(lambda: shown.append(1))

    assert shown == [1], "the click was eaten and never replayed"


def test_wiring_the_callback_does_not_invent_a_show():
    """The other half, and the one that matters more.

    A fix that simply called on_show whenever it was set would raise the window
    on every single launch -- which is exactly what --start-hidden exists to
    prevent, and how autostart runs the app. Only a request that really arrived
    may be replayed.
    """
    engine = _engine()
    shown: list[int] = []

    engine.set_on_show(lambda: shown.append(1))

    assert shown == [], "raised the window with nobody having asked"


def test_a_show_with_the_window_already_up_goes_straight_through():
    engine = _engine()
    shown: list[int] = []
    engine.set_on_show(lambda: shown.append(1))
    hotkey_path().write_text("show", encoding="utf-8")

    engine._consume_hotkey()

    assert shown == [1]
    assert engine._pending_show is False, "served and still held as pending"


def test_a_held_show_is_replayed_once_and_not_again():
    """Two windows over one click would be worse than none."""
    engine = _engine()
    hotkey_path().write_text("show", encoding="utf-8")
    engine._consume_hotkey()

    shown: list[int] = []
    engine.set_on_show(lambda: shown.append(1))
    engine.set_on_show(lambda: shown.append(2))

    assert shown == [1]
