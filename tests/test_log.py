"""The diagnostic log, and what has to be in it to be worth having.

v1.6.0 shipped an AMD path that has never run on AMD hardware and asked users
to report back. A report needs something to report, and until now the only
thing this app could say about itself was one line of `_error` text in a
window that closes.
"""
from __future__ import annotations

from fluid_motion import log as log_mod


def test_the_log_lands_under_appdata_not_next_to_the_exe(tmp_path, monkeypatch):
    """A packaged app sits in Program Files or a read-only folder, so a log
    beside the exe is a log that silently fails to write."""
    log_mod.log("hello")

    target = log_mod.path()
    assert target.is_file()
    assert "FluidMotion" in str(target)
    assert "hello" in target.read_text(encoding="utf-8")


def test_a_line_carries_a_timestamp():
    log_mod.log("something happened")

    line = log_mod.path().read_text(encoding="utf-8").strip()
    stamp = line.split(" ", 1)[0]
    # ISO-8601 to milliseconds, the same shape AX Player's log uses.
    assert stamp.count(":") == 2 and "T" in stamp, stamp


def test_a_failed_write_never_reaches_the_caller(monkeypatch):
    """Logging is a diagnostic, not a feature. A full disk or a locked file
    must not take down the watcher loop that was trying to report a problem --
    which would turn a logged failure into an unlogged crash."""
    def explode(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", explode)

    log_mod.log("this cannot be written")  # must not raise


def test_the_log_rotates_instead_of_growing_without_end(monkeypatch):
    """The watcher ticks three times a second forever. The call sites log
    changes rather than state, but the cap is what stops one thing that starts
    changing every tick from burying everything before it."""
    monkeypatch.setattr(log_mod, "MAX_BYTES", 200)

    for i in range(60):
        log_mod.log(f"line {i} padded out to make this exceed the cap quickly")

    target = log_mod.path()
    previous = target.with_name(target.name + ".1")
    assert previous.is_file(), "nothing was rotated out"
    assert target.stat().st_size < 4000, "the live file kept growing past the cap"


def test_exceptions_are_logged_with_their_traceback():
    """The watcher swallows exceptions to stay alive. Without the traceback the
    log says only that something failed, which is where the old `_error`
    string already was."""
    try:
        raise ValueError("the actual cause")
    except ValueError:
        log_mod.log_exc("watcher tick")

    text = log_mod.path().read_text(encoding="utf-8")
    assert "watcher tick" in text
    assert "ValueError: the actual cause" in text
    assert "Traceback" in text


# -- what the engine actually records --------------------------------------
def test_startup_records_the_machine_and_what_it_resolved_to(monkeypatch):
    """The header of every bug report: which build, which adapters, which
    backend that resolved to, and where it is going to look."""
    from fluid_motion.config import Settings
    from fluid_motion.core import watcher as watcher_mod

    monkeypatch.setattr(
        watcher_mod, "detect_adapters", lambda **_kw: ["AMD Radeon RX 7900 XTX"]
    )
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: r)
    monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda r: None)
    monkeypatch.setattr(watcher_mod.Engine, "_loop", lambda self: None)
    monkeypatch.setattr(watcher_mod.Engine, "_heartbeat_loop", lambda self: None)
    monkeypatch.setattr(watcher_mod.Engine, "_hotkey_loop", lambda self: None)

    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))
    engine.start()
    engine.stop()

    text = log_mod.path().read_text(encoding="utf-8")
    assert "Fluid Motion" in text, "the build that produced the log is not named"
    assert "AMD Radeon RX 7900 XTX" in text, "the adapters that decided the backend are missing"
    assert "backend=" in text
    assert "Z:/cfg" in text


def test_a_backend_change_is_recorded_with_what_caused_it(monkeypatch):
    """The setting can stay on "auto" while the answer moves, so the adapters
    are logged alongside -- otherwise the line says what changed and not why."""
    from fluid_motion.config import Settings
    from fluid_motion.core import gpu
    from fluid_motion.core import watcher as watcher_mod

    vendors = [{gpu.NVIDIA}]
    monkeypatch.setattr(watcher_mod, "available_vendors", lambda **_kw: vendors[0])
    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))

    vendors[0] = {gpu.AMD}
    engine._backend()

    text = log_mod.path().read_text(encoding="utf-8")
    assert "trt -> ncnn" in text, "the backend change was not recorded"


def test_an_unchanged_backend_is_not_logged_every_tick(monkeypatch):
    """_backend() runs several times per tick. The point of the log is the line
    that mattered, not three a second saying nothing moved."""
    from fluid_motion.config import Settings
    from fluid_motion.core import gpu
    from fluid_motion.core import watcher as watcher_mod

    monkeypatch.setattr(watcher_mod, "available_vendors", lambda **_kw: {gpu.NVIDIA})
    engine = watcher_mod.Engine(Settings(mpv_root="Z:/cfg"))

    for _ in range(50):
        engine._backend()

    lines = [
        ln for ln in log_mod.path().read_text(encoding="utf-8").splitlines()
        if "backend:" in ln
    ]
    assert len(lines) == 1, f"logged {len(lines)} times for one unchanged backend"
