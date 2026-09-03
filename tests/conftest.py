import pytest


@pytest.fixture(autouse=True)
def isolated_appdata(tmp_path, monkeypatch):
    """No test may ever write to the real %APPDATA%\\FluidMotion.

    Every path in fluid_motion.paths that touches config.json, the engine
    cache, or the heartbeat/hotkey/seek_hold files resolves through the
    APPDATA env var. A test that builds an Engine/Settings and lets it save
    (directly or via update_settings()/set_enabled()) will otherwise happily
    overwrite the real user's config on whatever machine runs the suite --
    which is exactly what happened here: a test's tmp_path ended up written
    into the live mpv_root, breaking the app until someone noticed. Autouse
    means every test gets this for free; nobody has to remember to ask for it.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    # The log resolves its path once and keeps it in a module global, so the
    # env var alone is not enough: whichever test logs first pins the
    # directory, and the other two hundred write into that one's tmp_path.
    # Not destructive on its own -- the real %APPDATA% is never reached, and
    # that was checked -- but it makes a test that reads the log see other
    # tests' lines, and it is one refactor away from being destructive. AX
    # Player's conftest resets its equivalent for the same reason.
    from fluid_motion import log as log_mod

    monkeypatch.setattr(log_mod, "_log_path", None)
