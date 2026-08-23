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
