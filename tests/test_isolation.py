"""The conftest fixture that keeps the suite off the real %APPDATA%.

Its own docstring says this has already happened once -- "a test's tmp_path
ended up written into the live mpv_root, breaking the app until someone
noticed" -- and nothing has ever checked that the protection still works. AX
Player's conftest has a guard for its equivalent; this side had none.

The check is self-calibrating rather than a written-out list of paths. For
every zero-argument helper in fluid_motion.paths that returns a Path: if its
answer sits under %APPDATA% for one value of the variable, it has to move when
the variable moves. A helper that stops tracking it -- because someone cached
it in a module global, which is exactly what log._log_path does and why the
conftest resets that too -- fails here instead of writing into the real
directory on whoever runs the suite next.

Read-only locations (ui_dir, resources_dir, project_root...) never sit under
%APPDATA%, so they are excluded by the same rule rather than by a list.
"""
import inspect
from pathlib import Path

import pytest

from fluid_motion import paths


def _appdata_helpers() -> list[str]:
    """Every no-argument paths.* helper that answers with a Path."""
    found = []
    for name, fn in vars(paths).items():
        if name.startswith("_") or not callable(fn) or inspect.isclass(fn):
            continue
        if getattr(fn, "__module__", None) != paths.__name__:
            continue
        sig = inspect.signature(fn)
        if any(p.default is inspect.Parameter.empty for p in sig.parameters.values()):
            continue
        found.append(name)
    return sorted(found)


def _answers(monkeypatch, appdata: Path) -> dict[str, Path]:
    monkeypatch.setenv("APPDATA", str(appdata))
    out = {}
    for name in _appdata_helpers():
        try:
            value = getattr(paths, name)()
        except Exception:
            continue
        if isinstance(value, Path):
            out[name] = value
    return out


def test_the_helpers_are_actually_being_enumerated():
    """Guards the guard: a rename that empties this list would make every
    assertion below pass by checking nothing."""
    names = _appdata_helpers()
    for expected in ("roaming_dir", "config_path", "engine_cache_dir", "heartbeat_path"):
        assert expected in names, f"{expected} dropped out of the enumeration"


def test_every_path_under_appdata_moves_when_appdata_moves(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"

    before = _answers(monkeypatch, first)
    after = _answers(monkeypatch, second)

    tracked = [n for n, p in before.items() if first in p.parents or p == first]
    assert tracked, "no helper resolved under APPDATA at all -- the check is measuring nothing"

    stuck = [
        name for name in tracked
        if not (second in after[name].parents or after[name] == second)
    ]
    assert not stuck, (
        f"these still answer with the old %APPDATA% after it changed: {stuck}. "
        "A cached module global will write into the real user's directory."
    )


def test_the_suite_is_not_pointed_at_a_real_appdata():
    """The blunt half. If the fixture stopped applying entirely, every path
    above would agree with each other and still be wrong."""
    home = Path.home()
    roaming = paths.roaming_dir()

    assert roaming != home / "AppData" / "Roaming" / "FluidMotion", (
        "the suite is writing to the real %APPDATA%\\FluidMotion"
    )
    assert "tmp" in str(roaming).lower() or "temp" in str(roaming).lower(), (
        f"roaming_dir() is {roaming}, which is not a temporary directory"
    )


@pytest.mark.parametrize("name", ["config_path", "engine_cache_dir", "heartbeat_path",
                                  "hotkey_path", "debug_log_path"])
def test_each_written_path_lands_inside_the_redirected_dir(name):
    """Named individually as well, so a failure says which one escaped rather
    than only that something did."""
    value = getattr(paths, name)()
    roaming = paths.roaming_dir()

    assert roaming in value.parents or value.parent == roaming or roaming == value, (
        f"{name}() is {value}, outside {roaming}"
    )
