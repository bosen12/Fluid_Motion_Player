"""Everything the web view reads off `state` has to be something state() puts there.

Engine.state() is the whole contract with the UI: it is polled every 900ms and
nothing else crosses that boundary. It produces fifteen keys and app.js reads
twelve of them, and the only coverage was `engine.state()["error"]` in two
tests -- so a key renamed on the Python side blanks a panel in silence. There
is no error, no exception, no failing test; `undefined` renders as an empty
readout, which is indistinguishable from "nothing to report yet".

The names come from app.js rather than a list written out here, for the reason
§9.17 settled: a hand-kept list is true of the fields somebody remembered. It
is the same shape as HANDOFF 9.38's finding one level up -- a producer nobody
was watching because every test built its output by hand.
"""
import re

import pytest

from fluid_motion.config import Settings
from fluid_motion.core.watcher import Engine
from fluid_motion.paths import ui_dir


def _app_js() -> str:
    return (ui_dir() / "app.js").read_text(encoding="utf-8")


def _top_level_reads() -> set[str]:
    return set(re.findall(r"\bstate\.([a-z_]+)", _app_js()))


def _nested_reads() -> set[tuple[str, str]]:
    return set(re.findall(r"\bstate\.([a-z_]+)\.([a-z_]+)", _app_js()))


@pytest.fixture()
def state() -> dict:
    return Engine(Settings()).state()


def test_the_reads_are_actually_being_found():
    """Guards the guard: a regex that stopped matching would make every
    assertion below pass by checking nothing."""
    reads = _top_level_reads()
    assert len(reads) >= 10, f"only found {sorted(reads)} -- the scan broke"
    for expected in ("players", "settings", "runtime", "gpu", "bootstrap"):
        assert expected in reads, f"app.js no longer reads state.{expected}"


def test_every_field_the_ui_reads_is_produced(state):
    missing = sorted(name for name in _top_level_reads() if name not in state)
    assert not missing, (
        f"app.js reads state.{missing} and state() does not produce it. "
        "The UI shows an empty readout, which looks like 'nothing yet'."
    )


def test_every_nested_field_the_ui_reads_is_produced(state):
    """The second level matters as much: state.gpu.name and
    state.bootstrap.running come from separate to_dict()s that can drift on
    their own."""
    missing = []
    for parent, child in sorted(_nested_reads()):
        value = state.get(parent)
        if isinstance(value, dict) and child not in value:
            missing.append(f"{parent}.{child}")
    assert not missing, f"app.js reads {missing}, which state() does not produce"


def test_the_keys_the_bridge_promises_are_all_there(state):
    """A blunt floor under the derived checks. If app.js were rewritten to read
    nothing, every test above would pass while the contract was gone."""
    for key in ("settings", "players", "gpu", "runtime", "error", "bootstrap",
                "engine_cache", "backend", "adapters", "connected", "player_count"):
        assert key in state, f"state() stopped producing {key!r}"
