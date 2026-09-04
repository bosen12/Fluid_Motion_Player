"""Every element id app.js reaches for has to exist in index.html.

This is the same seam test_state_contract.py guards, one layer further out.
That one covers Python -> app.js (state keys); nothing covered app.js -> the
document, and that side fails harder.

The failure is not a blank panel. bind() runs seventeen
`$("id").addEventListener(...)` in a row, and it is called from the
DOMContentLoaded handler with no try/catch:

    document.addEventListener("DOMContentLoaded", () => {
      bind();            <- a missing id throws here
      render(mock);      <- never runs
      refresh();         <- never runs
      setInterval(refresh, 900);   <- never runs
    });

So one renamed id in a 5 KB hand-edited file leaves the window sitting on its
static skeleton: nothing polls, no control responds, and the toast that would
have shown the error lives inside refresh(), which never started. Python is
fine, the tray is fine, fluid_debug.log says nothing. The user sees an app
that opened and does nothing -- indistinguishable from "no mpv connected yet".

The names are read out of app.js rather than listed here, for the reason §9.17
settled: a hand-kept list is true of the ids somebody remembered.
"""
import re

import pytest

from fluid_motion.paths import ui_dir

QUOTED = r"['\"`]([^'\"`]+)['\"`]"


def _read(name: str) -> str:
    return (ui_dir() / name).read_text(encoding="utf-8")


def _looked_up() -> set[str]:
    """Ids app.js asks the document for, via getElementById or its $ wrapper."""
    js = _read("app.js")
    found = set(re.findall(rf"getElementById\(\s*{QUOTED}", js))
    # (?<![\w.]) so a member call like foo.$( ) is not mistaken for the helper.
    found |= set(re.findall(rf"(?<![\w.])\$\(\s*{QUOTED}\s*\)", js))
    return found


def _declared() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', _read("index.html")))


def test_the_dollar_helper_is_still_an_id_lookup():
    """Guards the guard, and it is the guard's one real assumption.

    Most of the lookups below are found by matching $("..."). If $ is ever
    changed to wrap querySelector instead, those strings become selectors and
    every assertion here would be comparing the wrong kind of name against
    index.html -- while still passing, because "#live" is not an id either way
    only when someone also updates the call sites.
    """
    js = _read("app.js")
    assert re.search(r"function \$\(\s*id\s*\)\s*\{\s*return document\.getElementById\(\s*id\s*\)", js), (
        "the $ helper is no longer a plain getElementById wrapper; the "
        "extraction below is reading strings whose meaning has changed"
    )


def test_the_extraction_actually_finds_the_lookups():
    """An empty set would make the contract below pass by checking nothing --
    which is how a green suite hides a dead UI."""
    found = _looked_up()
    assert len(found) >= 20, f"only found {sorted(found)}"
    for expected in ("toggle", "toast", "profiles"):
        assert expected in found, f"{expected} dropped out of the extraction"


def test_every_id_app_js_reaches_for_exists_in_the_document():
    missing = sorted(_looked_up() - _declared())
    assert not missing, (
        f"app.js looks up ids index.html does not define: {missing}. "
        "In bind() that throws out of DOMContentLoaded before the poll is "
        "ever scheduled, and the toast that would report it is inside "
        "refresh(), which never runs."
    )


@pytest.mark.parametrize("control", ("toggle", "profiles", "models", "toast"))
def test_the_controls_bind_runs_on_are_present(control):
    """Named individually because bind() is the unprotected path: these are
    not merely read, they are the ones whose absence stops the app starting."""
    assert control in _declared()
