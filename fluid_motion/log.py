"""Minimal file-based diagnostic log.

This app runs as a tray icon with no console, and the only channel it has ever
had for saying what went wrong is a single line of `_error` text in the UI.
Everything else -- which adapters were found, which backend that resolved to,
what `diagnose()` objected to, which exception the watcher loop swallowed to
stay alive -- existed only in memory and was gone the moment the window
closed.

That was survivable while the only supported GPU was the one on the developer's
desk. It stopped being survivable when v1.6.0 shipped an AMD path that has
never run on AMD hardware and asked users to report back: a report needs
something to report.

Deliberately the same shape as AX Player's `debug_log.py`, including the
rotation policy, so the two projects' logs read alike and neither has to be
learned separately.

Not the stdlib `logging` module: this needs one file, one format and no
configuration surface, and `logging`'s handler/level machinery is a larger
thing to get wrong (a stray `basicConfig` elsewhere silently redirecting it)
than the eight lines it would replace.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime

from fluid_motion.paths import debug_log_path

_log_path = None


def path():
    global _log_path
    if _log_path is None:
        _log_path = debug_log_path()
    return _log_path


# Rotate at this size, keeping one previous generation.
#
# The watcher ticks three times a second forever, so anything that logs per
# tick would bury the one line that mattered -- which is why the call sites
# log state *changes* rather than state. The cap is the backstop for the case
# where something starts changing every tick anyway.
#
# One generation, not many: this is a live diagnostic, not an archive.
MAX_BYTES = 1024 * 1024


def _rotate_if_needed(target) -> None:
    try:
        if target.stat().st_size < MAX_BYTES:
            return
    except OSError:
        return  # not there yet, or unreadable -- either way nothing to rotate
    try:
        os.replace(target, target.with_name(target.name + ".1"))
    except OSError:
        pass


def log(msg: str) -> None:
    try:
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(target)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='milliseconds')} {msg}\n")
    except OSError:
        pass


def log_exc(context: str) -> None:
    log(f"{context}: {traceback.format_exc()}")
