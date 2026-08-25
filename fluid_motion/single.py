from __future__ import annotations

import os
from typing import Any

from fluid_motion.paths import hotkey_path

_MUTEX_HANDLE: Any = None
_MUTEX_NAME = "Local\\FluidMotion.Singleton"


def try_acquire() -> bool:
    """True if this process owns the singleton. False if another copy is already running."""
    global _MUTEX_HANDLE
    if os.name != "nt":
        return True
    import ctypes

    # use_last_error, rather than calling kernel32.GetLastError() as a second
    # foreign call: ctypes makes no promise about what happens to the thread's
    # last-error value between two calls, and losing ERROR_ALREADY_EXISTS here
    # means a second copy starts and both drive the same mpv over IPC,
    # applying and removing the filter against each other.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    already = ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
    if not handle:
        return True
    if already:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return False
    _MUTEX_HANDLE = handle
    return True


def notify_running(action: str = "show") -> None:
    hotkey_path().write_text(action.strip() or "show", encoding="utf-8")


def handover_or_continue() -> bool:
    """If another instance is alive, ask it to show and exit. Return True to keep running."""
    if try_acquire():
        return True
    try:
        notify_running("show")
    except OSError:
        pass
    return False


def exit_duplicate() -> None:
    raise SystemExit(0)
