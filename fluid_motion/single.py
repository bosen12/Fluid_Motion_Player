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

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    already = kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
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
