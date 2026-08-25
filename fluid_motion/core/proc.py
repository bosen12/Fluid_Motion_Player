from __future__ import annotations

import os
import subprocess
from typing import Any


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def hidden_run_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name != "nt":
        return kwargs
    kwargs["creationflags"] = CREATE_NO_WINDOW
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    kwargs["startupinfo"] = startup
    return kwargs


def run_hidden(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    kwargs.setdefault("capture_output", True)
    # Explicit UTF-8 rather than text=True's locale-dependent decode. Python's
    # default text mode follows the system ANSI codepage (cp950 on a zh-TW
    # machine, and similar elsewhere), so the first non-ASCII byte a child
    # writes -- an accented GPU name from nvidia-smi, a path in 7z output --
    # raises UnicodeDecodeError inside subprocess's own reader thread. That
    # surfaces nowhere in a windowed build: the call just fails and whatever
    # depended on it silently stops working.
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    kwargs.setdefault("check", False)
    kwargs.update(hidden_run_kwargs())
    return subprocess.run(args, **kwargs)
