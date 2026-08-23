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
    kwargs.setdefault("text", True)
    kwargs.setdefault("check", False)
    kwargs.update(hidden_run_kwargs())
    return subprocess.run(args, **kwargs)
