from __future__ import annotations

import os
import subprocess
import threading
from typing import Any


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

# Every child this app has running, so quit_app can take them with it.
#
# quit_app ends in os._exit(0) -- it has to, because pywebview and pystray
# otherwise leave a headless process with no tray icon behind. os._exit runs
# no finally blocks and joins no threads, and on Windows a child does not die
# with its parent. So quitting during an install left 7z extracting a 2.6 GB
# archive into %APPDATA%\FluidMotion with nothing watching it: reproduced on
# this exact code path, parent gone, child still running.
#
# The consequence is not just wasted disk. install_runtime's `_bootstrapping`
# guard -- whose comment reasons carefully about "a second 3.5 GB download
# into the same cache" extracting "over the first" -- is a per-process flag.
# It cannot see a 7z left over from the *previous* session, so the next launch
# happily starts a second extraction into the directory the orphan is still
# writing.
_live: set[subprocess.Popen] = set()
_live_lock = threading.Lock()


def terminate_children() -> int:
    """Kill anything still running, and say how many. Called before os._exit."""
    with _live_lock:
        children = list(_live)
    killed = 0
    for child in children:
        try:
            if child.poll() is None:
                child.kill()
                killed += 1
        except OSError:
            pass
    for child in children:
        try:
            child.wait(timeout=5)
        except (subprocess.SubprocessError, OSError):
            pass
    return killed


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

    # subprocess.run's own body, so the Popen can be registered while it runs.
    # run() keeps its handle to itself, and there is no way to reach the child
    # of a call that is still blocked in it.
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    with subprocess.Popen(args, **kwargs) as proc:
        with _live_lock:
            _live.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(args, timeout, stdout, stderr)
        except BaseException:
            proc.kill()
            raise
        finally:
            with _live_lock:
                _live.discard(proc)
    result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
