from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

from fluid_motion.core.mpv_detect import candidate_pipes


class IpcError(RuntimeError):
    pass


class MpvIpc:
    def __init__(self, handle: Any, kind: str, path: str):
        self._handle = handle
        self.kind = kind
        self.path = path
        self._req = 1

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if self.kind == "pipe":
                handle.close()
            else:
                handle.close()
        except OSError:
            pass

    def command(self, *args: Any, timeout: float = 2.5) -> Any:
        payload = {"command": list(args), "request_id": self._req}
        self._req += 1
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._write(raw)
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = self._read(4096)
            if not chunk:
                time.sleep(0.02)
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if msg.get("request_id") == payload["request_id"]:
                    if msg.get("error") not in (None, "success"):
                        raise IpcError(str(msg.get("error")))
                    return msg.get("data")
        raise IpcError("mpv IPC timed out")

    def get(self, name: str) -> Any:
        return self.command("get_property", name)

    def set(self, name: str, value: Any) -> Any:
        return self.command("set_property", name, value)

    def _write(self, data: bytes) -> None:
        if self._handle is None:
            raise IpcError("IPC closed")
        if self.kind == "pipe":
            import pywintypes
            import win32file

            try:
                win32file.WriteFile(self._handle, data)
            except pywintypes.error as exc:  # pragma: no cover
                raise IpcError(str(exc)) from exc
        else:
            self._handle.sendall(data)

    def _read(self, n: int) -> bytes:
        if self._handle is None:
            return b""
        if self.kind == "pipe":
            import pywintypes
            import win32file

            try:
                _, data = win32file.ReadFile(self._handle, n)
                return data or b""
            except pywintypes.error:
                return b""
        self._handle.settimeout(0.15)
        try:
            return self._handle.recv(n)
        except (TimeoutError, socket.timeout, BlockingIOError, OSError):
            return b""


def _open_named_pipe(path: str) -> MpvIpc | None:
    if os.name != "nt":
        return None
    try:
        import pywintypes
        import win32file
        import win32pipe
    except ImportError:
        return None
    pipe = path
    if not pipe.startswith("\\\\.\\pipe\\") and not os.path.exists(pipe):
        # mpvSockets stores a filesystem-looking name; try as pipe too
        pipe = r"\\.\pipe\\" + path.replace("\\", "/").replace(":", "")
    try:
        handle = win32file.CreateFile(
            path if path.startswith("\\\\.\\pipe\\") or os.path.exists(path) else pipe,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None,
        )
        win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_BYTE, None, None)
        return MpvIpc(handle, "pipe", path)
    except (OSError, pywintypes.error):
        return None


def _open_socket(path: str) -> MpvIpc | None:
    if not path or path.startswith("\\\\.\\pipe\\"):
        return None
    if not os.path.exists(path):
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.6)
    try:
        sock.connect(path)
    except OSError:
        sock.close()
        return None
    sock.setblocking(False)
    return MpvIpc(sock, "unix", path)


def connect_pid(pid: int, extra: list[str] | tuple[str, ...] | None = None) -> MpvIpc | None:
    extras = extra or ()
    for name in candidate_pipes(pid, extras):
        ipc = _open_named_pipe(name) if os.name == "nt" else None
        if ipc is None:
            ipc = _open_socket(name)
        if ipc is not None:
            try:
                ipc.get("mpv-version")
            except IpcError:
                ipc.close()
                continue
            return ipc
    return None


def enumerate_windows_pipes(prefix: str = "fluid-mpv-") -> list[str]:
    if os.name != "nt":
        return []
    try:
        import win32file
    except ImportError:
        return []
    found: list[str] = []
    try:
        for entry in win32file.FindFiles(r"\\.\pipe\*"):
            name = entry[8] if len(entry) > 8 else ""
            if isinstance(name, str) and prefix in name.lower():
                found.append(rf"\\.\pipe\{name}")
    except OSError:
        return []
    return found
