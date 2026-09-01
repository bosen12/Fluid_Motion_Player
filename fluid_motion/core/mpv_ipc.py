from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

from fluid_motion.core.mpv_detect import candidate_pipes


class IpcError(RuntimeError):
    pass


def as_win_pipe(path: str) -> str:
    """mpv on Windows prefixes the ipc-server string with \\\\.\\pipe\\ as-is."""
    if path.startswith("\\\\.\\pipe\\"):
        return path
    return "\\\\.\\pipe\\" + path


class MpvIpc:
    def __init__(self, handle: Any, kind: str, path: str):
        self._handle = handle
        self.kind = kind
        self.path = path
        self._req = 1
        # mpv pushes unsolicited events down the same connection, so a read very
        # often ends mid-line. Carrying the tail over means the next command
        # reassembles it instead of throwing the fragment away and then failing
        # to parse the fused line that follows.
        self._buf = b""

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        # Both a win32 pipe handle and a socket close the same way; the two
        # branches this used to have were identical.
        try:
            handle.close()
        except OSError:
            pass

    def command(self, *args: Any, timeout: float = 2.5) -> Any:
        payload = {"command": list(args), "request_id": self._req}
        self._req += 1
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._write(raw)
        deadline = time.time() + timeout
        # Backs off from 0.5ms so a prompt mpv still answers in well under a
        # millisecond -- snapshot_playback() issues a dozen of these per tick.
        idle = 0.0005
        while True:
            # Drain first: the reply may already be sitting in the carried-over
            # buffer, in which case there is nothing left to read for it.
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
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
            if time.time() >= deadline:
                raise IpcError("mpv IPC timed out")
            chunk = self._read(4096)
            if not chunk:
                time.sleep(idle)
                idle = min(idle * 2, 0.01)
                continue
            idle = 0.0005
            self._buf += chunk

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
            import win32pipe

            # The handle is synchronous (no FILE_FLAG_OVERLAPPED), so ReadFile
            # blocks until bytes arrive -- which made command()'s deadline
            # unenforceable: an mpv that stopped servicing its IPC (busy
            # compiling a TensorRT engine, wedged vo) parked the caller here
            # forever while holding the apply lock. Peek first and let the
            # caller's deadline do its job.
            try:
                _, avail, _ = win32pipe.PeekNamedPipe(self._handle, 0)
            except pywintypes.error:
                return b""
            if not avail:
                return b""
            try:
                _, data = win32file.ReadFile(self._handle, min(n, avail))
                return data or b""
            except pywintypes.error:
                return b""
        self._handle.settimeout(0.15)
        try:
            return self._handle.recv(n)
        except (TimeoutError, socket.timeout, BlockingIOError, OSError):
            return b""


def _try_create_pipe(path: str) -> MpvIpc | None:
    try:
        import pywintypes
        import win32file
        import win32pipe
    except ImportError:
        return None
    try:
        handle = win32file.CreateFile(
            path,
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


def _open_named_pipe(path: str) -> MpvIpc | None:
    if os.name != "nt":
        return None
    seen: set[str] = set()
    for candidate in (path, as_win_pipe(path)):
        if candidate in seen:
            continue
        seen.add(candidate)
        ipc = _try_create_pipe(candidate)
        if ipc is not None:
            return ipc
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


def connect_pid(
    pid: int,
    extra: list[str] | tuple[str, ...] | None = None,
    *,
    allow_ambiguous: bool = True,
) -> MpvIpc | None:
    extras = extra or ()
    for name in candidate_pipes(pid, extras, allow_ambiguous=allow_ambiguous):
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
    found: list[str] = []
    try:
        for name in os.listdir(r"\\.\pipe\\"):
            if prefix.lower() in name.lower():
                found.append(as_win_pipe(name))
    except OSError:
        return []
    return found
