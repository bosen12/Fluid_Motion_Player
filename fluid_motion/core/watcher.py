from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fluid_motion.config import Settings, save_settings
from fluid_motion.core.bootstrap import ensure_input_binding, install_lua
from fluid_motion.core.gpu import snapshot as gpu_snapshot
from fluid_motion.core.inject import apply, fps_fraction_label, interpolation_active, remove, snapshot_playback
from fluid_motion.core.mpv_detect import PlayerProcess, find_mpv_executable, iter_mpv_processes, mpv_root_from
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc, connect_pid
from fluid_motion.core.runtime import diagnose
from fluid_motion.core.vs_script import parse_fps, target_multi


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._ipc: dict[int, MpvIpc] = {}
        self._players: list[PlayerProcess] = []
        self._gpu = gpu_snapshot()
        self._runtime = diagnose(settings.mpv_root)
        self._error = ""
        self._bootstrap_message = ""
        self._bootstrap_progress = 0.0
        self._bootstrapping = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_change: Callable[[], None] | None = None

    def start(self, on_change: Callable[[], None] | None = None) -> None:
        self._on_change = on_change
        try:
            root = Path(self.settings.mpv_root)
            install_lua(root)
            ensure_input_binding(root)
        except OSError as exc:
            self._error = str(exc)
        self._thread = threading.Thread(target=self._loop, name="fluid-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for ipc in self._ipc.values():
                ipc.close()
            self._ipc.clear()

    def _loop(self) -> None:
        while not self._stop.wait(0.9):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 — keep the watcher alive
                self._error = str(exc)
            if self._on_change:
                try:
                    self._on_change()
                except Exception:
                    pass

    def tick(self) -> None:
        players = iter_mpv_processes()
        live: dict[int, MpvIpc] = {}
        with self._lock:
            old = self._ipc
        for player in players:
            ipc = old.get(player.pid)
            if ipc is None:
                ipc = connect_pid(player.pid)
            if ipc is None:
                continue
            try:
                info = snapshot_playback(ipc)
            except IpcError:
                ipc.close()
                continue
            player.pipe = ipc.path
            player.connected = True
            player.media = str(info.get("media") or "")
            player.width = int(info.get("width") or 0)
            player.height = int(info.get("height") or 0)
            player.fps = fps_fraction_label(info.get("fps"))
            player.paused = bool(info.get("paused"))
            player.interpolation = bool(info.get("interpolation"))
            src = parse_fps(info.get("fps"))
            multi, _ = target_multi(self.settings.profile, src, info.get("display_fps"))
            if src:
                try:
                    out = src * (multi if not isinstance(multi, int) else multi)
                    player.estimated_vfps = f"{float(out):.3f}".rstrip("0").rstrip(".")
                except Exception:
                    player.estimated_vfps = ""
            live[player.pid] = ipc
            if self.settings.enabled and self._runtime.ready and not player.interpolation:
                try:
                    apply(ipc, self.settings, Path(self.settings.mpv_root))
                    player.interpolation = True
                except IpcError as exc:
                    self._error = str(exc)
        for pid, ipc in old.items():
            if pid not in live:
                ipc.close()
        with self._lock:
            self._ipc = live
            self._players = players
            self._gpu = gpu_snapshot()
            self._runtime = diagnose(self.settings.mpv_root)

    def set_enabled(self, enabled: bool) -> None:
        self.settings.enabled = bool(enabled)
        save_settings(self.settings)
        with self._lock:
            ipcs = list(self._ipc.items())
        for pid, ipc in ipcs:
            try:
                if enabled:
                    if not self._runtime.ready:
                        self._error = "TensorRT 執行環境尚未就緒"
                        return
                    apply(ipc, self.settings, Path(self.settings.mpv_root))
                else:
                    remove(ipc)
            except IpcError as exc:
                self._error = str(exc)
        self.tick()

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        save_settings(self.settings)
        if self.settings.enabled:
            self.set_enabled(True)

    def state(self) -> dict[str, Any]:
        with self._lock:
            players = [p.to_dict() for p in self._players]
            gpu = self._gpu.to_dict()
            runtime = self._runtime.to_dict()
            error = self._error
            boot = {
                "running": self._bootstrapping,
                "message": self._bootstrap_message,
                "progress": self._bootstrap_progress,
            }
        exe = find_mpv_executable(self.settings.mpv_root)
        return {
            "settings": self.settings.to_dict(),
            "players": players,
            "gpu": gpu,
            "runtime": runtime,
            "error": error,
            "bootstrap": boot,
            "mpv_exe": str(exe) if exe else "",
            "player_count": len(players),
            "connected": sum(1 for p in players if p.get("connected")),
        }

    def start_bootstrap(self) -> None:
        if self._bootstrapping:
            return

        def _run() -> None:
            self._bootstrapping = True
            self._error = ""
            try:
                from fluid_motion.core.bootstrap import install_runtime

                def cb(message: str, ratio: float) -> None:
                    self._bootstrap_message = message
                    self._bootstrap_progress = ratio

                install_runtime(Path(self.settings.mpv_root), cb)
                install_lua(Path(self.settings.mpv_root))
                ensure_input_binding(Path(self.settings.mpv_root))
                self._runtime = diagnose(self.settings.mpv_root)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self._bootstrap_message = f"安裝失敗：{exc}"
            finally:
                self._bootstrapping = False

        threading.Thread(target=_run, name="fluid-bootstrap", daemon=True).start()
