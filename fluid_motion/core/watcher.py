from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fluid_motion.config import Settings, save_settings
from fluid_motion.paths import heartbeat_path, hotkey_path, seek_hold_path
from fluid_motion.core.bootstrap import ensure_input_binding, install_lua
from fluid_motion.core.engine_cache import info as engine_cache_info, next_growth_deadline
from fluid_motion.core.gpu import flicker_risk, snapshot as gpu_snapshot
from fluid_motion.core.inject import (
    SETTLE_SECONDS,
    apply,
    interpolation_held_off,
    is_settling,
    live_fps_label,
    live_source_fps,
    measured_output_fps,
    output_shortfall,
    remove,
    snapshot_playback,
)
from fluid_motion.core.mpv_detect import PlayerProcess, find_mpv_executable, iter_mpv_processes, mpv_root_from
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc, connect_pid
from fluid_motion.core.runtime import diagnose
from fluid_motion.core.vs_script import parse_fps, target_multi

ENGINE_GROWTH_GRACE = 3.0


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
        self._settling_until = 0.0
        self._engine_cache = engine_cache_info()
        self._engine_bytes = self._engine_cache.total_bytes
        self._engine_growing_until = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._on_change: Callable[[], None] | None = None
        self._on_show: Callable[[], None] | None = None

    def start(
        self,
        on_change: Callable[[], None] | None = None,
        on_show: Callable[[], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._on_show = on_show
        try:
            root = Path(self.settings.mpv_root)
            install_lua(root)
            ensure_input_binding(root)
        except OSError as exc:
            self._error = str(exc)
        self._thread = threading.Thread(target=self._loop, name="fluid-watcher", daemon=True)
        self._thread.start()
        # Own thread: a slow vf apply (mpv rebuilding the VS/TensorRT pipeline
        # can take well over the Lua side's 4s alive window) must not stall
        # this, or mpv wrongly reports Fluid Motion as not running mid-apply.
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="fluid-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        self._write_heartbeat()

    def set_on_show(self, on_show: Callable[[], None] | None) -> None:
        self._on_show = on_show

    def stop(self) -> None:
        self._stop.set()
        try:
            heartbeat_path().unlink(missing_ok=True)
        except OSError:
            pass
        with self._lock:
            ipcs = list(self._ipc.values())
            self._ipc.clear()
        for ipc in ipcs:
            try:
                remove(ipc)
            except IpcError:
                pass
            ipc.close()

    def _write_heartbeat(self) -> None:
        try:
            heartbeat_path().write_text(str(time.time()), encoding="utf-8")
        except OSError:
            pass

    def _seek_hold_age(self) -> float | None:
        try:
            return time.time() - seek_hold_path().stat().st_mtime
        except OSError:
            return None

    def _held_off(self, seeking: bool = False) -> bool:
        return interpolation_held_off(seeking, self._seek_hold_age())

    def _mark_settling(self) -> None:
        self._settling_until = time.monotonic() + SETTLE_SECONDS

    def _consume_hotkey(self) -> None:
        path = hotkey_path()
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip().lower()
            path.unlink(missing_ok=True)
        except OSError:
            return
        if text in {"toggle", "on", "off"}:
            if text == "on":
                want = True
            elif text == "off":
                want = False
            else:
                want = not self.settings.enabled
            self.set_enabled(want)
        elif text == "show" and self._on_show:
            try:
                self._on_show()
            except Exception:
                pass

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(1.0):
            self._write_heartbeat()

    def _loop(self) -> None:
        while not self._stop.wait(0.3):
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
        self._write_heartbeat()
        self._consume_hotkey()
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
            player.paused = bool(info.get("paused"))
            player.interpolation = bool(info.get("interpolation"))
            src = live_source_fps(
                info.get("container_fps") or info.get("fps"),
                info.get("estimated_vfps"),
                player.interpolation,
            )
            player.fps = live_fps_label(src)
            multi, _ = target_multi(self.settings.profile, parse_fps(src), info.get("display_fps"))
            target = None
            if src:
                try:
                    target = float(src) * float(multi)
                    player.target_fps = live_fps_label(target)
                except Exception:
                    player.target_fps = ""
            else:
                player.target_fps = ""
            measured = measured_output_fps(
                info.get("display_fps"),
                info.get("vsync_ratio"),
                info.get("estimated_display_fps"),
            )
            short = output_shortfall(
                measured,
                target,
                paused=player.paused,
                interpolating=player.interpolation,
            )
            player.settling = is_settling(short, time.monotonic(), self._settling_until)
            player.fps_ok = not short or player.settling
            if measured:
                label = live_fps_label(measured)
                player.output_fps = label
                player.estimated_vfps = label
            elif target and (player.interpolation or self.settings.enabled):
                player.output_fps = ""
                player.estimated_vfps = ""
            else:
                player.output_fps = ""
                player.estimated_vfps = ""
            live[player.pid] = ipc
            held = self._held_off(bool(info.get("seeking")))
            if held:
                if player.interpolation:
                    try:
                        remove(ipc)
                        player.interpolation = False
                        self._error = ""
                    except IpcError as exc:
                        self._error = str(exc)
            elif self.settings.enabled and self._runtime.ready:
                if not player.interpolation:
                    try:
                        apply(ipc, self.settings, Path(self.settings.mpv_root))
                        player.interpolation = True
                        self._mark_settling()
                        self._error = ""
                    except IpcError as exc:
                        self._error = str(exc)
            elif player.interpolation:
                try:
                    remove(ipc)
                    player.interpolation = False
                    self._error = ""
                except IpcError as exc:
                    self._error = str(exc)
        for pid, ipc in old.items():
            if pid not in live:
                ipc.close()
        cache = engine_cache_info()
        now = time.monotonic()
        self._engine_growing_until = next_growth_deadline(
            cache.total_bytes, self._engine_bytes, now, self._engine_growing_until, ENGINE_GROWTH_GRACE
        )
        self._engine_bytes = cache.total_bytes
        with self._lock:
            self._ipc = live
            self._players = players
            self._gpu = gpu_snapshot()
            self._runtime = diagnose(self.settings.mpv_root)
            self._engine_cache = cache

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
                    if self._held_off():
                        self._error = ""
                        continue
                    apply(ipc, self.settings, Path(self.settings.mpv_root), announce=True)
                    self._mark_settling()
                    self._error = ""
                else:
                    remove(ipc, announce=True)
                    self._error = ""
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
            engine_cache = self._engine_cache.to_dict()
            engine_growing_until = self._engine_growing_until
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
            "gpu_safe_mode": flicker_risk(gpu.get("name", "")) and not self.settings.force_accel,
            "runtime": runtime,
            "error": error,
            "bootstrap": boot,
            "mpv_exe": str(exe) if exe else "",
            "player_count": len(players),
            "connected": sum(1 for p in players if p.get("connected")),
            "engine_cache": engine_cache,
            "engine_compiling": time.monotonic() < engine_growing_until,
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
