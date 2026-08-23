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
    resolve_multi,
    snapshot_playback,
)
from fluid_motion.core.mpv_detect import PlayerProcess, find_mpv_executable, iter_mpv_processes, mpv_root_from
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc, connect_pid
from fluid_motion.core.runtime import diagnose
from fluid_motion.core.vs_script import parse_fps, target_multi

ENGINE_GROWTH_GRACE = 3.0
# A failed apply() is retried by the next tick, but not at the 0.3s tick rate:
# an mpv that is refusing vf commands would otherwise be hammered forever.
APPLY_RETRY_BACKOFF = 2.0
# How long a UI-thread settings change waits for the background tick to finish
# an apply before handing the work off to the next tick instead of blocking.
UI_APPLY_WAIT = 0.5


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
        # Guards apply()/remove() specifically: tick() (background thread)
        # and set_enabled()/update_settings() (pywebview bridge thread) both
        # decide independently whether to re-apply, and a slow apply() (mpv
        # rebuilding its pipeline) leaves a multi-second window where the two
        # can race and stomp on each other's vf remove+add / .vpy write.
        self._apply_lock = threading.Lock()
        # The settings that actually reached each mpv. player.interpolation
        # only says *a* filter is loaded, never whether it matches the current
        # settings -- so a change that got skipped (seek hold-off) or failed
        # (IpcError) used to leave mpv running the old filter forever, with
        # nothing left to notice the mismatch. Comparing against this snapshot
        # is what makes the next tick pick the change back up.
        self._applied: dict[int, tuple] = {}
        self._retry_at: dict[int, float] = {}
        self._in_hotkey = False
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
            ipcs = list(self._ipc.items())
            self._ipc.clear()
        for pid, ipc in ipcs:
            # Bounded wait: shutdown must not hang behind an in-flight apply.
            self._remove_from(pid, ipc, wait=2.0)
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

    def _filter_key(self, multi: int) -> tuple:
        """Everything that changes the generated .vpy, plus the resolved multi.

        multi is part of the key so that switching to a file with a different
        source fps re-applies too -- mpv keeps the vf across file loads, and the
        old multiplier is wrong for the new source.
        """
        cfg = self.settings
        return (
            cfg.profile,
            int(cfg.rife_model),
            float(cfg.scene_threshold),
            int(cfg.trt_streams),
            bool(cfg.fp16),
            bool(cfg.cuda_graph),
            bool(cfg.force_accel),
            str(cfg.mpv_root),
            int(multi),
        )

    def _invalidate(self, pid: int) -> None:
        """Forget what was applied, so the next tick re-applies from scratch."""
        with self._lock:
            self._applied.pop(pid, None)

    def _apply_to(
        self, pid: int, ipc: MpvIpc, multi: int, *, announce: bool = False, wait: float = -1
    ) -> bool:
        """Apply the current settings to one mpv and record what was applied.

        False means mpv did not get them -- it refused (IpcError, retried after
        a backoff), or another thread is mid-apply and this caller chose not to
        queue behind it. Either way the pid is left invalidated, so tick()
        reconciles it. wait=-1 blocks; a UI-thread caller passes a short wait so
        a slow apply cannot freeze the window.
        """
        key = self._filter_key(multi)
        if not self._apply_lock.acquire(timeout=wait):
            self._invalidate(pid)
            return False
        try:
            apply(ipc, self.settings, Path(self.settings.mpv_root), announce=announce)
        except IpcError as exc:
            self._error = str(exc)
            with self._lock:
                self._applied.pop(pid, None)
                self._retry_at[pid] = time.monotonic() + APPLY_RETRY_BACKOFF
            return False
        finally:
            self._apply_lock.release()
        with self._lock:
            self._applied[pid] = key
            self._retry_at.pop(pid, None)
        self._mark_settling()
        self._error = ""
        return True

    def _remove_from(self, pid: int, ipc: MpvIpc, *, announce: bool = False, wait: float = -1) -> bool:
        if not self._apply_lock.acquire(timeout=wait):
            self._invalidate(pid)
            return False
        try:
            remove(ipc, announce=announce)
        except IpcError as exc:
            self._error = str(exc)
            return False
        finally:
            self._apply_lock.release()
        self._invalidate(pid)
        self._error = ""
        return True

    def _consume_hotkey(self) -> None:
        # set_enabled() ends with tick(), which calls back in here. The hotkey
        # file is unlinked before dispatch so that terminates, but a hotkey
        # landing inside that window would nest another level; guard instead of
        # relying on the timing.
        if self._in_hotkey:
            return
        path = hotkey_path()
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip().lower()
            path.unlink(missing_ok=True)
        except OSError:
            return
        self._in_hotkey = True
        try:
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
        finally:
            self._in_hotkey = False

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

    def tick(self, *, apply_wait: float = -1) -> None:
        """apply_wait is how long to queue behind an in-flight apply.

        The background loop blocks (-1) because it *is* the worker. A tick
        driven from the pywebview bridge thread passes a short wait so a slow
        apply cannot freeze the window; whatever it skips stays invalidated and
        the next background tick finishes it.
        """
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
            want_multi = resolve_multi(info, self.settings)
            with self._lock:
                applied = self._applied.get(player.pid)
                retry_at = self._retry_at.get(player.pid, 0.0)
            if held:
                # The hold-off drops the filter but must not count as a settled
                # state: forget the snapshot so the settings are re-applied once
                # the seek quiets down, even if nothing else changes afterwards.
                if player.interpolation:
                    if self._remove_from(player.pid, ipc, wait=apply_wait):
                        player.interpolation = False
                else:
                    self._invalidate(player.pid)
            elif self.settings.enabled and self._runtime.ready:
                # Stale settings, not just a missing filter. multi <= 1 is a
                # legitimately applied no-op (nothing left to interpolate), so it
                # must not read as "filter missing" and re-apply every tick.
                stale = applied != self._filter_key(want_multi)
                missing = want_multi > 1 and not player.interpolation
                if (stale or missing) and time.monotonic() >= retry_at:
                    if self._apply_to(player.pid, ipc, want_multi, wait=apply_wait):
                        player.interpolation = want_multi > 1
            elif player.interpolation or applied is not None:
                if self._remove_from(player.pid, ipc, wait=apply_wait):
                    player.interpolation = False
        for pid, ipc in old.items():
            if pid not in live:
                ipc.close()
        with self._lock:
            for pid in [p for p in self._applied if p not in live]:
                self._applied.pop(pid, None)
            for pid in [p for p in self._retry_at if p not in live]:
                self._retry_at.pop(pid, None)
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
            if enabled:
                if not self._runtime.ready:
                    self._error = "TensorRT 執行環境尚未就緒"
                    return
                # Invalidate first: whether the apply below is skipped (hold-off),
                # fails, or never runs at all, tick() has to be left able to see
                # that this mpv is out of date and finish the job later.
                self._invalidate(pid)
                if self._held_off():
                    self._error = ""
                    continue
                try:
                    info = snapshot_playback(ipc)
                except IpcError as exc:
                    self._error = str(exc)
                    continue
                # Short wait: set_enabled() runs on the pywebview bridge thread,
                # so blocking here for a multi-second apply freezes the window.
                # Handing off to tick() costs at most one 0.3s tick.
                self._apply_to(
                    pid, ipc, resolve_multi(info, self.settings), announce=True, wait=UI_APPLY_WAIT
                )
            else:
                self._remove_from(pid, ipc, announce=True, wait=UI_APPLY_WAIT)
        self.tick(apply_wait=UI_APPLY_WAIT)

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        if self.settings.enabled:
            self.set_enabled(True)  # saves; no need to write the file twice
        else:
            save_settings(self.settings)

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
