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
from fluid_motion.core.gpu import (
    available_vendors,
    detect_adapters,
    flicker_risk,
    snapshot as gpu_snapshot,
)
from fluid_motion.core.inject import (
    SEEK_HOLD_MAX_AGE,
    SETTLE_SECONDS,
    adopt_hwdec,
    apply,
    flush_hwdec_state,
    forget_hwdec,
    interpolation_active,
    hwdec_pids,
    interpolation_held_off,
    is_settling,
    stranded_hwdec,
    drop_ratio,
    filtered_output_fps,
    live_fps_label,
    live_source_fps,
    output_shortfall,
    playback_is_clean,
    player_config_dir,
    rate_snapshot,
    realtime_label,
    realtime_ratio,
    remove,
    resolve_multi,
    snapshot_playback,
)
from fluid_motion.core.mpv_detect import PlayerProcess, find_mpv_executable, iter_mpv_processes, mpv_root_from
from fluid_motion.core.mpv_ipc import IpcError, MpvIpc, connect_pid
from fluid_motion.core.runtime import RuntimeStatus, diagnose, missing_labels
from fluid_motion.core.vs_script import (
    effective_backend,
    parse_fps,
    resolve_backend,
    target_multi,
)

ENGINE_GROWTH_GRACE = 3.0
# How often the housekeeping half of tick() runs -- the TensorRT engine-cache
# scan, the runtime diagnosis and the nvidia-smi snapshot. None of it depends
# on the tick rate: at 0.3s an app sitting in the tray with no player open was
# walking two directory trees three times a second and spawning nvidia-smi
# every 1.5s (its own cache TTL), forever. The active figure still refreshes
# the readouts faster than anyone reads them.
HOUSEKEEPING_ACTIVE = 1.0
HOUSEKEEPING_IDLE = 5.0
# A failed apply() is retried by the next tick, but not at the 0.3s tick rate:
# an mpv that is refusing vf commands would otherwise be hammered forever.
APPLY_RETRY_BACKOFF = 2.0

# The hotkey file is polled far faster than tick() runs. Noticing an F3 press
# is a single stat() -- it does not need to wait behind tick()'s IPC round
# trips and process enumeration, which is what used to put up to 300ms in
# front of every toggle before any actual work started.
HOTKEY_POLL = 0.05

# Reconcile interval.
TICK_SECONDS = 0.3


def is_disconnect_error(message: str) -> bool:
    """True when mpv closed the named pipe; not a filter-setup failure."""
    text = str(message)
    lowered = text.lower()
    if "管道正關閉" in text or "broken pipe" in lowered:
        return True
    if "writefile" in lowered and "232" in text:
        return True
    return "ipc closed" in lowered


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
        self._runtime = diagnose(settings.mpv_root, backend=self._backend())
        self._error = ""
        self._bootstrap_message = ""
        self._bootstrap_progress = 0.0
        self._bootstrapping = False
        self._settling_until = 0.0
        self._engine_cache = engine_cache_info()
        self._engine_bytes = self._engine_cache.total_bytes
        self._engine_growing_until = 0.0
        self._housekeeping_at = 0.0
        # Readiness of each *player's own* config dir, which is the directory
        # the filter is actually written into and loaded from -- not
        # settings.mpv_root, which is only where the UI's panel looks. Cached
        # by directory (two players can share one) and refreshed with the rest
        # of the housekeeping; diagnose() measures 0.33ms, so the cache is
        # about not doing it three times a second rather than about cost.
        self._runtime_cache: dict[tuple[str, str], RuntimeStatus] = {}
        # config-dir per pid. mpv answers this over IPC and it cannot change
        # while the process lives, so it is asked once rather than every tick.
        self._config_dirs: dict[int, Path] = {}
        self._player_ready: dict[int, tuple[bool, str]] = {}
        # Config dirs whose IPC script has been checked/installed this session.
        self._scripted: set[str] = set()
        self._needs_restart: set[int] = set()
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
        # Per-pid (time-pos, drop-count, monotonic) of the last usable sample,
        # plus the smoothed ratios built from them. Both speed and drop rate
        # can only be had by differencing two observations, so they have to be
        # carried across ticks.
        self._realtime_seen: dict[int, tuple[float, float, float]] = {}
        self._realtime: dict[int, float] = {}
        self._drops: dict[int, float] = {}
        self._in_hotkey = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._hotkey_thread: threading.Thread | None = None
        self._on_change: Callable[[], None] | None = None
        self._on_show: Callable[[], None] | None = None

    def start(
        self,
        on_change: Callable[[], None] | None = None,
        on_show: Callable[[], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._on_show = on_show
        self._sweep_seek_holds()
        self._recover_stranded_hwdec()
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
        # Likewise its own thread, and the sole consumer of the hotkey file:
        # polling it here rather than from tick() is what makes F3 feel
        # immediate, and a single consumer means two threads can't both read
        # the same press before either unlinks it.
        self._hotkey_thread = threading.Thread(target=self._hotkey_loop, name="fluid-hotkey", daemon=True)
        self._hotkey_thread.start()
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

    def _sweep_seek_holds(self) -> None:
        """Once at start-up: drop the old shared file and any orphaned holds.

        The lua removes its own on resume and on shutdown, but an mpv killed
        mid-seek cannot. Nothing reads a stale one -- they age out past
        SEEK_HOLD_MAX_AGE within two seconds -- so this is only about not
        accumulating a file per pid ever seen.
        """
        try:
            seek_hold_path().unlink(missing_ok=True)
        except OSError:
            pass
        cutoff = time.time() - SEEK_HOLD_MAX_AGE
        try:
            for stale in seek_hold_path().parent.glob("seek_hold-*"):
                try:
                    # Only ones nothing could still be relying on. Launching
                    # while a player is mid-drag would otherwise delete the
                    # hold it wrote moments ago, and the first tick could then
                    # apply the filter into the middle of the seek -- which is
                    # what the file is there to prevent. Anything past
                    # SEEK_HOLD_MAX_AGE is already ignored by every reader.
                    if stale.stat().st_mtime > cutoff:
                        continue
                    stale.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def _recover_stranded_hwdec(self) -> None:
        """Undo a previous run's copy-back switch on players still running.

        Only reachable when the last run did not get to remove() -- killed,
        crashed, or the machine went down mid-playback. The lua takes the
        stale filter off by itself once this app stops answering its
        heartbeat, but it has no way to know what hwdec was before, so the
        player would otherwise stay on copy-back for the rest of its life.

        Skipped for any player that still has our filter loaded: that one is
        legitimately on copy-back and about to be adopted by this run.
        """
        stranded = stranded_hwdec()
        if not stranded:
            return
        for pid, previous in stranded.items():
            ipc = connect_pid(pid)
            if ipc is None:
                # Not answering. Keep the record rather than discarding the
                # only copy of the mode -- it gets another chance next run,
                # and stranded_hwdec() drops it once the process is gone.
                adopt_hwdec(pid, previous)
                continue
            try:
                if interpolation_active(ipc):
                    # Still filtered, so copy-back is still needed. Carry the
                    # value forward instead of restoring it now, or remove()
                    # would later find nothing to put back.
                    adopt_hwdec(pid, previous)
                    continue
                ipc.set("hwdec", previous)
            except IpcError:
                adopt_hwdec(pid, previous)
            finally:
                ipc.close()
        # One write, from the map, now that every record has been decided:
        # what was restored is absent from it and what was kept is in it.
        flush_hwdec_state()

    def _seek_hold_age(self, pid: int) -> float | None:
        """Age of this player's hold, falling back to the pre-v1.4.5 name.

        install_lua only rewrites the script on disk. An mpv that was already
        running has the old one in memory -- mpv loads scripts at launch --
        and that one writes the shared `seek_hold`. Without this fallback the
        per-pid file simply never appears for such a player, _held_off drops
        to the bare `seeking` property, and the post-seek debounce is gone:
        the tick after mpv clears `seeking` re-applies the filter, so dragging
        the seek bar rebuilds the VapourSynth/TensorRT pipeline between drags.

        The shared file cannot say which player wrote it, so while one is
        still on the old script its seeks do hold off the others -- exactly
        the behaviour this release set out to fix. That is the pre-v1.4.5
        behaviour for that player either way, and it ends when it restarts.
        """
        for path in (seek_hold_path(pid), seek_hold_path()):
            try:
                return time.time() - path.stat().st_mtime
            except OSError:
                continue
        return None

    def _held_off(self, pid: int, seeking: bool = False) -> bool:
        return interpolation_held_off(seeking, self._seek_hold_age(pid))

    def _player_root(self, pid: int, ipc: MpvIpc) -> Path:
        """The config dir this player will load the filter from.

        Asked of mpv itself, because one configured mpv_root cannot answer it
        for everyone: an embedded host runs out of its own runtime folder and
        a standalone mpv lives wherever it was installed. Falls back to the
        setting only when mpv declines to answer.
        """
        cached = self._config_dirs.get(pid)
        if cached is not None:
            return cached
        root = player_config_dir(ipc) or Path(self.settings.mpv_root)
        self._config_dirs[pid] = root
        return root

    def _backend(self) -> str:
        """The backend that will actually be generated for this machine."""
        return resolve_backend(self.settings.backend, available_vendors())

    def _runtime_for(self, root: Path) -> RuntimeStatus:
        # Keyed by backend as well as directory: what "ready" requires differs
        # between them (TensorRT+CUDA vs vsncnn.dll), so a cache keyed on the
        # path alone would keep answering with the old backend's verdict after
        # the setting changed.
        backend = self._backend()
        key = (str(root), backend)
        status = self._runtime_cache.get(key)
        if status is None:
            status = diagnose(root, backend=backend)
            self._runtime_cache[key] = status
        return status

    def _ensure_player_scripts(self, root: Path, pid: int) -> None:
        """Put the IPC script in this player's own config dir.

        start() installs it into settings.mpv_root and nothing else ever did,
        so a player running out of any other directory never got it: no F3
        binding (AX Player's own toolbar button sends F3 and was therefore a
        silent no-op -- confirmed against a live instance, which reported no
        F3 binding at all), no post-seek hold-off, and no stale-filter
        cleanup when this app goes away.

        Only ever adds a missing file; an existing one is left alone. mpv
        loads scripts at launch, so a player that is already running has to be
        restarted before it takes effect -- which is what needs_restart says.
        """
        key = str(root)
        if key in self._scripted:
            return
        self._scripted.add(key)
        if (root / "scripts" / "zz-fluid-ipc.lua").is_file():
            return
        try:
            install_lua(root)
            ensure_input_binding(root)
        except OSError as exc:
            self._error = str(exc)
            return
        self._needs_restart.add(pid)

    def _mark_settling(self) -> None:
        self._settling_until = time.monotonic() + SETTLE_SECONDS

    def _update_realtime(self, pid: int, info: dict[str, Any], target_fps: float | None) -> None:
        """Difference this tick's samples against the last to get speed and drops.

        Paused and seeking players are dropped rather than sampled: neither
        advances time-pos for a reason that says anything about whether the
        pipeline can keep up, and feeding either in would read as a stall.

        Drops are tracked alongside because realtime alone does not mean
        smooth -- mpv can hold the clock by discarding frames, and that reads
        as a perfectly healthy 1.00 while the picture stutters.
        """
        pos = info.get("time_pos")
        if info.get("paused") or info.get("seeking") or pos is None:
            self._realtime_seen.pop(pid, None)
            return
        now = time.monotonic()
        drops = info.get("drops")
        drops = float(drops) if isinstance(drops, (int, float)) else 0.0
        last = self._realtime_seen.get(pid)
        self._realtime_seen[pid] = (float(pos), drops, now)
        if last is None:
            return
        last_pos, last_drops, last_at = last
        elapsed = now - last_at
        ratio = realtime_ratio(
            float(pos) - last_pos,
            elapsed,
            speed=float(info.get("speed") or 1.0),
            previous=self._realtime.get(pid),
        )
        if ratio is not None:
            self._realtime[pid] = ratio
        dropped = drop_ratio(
            drops - last_drops, elapsed, target_fps, previous=self._drops.get(pid)
        )
        if dropped is not None:
            self._drops[pid] = dropped

    def _filter_key(self, multi: int) -> tuple:
        """Everything that changes the generated .vpy, plus the resolved multi.

        multi is part of the key so that switching to a file with a different
        source fps re-applies too -- mpv keeps the vf across file loads, and the
        old multiplier is wrong for the new source.
        """
        cfg = self.settings
        streams, graph, _ = effective_backend(
            self._gpu.name,
            trt_streams=cfg.trt_streams,
            cuda_graph=cfg.cuda_graph,
            force_accel=cfg.force_accel,
        )
        return (
            cfg.profile,
            int(cfg.rife_model),
            float(cfg.scene_threshold),
            int(streams),
            bool(cfg.fp16),
            bool(graph),
            bool(cfg.force_accel),
            str(cfg.mpv_root),
            # The resolved backend, not cfg.backend: "auto" can answer
            # differently without the setting moving at all, and it decides
            # which Backend() the .vpy constructs -- the largest difference
            # this key can describe. Switching through the UI happens to
            # re-apply anyway (set_enabled invalidates first), which is exactly
            # why leaving it out looked harmless; a change arriving any other
            # way was silently kept, with mpv still running the old backend
            # while the panel reported the new one.
            #
            # Before multi, not after: multi is last by convention and a test
            # reads it as key[-1].
            self._backend(),
            int(multi),
        )

    def _invalidate(self, pid: int) -> None:
        """Forget what was applied, so the next tick re-applies from scratch."""
        with self._lock:
            self._applied.pop(pid, None)

    def _apply_to(
        self,
        pid: int,
        ipc: MpvIpc,
        multi: int,
        *,
        announce: bool = False,
        wait: float = -1,
        info: dict[str, Any] | None = None,
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
            # This player's own config dir, not the globally configured root:
            # the .vpy is only loadable by the mpv that will read it, and a
            # single setting cannot be right for an embedded host and a
            # standalone mpv at the same time. Falls back to the setting when
            # mpv declines to answer.
            root = player_config_dir(ipc) or Path(self.settings.mpv_root)
            apply(
                ipc, self.settings, root,
                announce=announce, info=info, pid=pid, backend=self._backend(),
            )
        except IpcError as exc:
            gone = is_disconnect_error(str(exc))
            if not gone:
                self._error = str(exc)
            with self._lock:
                self._applied.pop(pid, None)
                if not gone:
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
            remove(ipc, announce=announce, pid=pid)
        except IpcError as exc:
            if not is_disconnect_error(str(exc)):
                self._error = str(exc)
            return False
        finally:
            self._apply_lock.release()
        self._invalidate(pid)
        self._error = ""
        return True

    def _consume_hotkey(self) -> None:
        # Only _hotkey_loop calls this, so presses are dispatched one at a
        # time and the file has a single consumer. The guard stays because
        # set_enabled() ends with tick(), and a dispatch that ends up back
        # here would otherwise nest.
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

    def _hotkey_loop(self) -> None:
        while not self._stop.wait(HOTKEY_POLL):
            try:
                self._consume_hotkey()
            except Exception as exc:  # noqa: BLE001 — keep the poller alive
                self._error = str(exc)

    def _loop(self) -> None:
        # Waking this early when the seek hold clears was tried, on the
        # reasoning that the filter sits off for up to a tick longer than it
        # needs to. Measured against the plain cadence it did not help: median
        # time for interpolation to return went 0.71s -> 0.84s and the spread
        # widened, because the wait it removes is small next to the variance
        # in rebuilding the VapourSynth script. Not worth the extra signal.
        while not self._stop.wait(TICK_SECONDS):
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
        players = iter_mpv_processes()
        live: dict[int, MpvIpc] = {}
        with self._lock:
            old = self._ipc
        # With more than one player around, a pipe name that does not name a
        # pid cannot tell them apart -- see _AMBIGUOUS_PIPES. It stays allowed
        # for a lone player, where it is sometimes the only way in and cannot
        # be pointing at anybody else.
        allow_ambiguous = len(players) <= 1
        seen_pids = {player.pid for player in players}
        for player in players:
            ipc = old.get(player.pid)
            if ipc is None:
                ipc = connect_pid(player.pid, allow_ambiguous=allow_ambiguous)
            if ipc is None:
                continue
            try:
                info = snapshot_playback(ipc)
            except IpcError:
                ipc.close()
                continue
            player.pipe = ipc.path
            player.connected = True
            root = self._player_root(player.pid, ipc)
            self._ensure_player_scripts(root, player.pid)
            status = self._runtime_for(root)
            player.config_dir = str(root)
            player.ready = status.ready
            player.missing = "、".join(missing_labels(status))
            player.needs_restart = player.pid in self._needs_restart
            self._player_ready[player.pid] = (status.ready, player.missing)
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
            measured = filtered_output_fps(
                info.get("estimated_vfps"),
                info.get("display_fps"),
                info.get("vsync_ratio"),
                info.get("estimated_display_fps"),
                interpolating=player.interpolation,
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
            # Drops are only meaningful against the rate that was being aimed
            # for, so this runs after `target` is known. Without a filter the
            # target is just the source rate.
            self._update_realtime(player.pid, info, target if player.interpolation else src)
            player.realtime = self._realtime.get(player.pid)
            player.realtime_label = realtime_label(player.realtime)
            player.drop_rate = self._drops.get(player.pid)
            player.playback_clean = playback_is_clean(player.realtime, player.drop_rate)
            live[player.pid] = ipc
            seeking = bool(info.get("seeking"))
            held = self._held_off(player.pid, seeking)
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
            elif not info.get("vf_ok", True):
                # The vf read failed, so player.interpolation is a default and
                # not a reading -- "filter missing" and "mpv did not answer"
                # look identical from here. Acting on it would send a vf add
                # into an mpv that is merely busy (compiling an engine, wedged
                # vo), which is when it can least afford one. Leave the
                # recorded state alone; the next tick decides on a real answer.
                pass
            elif self.settings.enabled and player.ready:
                # Stale settings, not just a missing filter. multi <= 1 is a
                # legitimately applied no-op (nothing left to interpolate), so it
                # must not read as "filter missing" and re-apply every tick.
                stale = applied != self._filter_key(want_multi)
                missing = want_multi > 1 and not player.interpolation
                if (stale or missing) and time.monotonic() >= retry_at:
                    # tick() already has a full snapshot for this player; apply()
                    # reads only the rate fields out of it, which are present.
                    if self._apply_to(player.pid, ipc, want_multi, wait=apply_wait, info=info):
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
            for pid in [p for p in self._realtime if p not in live]:
                self._realtime.pop(pid, None)
            for pid in [p for p in self._realtime_seen if p not in live]:
                self._realtime_seen.pop(pid, None)
            for pid in [p for p in self._drops if p not in live]:
                self._drops.pop(pid, None)
            for pid in [p for p in self._config_dirs if p not in seen_pids]:
                self._config_dirs.pop(pid, None)
            for pid in [p for p in self._player_ready if p not in seen_pids]:
                self._player_ready.pop(pid, None)
            # A restarted player is a new pid, which is exactly the signal
            # that the script it was missing has now been loaded.
            self._needs_restart.intersection_update(seen_pids)
            # Against the processes that were *seen*, not the ones that
            # answered. Everything else swept here is recoverable -- dropping
            # _applied just re-applies next tick -- but the saved hwdec is
            # not: ensure_copyback_hwdec returns early once the player is
            # already on copy-back, so nothing ever writes the entry back.
            # Keying this on `live` meant one tick that failed to connect (a
            # pipe not answering while mpv builds a TensorRT engine) lost the
            # mode for good, and the player stayed on auto-copy for the rest
            # of its life, paying the GPU->CPU transfer with no filter on.
            for pid in [p for p in hwdec_pids() if p not in seen_pids]:
                forget_hwdec(pid)
            if not live:
                self._error = ""
        now = time.monotonic()
        # The engine cache only grows while a TensorRT engine is compiling, and
        # that only happens with a player connected -- so an idle app can look
        # far less often without ever missing one.
        if now >= self._housekeeping_at:
            self._housekeeping_at = now + (
                HOUSEKEEPING_ACTIVE if (live or self._bootstrapping) else HOUSEKEEPING_IDLE
            )
            cache = engine_cache_info()
            self._engine_growing_until = next_growth_deadline(
                cache.total_bytes, self._engine_bytes, now, self._engine_growing_until, ENGINE_GROWTH_GRACE
            )
            self._engine_bytes = cache.total_bytes
            gpu = gpu_snapshot()
            runtime = diagnose(self.settings.mpv_root, backend=self._backend())
            # Dropped rather than refreshed in place: the next tick re-reads
            # only the dirs it actually has players in, and a directory that
            # became ready (the installer just finished) has to be picked up.
            self._runtime_cache.clear()
            with self._lock:
                self._gpu = gpu
                self._runtime = runtime
                self._engine_cache = cache
        with self._lock:
            self._ipc = live
            self._players = players

    def set_enabled(self, enabled: bool) -> None:
        self.settings.enabled = bool(enabled)
        save_settings(self.settings)
        with self._lock:
            ipcs = list(self._ipc.items())
        if enabled and not ipcs and not self._runtime.ready:
            # Checked once, before the loop. It used to sit inside it, so with
            # no player connected the loop never ran and the message was never
            # set: the toggle flipped, the config saved, and the UI said
            # nothing about why interpolation had not started. It also
            # returned from inside the loop, skipping the trailing tick().
            #
            # Only when nothing is connected. With a player present its own
            # config dir is the one that decides, and settings.mpv_root being
            # ready says nothing about it -- that mismatch is what let the
            # filter be pushed into a player that could not load it.
            self._error = "TensorRT 執行環境尚未就緒"
            return
        for pid, ipc in ipcs:
            if enabled:
                ready, missing = self._player_ready.get(pid, (True, ""))
                if not ready:
                    # Applying anyway is not a worse message, it is a broken
                    # video: mpv accepts the vf, fails to construct the filter
                    # ("Failed to load VapourSynth VSScript library"), and ends
                    # up with no video or audio stream selected at all.
                    label = next(
                        (p.label or p.name for p in self._players if p.pid == pid), "播放器"
                    )
                    self._error = f"{label} 的設定目錄尚未安裝執行環境：{missing}"
                    continue
                # Invalidate first: whether the apply below is skipped (hold-off),
                # fails, or never runs at all, tick() has to be left able to see
                # that this mpv is out of date and finish the job later.
                self._invalidate(pid)
                if self._held_off(pid):
                    self._error = ""
                    continue
                try:
                    # Only the rate fields: this is on the click path, and the
                    # full snapshot is fourteen round trips to derive one
                    # integer. The same reading is handed to apply() so it does
                    # not immediately repeat the work.
                    info = rate_snapshot(ipc)
                except IpcError as exc:
                    self._error = str(exc)
                    continue
                # Short wait: set_enabled() runs on the pywebview bridge thread,
                # so blocking here for a multi-second apply freezes the window.
                # Handing off to tick() costs at most one 0.3s tick.
                self._apply_to(
                    pid, ipc, resolve_multi(info, self.settings),
                    announce=True, wait=UI_APPLY_WAIT, info=info,
                )
            else:
                self._remove_from(pid, ipc, announce=True, wait=UI_APPLY_WAIT)
        self.tick(apply_wait=UI_APPLY_WAIT)

    def update_settings(self, **kwargs: Any) -> None:
        # Through Settings.from_dict, not straight onto the dataclass: the
        # clamping and migration rules live there and used to run only at load
        # time, so a value set at runtime (scene_threshold=0.9 from the
        # bridge) was written into the .vpy and saved as-is, then silently
        # became 0.30 on the next launch. Same config, different playback
        # either side of a restart.
        data = self.settings.to_dict()
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        validated = Settings.from_dict(data)
        # Mutated in place rather than rebound: tick() and the bridge threads
        # both hold this same object.
        for key, value in validated.to_dict().items():
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
            # What the setting asked for is already in `settings`; this is what
            # it actually resolved to, plus the adapters it resolved against,
            # so the panel can say *why* rather than just what.
            "backend": self._backend(),
            "adapters": detect_adapters(),
        }

    def install_root(self) -> Path:
        """Where the runtime should be installed.

        A connected player's own config dir wins over the configured root:
        that is the mpv which will have to load the filter, and on a fresh
        install the configured value is only default_mpv_root()'s guess --
        typically C:\\mpv, which on most machines is an empty path nothing
        uses. Installing there produces a complete, correct runtime that the
        running player never reads.

        Raises when neither source yields somewhere an mpv actually lives:
        installing ~3.5 GB into a path with no mpv.exe (a bare C:\\mpv that
        default_mpv_root() fell back to, on a machine whose player lives
        somewhere else entirely) silently produces a complete runtime that
        nothing will ever load. Better to say so than to download it.
        """
        with self._lock:
            connections = list(self._ipc.values())
        for ipc in connections:
            root = player_config_dir(ipc)
            if root is not None:
                return root
        fallback = Path(self.settings.mpv_root)
        if not (fallback / "mpv.exe").is_file():
            raise RuntimeError(
                f"找不到要安裝的 mpv：{fallback} 沒有 mpv.exe。\n"
                "請先開啟播放器並開始播放，Fluid Motion 會直接向它問出正確的設定目錄。"
            )
        return fallback

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

                root = self.install_root()
                install_runtime(root, cb, backend=self._backend())
                install_lua(root)
                ensure_input_binding(root)
                # Remember it: the readiness panel reads diagnose(mpv_root),
                # so leaving the setting pointed at the old guess would report
                # the freshly installed runtime as still missing.
                if str(root) != str(self.settings.mpv_root):
                    self.update_settings(mpv_root=str(root))
                self._runtime = diagnose(root, backend=self._backend())
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self._bootstrap_message = f"安裝失敗：{exc}"
            finally:
                self._bootstrapping = False

        threading.Thread(target=_run, name="fluid-bootstrap", daemon=True).start()
