from __future__ import annotations

from typing import Any

from fluid_motion.core.bootstrap import autostart_path, set_autostart
from fluid_motion.core.watcher import Engine


class Bridge:
    def __init__(self, engine: Engine, hide: Any, quit_app: Any):
        self.engine = engine
        self._hide = hide
        self._quit = quit_app

    def get_state(self) -> dict[str, Any]:
        return self.engine.state()

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        self.engine.set_enabled(bool(enabled))
        return self.get_state()

    def set_profile(self, profile: str) -> dict[str, Any]:
        self.engine.update_settings(profile=str(profile))
        return self.get_state()

    def set_model(self, model: int) -> dict[str, Any]:
        self.engine.update_settings(rife_model=int(model))
        return self.get_state()

    def set_scene(self, value: float) -> dict[str, Any]:
        self.engine.update_settings(scene_threshold=float(value))
        return self.get_state()

    def set_backend(self, backend: str) -> dict[str, Any]:
        # Validated in Settings.from_dict, same as every other setter here:
        # anything unrecognised falls back to "auto" rather than pinning the
        # machine to a backend its GPU cannot run.
        self.engine.update_settings(backend=str(backend))
        return self.get_state()

    def set_autostart(self, enabled: bool) -> dict[str, Any]:
        from fluid_motion.paths import launch_command

        set_autostart(bool(enabled), launch_command())
        self.engine.update_settings(autostart=bool(enabled))
        return self.get_state()

    def start_setup(self) -> dict[str, Any]:
        self.engine.start_bootstrap()
        return self.get_state()

    def clear_engine_cache(self) -> dict[str, Any]:
        from fluid_motion.core.engine_cache import clear

        clear()
        return self.get_state()

    def open_engine_cache(self) -> None:
        import os

        from fluid_motion.paths import engine_cache_dir

        os.startfile(str(engine_cache_dir()))  # noqa: S606 — local folder, Windows-only app

    def hide(self) -> None:
        self._hide()

    def quit(self) -> None:
        self._quit()

    def autostart_file(self) -> str:
        return str(autostart_path())
