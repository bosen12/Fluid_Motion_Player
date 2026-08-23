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

    def set_streams(self, value: int) -> dict[str, Any]:
        self.engine.update_settings(trt_streams=int(value))
        return self.get_state()

    def set_autostart(self, enabled: bool) -> dict[str, Any]:
        from fluid_motion.paths import launch_command

        set_autostart(bool(enabled), launch_command())
        self.engine.update_settings(autostart=bool(enabled))
        return self.get_state()

    def start_setup(self) -> dict[str, Any]:
        self.engine.start_bootstrap()
        return self.get_state()

    def hide(self) -> None:
        self._hide()

    def quit(self) -> None:
        self._quit()

    def autostart_file(self) -> str:
        return str(autostart_path())
