from __future__ import annotations

import argparse
import os
import sys
import threading

from fluid_motion.config import load_settings
from fluid_motion.core.proc import terminate_children
from fluid_motion.core.watcher import Engine
from fluid_motion.icon import ensure_icon
from fluid_motion.log import log
from fluid_motion.paths import ui_dir
from fluid_motion.single import handover_or_continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fluid-motion", description="RIFE TensorRT 即時補幀")
    parser.add_argument("--start-hidden", action="store_true")
    parser.add_argument("--demo", action="store_true", help="只開啟介面，不連 mpv")
    args = parser.parse_args(argv)

    if not args.demo and not handover_or_continue():
        return 0

    settings = load_settings()
    if args.start_hidden:
        settings.start_hidden = True

    icon_path = ensure_icon()
    engine = Engine(settings)
    if not args.demo:
        engine.start()

    try:
        import webview
    except ImportError:
        print("請先安裝依賴：python -m pip install -r requirements.txt", file=sys.stderr)
        engine.stop()
        return 1

    from fluid_motion.api import Bridge

    window_holder: dict[str, object] = {}
    tray_icon: dict[str, object] = {}

    def hide() -> None:
        window = window_holder.get("w")
        if window is not None:
            window.hide()

    def quit_app() -> None:
        try:
            engine.stop()
        except Exception:
            pass
        icon = tray_icon.get("icon")
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        window = window_holder.get("w")
        if window is not None:
            try:
                window.destroy()
            except Exception:
                pass
        # engine.stop() reaches the tick thread and the IPC connections. It
        # does not reach the bootstrap thread, which is a daemon blocked inside
        # run_hidden waiting on 7z -- and os._exit below joins nothing. On
        # Windows that child then outlives the app: reproduced on this code
        # path, parent gone, extraction still running. Worse than the wasted
        # work, install_runtime's "one at a time" guard is a per-process flag,
        # so the next launch starts a second extraction into the directory the
        # orphan is still writing.
        try:
            killed = terminate_children()
            if killed:
                log(f"quit: killed {killed} child process(es) still running")
        except Exception:  # noqa: BLE001 — quitting must not be blockable
            pass
        # pywebview + pystray otherwise leave a headless process (no tray icon).
        os._exit(0)

    api = Bridge(engine, hide, quit_app)
    html = (ui_dir() / "index.html").resolve()
    window = webview.create_window(
        "Fluid Motion",
        url=html.as_uri(),
        js_api=api,
        width=980,
        height=640,
        min_size=(840, 560),
        background_color="#16110C",
        frameless=True,
        easy_drag=False,
        shadow=True,
    )
    window_holder["w"] = window

    tray_ok = threading.Event()

    def show_window() -> None:
        win = window_holder.get("w")
        if win is None:
            return
        try:
            win.show()
            win.restore()
        except Exception:
            pass

    def on_closing() -> bool:
        if tray_ok.is_set():
            hide()
            return False
        quit_app()
        return True

    window.events.closing += on_closing

    def tray() -> None:
        try:
            import pythoncom
            import pystray
            from PIL import Image
        except ImportError:
            return
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            image = Image.open(icon_path)

            def show(icon=None, item=None) -> None:
                show_window()

            def toggle(icon=None, item=None) -> None:
                engine.set_enabled(not engine.settings.enabled)

            def close(icon=None, item=None) -> None:
                try:
                    icon.stop()
                except Exception:
                    pass
                quit_app()

            menu = pystray.Menu(
                pystray.MenuItem("顯示 Fluid Motion", show, default=True),
                pystray.MenuItem(
                    lambda item: "關閉即時補幀" if engine.settings.enabled else "開啟即時補幀",
                    toggle,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("結束", close),
            )
            icon = pystray.Icon("FluidMotion", image, "Fluid Motion", menu)
            tray_icon["icon"] = icon
            tray_ok.set()
            run_detached = getattr(icon, "run_detached", None)
            if callable(run_detached):
                run_detached()
            else:
                icon.run()
        except Exception:
            tray_ok.clear()

    threading.Thread(target=tray, name="fluid-tray", daemon=True).start()
    tray_ok.wait(timeout=1.5)

    def shown() -> None:
        if settings.start_hidden and tray_ok.is_set():
            window.hide()

    engine.set_on_show(show_window)

    webview.start(shown, debug=False)
    engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
