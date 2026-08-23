from __future__ import annotations

import argparse
import sys
import threading

from fluid_motion.config import load_settings
from fluid_motion.core.watcher import Engine
from fluid_motion.paths import ui_dir
from fluid_motion.icon import ensure_icon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fluid-motion", description="RIFE 4.6 TensorRT 即時補幀")
    parser.add_argument("--start-hidden", action="store_true")
    parser.add_argument("--demo", action="store_true", help="只開啟介面，不連 mpv")
    args = parser.parse_args(argv)

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

    def hide() -> None:
        window = window_holder.get("w")
        if window is not None:
            window.hide()

    def quit_app() -> None:
        engine.stop()
        window = window_holder.get("w")
        if window is not None:
            window.destroy()

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

    def on_closing() -> bool:
        hide()
        return False

    window.events.closing += on_closing

    def tray() -> None:
        try:
            import pystray
            from PIL import Image
        except ImportError:
            return
        image = Image.open(icon_path)

        def show(icon=None, item=None) -> None:
            window.show()
            window.restore()

        def toggle(icon=None, item=None) -> None:
            engine.set_enabled(not engine.settings.enabled)

        def close(icon=None, item=None) -> None:
            icon.stop()
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
        icon = pystray.Icon("fluid-motion", image, "Fluid Motion", menu)
        icon.run()

    threading.Thread(target=tray, name="fluid-tray", daemon=True).start()

    def shown() -> None:
        if settings.start_hidden:
            window.hide()

    webview.start(shown, debug=False)
    engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
