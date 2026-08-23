from __future__ import annotations

from pathlib import Path

from fluid_motion.paths import resources_dir


def ensure_icon() -> Path:
    path = resources_dir() / "icon.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 200:
        return path
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # projector-lamp amber on emulsion black
    draw.rounded_rectangle((2, 2, 61, 61), radius=14, fill=(22, 16, 12, 255))
    draw.rounded_rectangle((10, 14, 54, 50), radius=6, fill=(36, 26, 16, 255), outline=(232, 168, 74, 255), width=2)
    # sprocket holes
    for y in (20, 32, 44):
        draw.ellipse((14, y - 3, 20, y + 3), fill=(232, 168, 74, 255))
        draw.ellipse((44, y - 3, 50, y + 3), fill=(232, 168, 74, 255))
    # interpolated frame ghost
    draw.rectangle((24, 22, 40, 42), fill=(232, 168, 74, 210))
    img.save(path)
    ico = resources_dir() / "icon.ico"
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    return path
