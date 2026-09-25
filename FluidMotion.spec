# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ("fluid_motion/ui", "fluid_motion/ui"),
    ("fluid_motion/resources", "fluid_motion/resources"),
]
binaries = []
hiddenimports = [
    "fluid_motion",
    "fluid_motion.app",
    "fluid_motion.api",
    "fluid_motion.icon",
    "fluid_motion.config",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "pystray._win32",
    "win32timezone",
    "win32file",
    "win32pipe",
    "win32gui",
    "win32con",
    "pythoncom",
    "pywintypes",
]

for pkg in ("webview", "pythonnet", "clr_loader", "pystray"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# PIL stays -- icon.py draws the tray icon with it. numpy does not: nothing in
# this app, pywebview or pystray imports it. It came in through PIL._typing,
# which imports numpy.typing under `if TYPE_CHECKING:` -- never true at runtime,
# but PyInstaller's analysis does not evaluate the condition -- and numpy's own
# submodules then dragged in yaml (numpy.__config__) and charset_normalizer
# (numpy.f2py). Traced from PyInstaller's xref graph; the build interpreter's
# site-packages is shared with AX Player, which is how numpy was there to be
# found. tests/test_bundle_contents.py pins that excluding it stays safe.
excludes = ["pytest", "unittest", "numpy"]

a = Analysis(
    ["packaging/launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FluidMotion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon="fluid_motion/resources/icon.ico",
)
