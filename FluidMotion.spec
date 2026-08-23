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

a = Analysis(
    ["packaging/launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "unittest"],
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
