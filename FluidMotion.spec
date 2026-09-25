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
#
# The rest were measured the same way -- never loaded by a running app -- and
# each one's importer already treats its absence as normal:
#
# - cryptography (9.5 MB raw): pywebview imports it only inside
#   __generate_ssl_cert(), which start() calls only for ssl=True. The default is
#   False and app.py never passes it.
# - PIL's optional codecs: _avif (7.5 MB), _imagingft (FreeType, 2.1 MB),
#   _webp, _imagingcms. The frozen app loads _imaging and _imagingmath and
#   nothing else from PIL; the tray icon is a PNG drawn with shapes, no text.
#   Every one of these is imported by Pillow under try/except ImportError --
#   AvifImagePlugin and WebPImagePlugin just report SUPPORTED = False -- so
#   Image.init() walking all the plugins is unaffected.
excludes = [
    "pytest",
    "unittest",
    "numpy",
    "cryptography",
    "PIL._avif",
    "PIL._imagingft",
    "PIL._webp",
    "PIL._imagingcms",
]

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
