from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from fluid_motion.core.proc import run_hidden
from fluid_motion.paths import download_dir, resources_dir

VSMLRT_TAG = "v15.16"
GITHUB_VSMLRT = f"https://github.com/AmusementClub/vs-mlrt/releases/download/{VSMLRT_TAG}"
GITHUB_MODELS = "https://github.com/AmusementClub/vs-mlrt/releases/download/external-models"
SEVENZR = "https://www.7-zip.org/a/7zr.exe"
# misc.SCDetect drives the scene-change check every generated .vpy makes, but
# MiscFilters was split out of the VapourSynth core (it is not in the portable
# package's vs-coreplugins either), so it has to be fetched separately. 19 KB.
MISCFILTERS_URL = (
    "https://github.com/vapoursynth/vs-miscfilters-obsolete/releases/download/R2/miscfilters-r2.7z"
)

# VapourSynth R70 is pinned deliberately: R79 replaced the flat portable
# layout with a wheel-only archive, and R70 is the newest release whose
# VSScript.dll placement matches what mpv's bridge expects without extra
# shuffling. Python 3.12 is what the R70 wheel targets (cp312).
VS_VERSION = 70
VS_PORTABLE_URL = (
    f"https://github.com/vapoursynth/vapoursynth/releases/download/"
    f"R{VS_VERSION}/VapourSynth64-Portable-R{VS_VERSION}.zip"
)
# Tried newest-first; python.org keeps old point releases, so the first hit
# wins and a pulled build just falls through to the next.
PYTHON_312 = ("3.12.10", "3.12.9", "3.12.8", "3.12.7")
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

Progress = Callable[[str, float], None]


def _progress(cb: Progress | None, message: str, ratio: float) -> None:
    if cb:
        cb(message, max(0.0, min(1.0, ratio)))


def _scaled(cb: Progress | None, lo: float, hi: float) -> Progress | None:
    """Remap a callback's 0..1 range onto lo..hi, so a sub-step can report its
    own progress without every caller having to know the outer layout."""
    if cb is None:
        return None

    def inner(message: str, ratio: float) -> None:
        cb(message, lo + (hi - lo) * max(0.0, min(1.0, ratio)))

    return inner


def vapoursynth_installed(mpv_root: Path) -> bool:
    """Both layouts count -- see the matching note in runtime.diagnose()."""
    root = Path(mpv_root)
    return (root / "VSScript.dll").is_file() or (root / "vapoursynth.dll").is_file()


def install_vapoursynth(mpv_root: Path, cb: Progress | None = None) -> None:
    """Add a portable VapourSynth to an mpv that lacks one.

    Official mpv Windows builds compile the VapourSynth bridge in
    (`-Dvapoursynth=enabled`) but ship none of its runtime, so `vf=vapoursynth`
    fails on a stock install -- which is every user who did not hand-assemble
    their mpv folder. Nothing else here can work without it: the RIFE filter
    *is* a VapourSynth script.

    This follows upstream's own Install-Portable-VapourSynth-R70.ps1 (embedded
    CPython + get-pip + the portable archive + the cp312 wheel) rather than
    copying some prebuilt bundle's layout, and merges it into the mpv root the
    way mpv+VapourSynth setups are normally arranged. Verified end to end
    against an official mpv build: `--vf=vapoursynth` loads and passes frames.
    """
    root = Path(mpv_root)
    if vapoursynth_installed(root):
        _progress(cb, "VapourSynth 已就緒", 1.0)
        return

    root.mkdir(parents=True, exist_ok=True)
    cache = download_dir()

    _progress(cb, "準備 VapourSynth…", 0.02)
    py_zip: Path | None = None
    for version in PYTHON_312:
        candidate = cache / f"python-{version}-embed-amd64.zip"
        url = f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
        try:
            _download(url, candidate, cb, f"Python {version}", (0.02, 0.30))
        except OSError:
            continue
        py_zip = candidate
        break
    if py_zip is None:
        raise RuntimeError("無法下載內嵌 Python 3.12（請檢查網路連線）")

    _progress(cb, "解壓內嵌 Python…", 0.32)
    with zipfile.ZipFile(py_zip) as archive:
        archive.extractall(root)

    # The embedded distribution ships site-packages disabled and no search
    # path beyond its own zip; pip needs `import site`, and the freshly
    # installed vapoursynth module needs Lib\site-packages on the path.
    pth = root / "python312._pth"
    if pth.is_file():
        text = pth.read_text(encoding="utf-8", errors="replace")
        text = text.replace("#import site", "import site")
        for entry in ("vs-scripts", "Lib\\site-packages"):
            if entry not in text:
                text = text.rstrip("\n") + f"\n{entry}\n"
        pth.write_text(text, encoding="utf-8")
    (root / "vs-plugins").mkdir(exist_ok=True)
    (root / "vs-scripts").mkdir(exist_ok=True)

    _progress(cb, "安裝 pip…", 0.36)
    get_pip = cache / "get-pip.py"
    _download(GET_PIP_URL, get_pip, cb, "get-pip", (0.36, 0.42))
    python_exe = root / "python.exe"
    result = run_hidden([str(python_exe), str(get_pip), "--no-warn-script-location"], timeout=600)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "pip 安裝失敗").strip()[-400:])
    # Console entry points are useless here and only add clutter next to mpv.
    scripts_dir = root / "Scripts"
    if scripts_dir.is_dir():
        for exe in scripts_dir.glob("*.exe"):
            exe.unlink(missing_ok=True)

    _progress(cb, "下載 VapourSynth…", 0.44)
    vs_zip = cache / f"VapourSynth64-Portable-R{VS_VERSION}.zip"
    _download(VS_PORTABLE_URL, vs_zip, cb, f"VapourSynth R{VS_VERSION}", (0.44, 0.80))

    _progress(cb, "解壓 VapourSynth…", 0.82)
    with zipfile.ZipFile(vs_zip) as archive:
        for name in archive.namelist():
            # docs and the C SDK are most of the archive and nothing reads
            # them from an mpv folder.
            if name.startswith(("doc/", "sdk/")):
                continue
            archive.extract(name, root)
    # Ships alongside the 3.12 build for people pinned to Python 3.8; keeping
    # it next to mpv only invites loading the wrong one.
    (root / "VSScriptPython38.dll").unlink(missing_ok=True)

    _progress(cb, "安裝 VapourSynth 模組…", 0.88)
    wheels = sorted((root / "wheel").glob("VapourSynth-*cp312*.whl")) if (root / "wheel").is_dir() else []
    if not wheels:
        raise RuntimeError("VapourSynth 套件包裡找不到 cp312 wheel")
    result = run_hidden([str(python_exe), "-m", "pip", "install", str(wheels[0])], timeout=600)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "VapourSynth 安裝失敗").strip()[-400:])

    if not vapoursynth_installed(root):
        raise RuntimeError("VapourSynth 安裝後仍找不到 VSScript.dll")
    _progress(cb, "VapourSynth 安裝完成", 1.0)


def _download(url: str, dest: Path, cb: Progress | None, label: str, span: tuple[float, float]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        _progress(cb, f"已快取 {label}", span[1])
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "FluidMotion/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    local = done / total
                    _progress(cb, f"下載 {label}  {done / 1e6:.0f}/{total / 1e6:.0f} MB", span[0] + (span[1] - span[0]) * local)
    tmp.replace(dest)
    _progress(cb, f"完成 {label}", span[1])


def _sevenzr(dest_dir: Path) -> Path:
    bundled = resources_dir() / "7zr.exe"
    if bundled.is_file():
        return bundled
    which = shutil.which("7zr") or shutil.which("7z")
    if which:
        return Path(which)
    path = dest_dir / "7zr.exe"
    if not path.is_file():
        _download(SEVENZR, path, None, "7zr", (0, 0))
    return path


def _extract(archive: Path, out_dir: Path, seven: Path, extra_args: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [str(seven), "x", str(archive), f"-o{out_dir}", "-y"]
    if extra_args:
        args.extend(extra_args)
    result = run_hidden(args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "7z 解壓縮失敗")


def install_runtime(mpv_root: Path, cb: Progress | None = None) -> None:
    """Download VapourSynth (if absent) + vs-mlrt TensorRT + RIFE into the mpv tree."""
    mpv_root = Path(mpv_root)

    # First, because everything below is a VapourSynth plugin or a script for
    # it -- installing TensorRT into an mpv that cannot load .vpy at all just
    # produces a 2.6 GB download and a filter that never applies.
    install_vapoursynth(mpv_root, _scaled(cb, 0.0, 0.18))
    cb = _scaled(cb, 0.18, 1.0)

    vs_plugins = mpv_root / "vs-plugins"
    vs_plugins.mkdir(parents=True, exist_ok=True)
    cache = download_dir()
    seven = _sevenzr(cache)

    _progress(cb, "準備 TensorRT 執行環境…", 0.02)
    part1 = cache / f"vsmlrt-windows-x64-tensorrt.{VSMLRT_TAG}.7z.001"
    part2 = cache / f"vsmlrt-windows-x64-tensorrt.{VSMLRT_TAG}.7z.002"
    _download(
        f"{GITHUB_VSMLRT}/vsmlrt-windows-x64-tensorrt.{VSMLRT_TAG}.7z.001",
        part1,
        cb,
        "TensorRT runtime (1/2)",
        (0.02, 0.38),
    )
    _download(
        f"{GITHUB_VSMLRT}/vsmlrt-windows-x64-tensorrt.{VSMLRT_TAG}.7z.002",
        part2,
        cb,
        "TensorRT runtime (2/2)",
        (0.38, 0.55),
    )

    _progress(cb, "解壓 TensorRT / vs-mlrt…", 0.56)
    _extract(part1, vs_plugins, seven)

    vsmlrt_src = vs_plugins / "vsmlrt.py"
    if vsmlrt_src.is_file():
        shutil.copy2(vsmlrt_src, mpv_root / "vsmlrt.py")

    # Without this every .vpy fails at load on misc.SCDetect, so the runtime
    # is not usable even with TensorRT and the models all in place -- which is
    # exactly what diagnose() reports as "not ready", leaving the user with a
    # complete 3.5 GB install that refuses to turn on.
    if not (vs_plugins / "MiscFilters.dll").is_file():
        misc_arc = cache / "miscfilters-r2.7z"
        _download(MISCFILTERS_URL, misc_arc, cb, "MiscFilters", (0.57, 0.60))
        misc_tmp = cache / "miscfilters-extract"
        _extract(misc_arc, misc_tmp, seven)
        # The archive keeps win32/ and win64/ side by side; only the 64-bit
        # build matches the mpv and VapourSynth used here.
        found = misc_tmp / "win64" / "MiscFilters.dll"
        if not found.is_file():
            candidates = [p for p in misc_tmp.rglob("MiscFilters.dll") if "win32" not in p.as_posix()]
            found = candidates[0] if candidates else found
        if found.is_file():
            shutil.copy2(found, vs_plugins / "MiscFilters.dll")
        shutil.rmtree(misc_tmp, ignore_errors=True)

    scripts_arc = cache / f"scripts.{VSMLRT_TAG}.7z"
    _download(f"{GITHUB_VSMLRT}/scripts.{VSMLRT_TAG}.7z", scripts_arc, cb, "vsmlrt.py", (0.62, 0.66))
    _extract(scripts_arc, mpv_root, seven)
    if not (mpv_root / "vsmlrt.py").is_file() and (vs_plugins / "vsmlrt.py").is_file():
        shutil.copy2(vs_plugins / "vsmlrt.py", mpv_root / "vsmlrt.py")

    models_arc = cache / f"models.{VSMLRT_TAG}.7z"
    _download(f"{GITHUB_VSMLRT}/models.{VSMLRT_TAG}.7z", models_arc, cb, "RIFE 模型包", (0.66, 0.88))
    models_dir = vs_plugins / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    _progress(cb, "解出 RIFE 4.6 ONNX…", 0.89)
    listed = run_hidden([str(seven), "l", "-ba", str(models_arc)])
    wanted = [
        col
        for line in listed.stdout.splitlines()
        for col in [line.strip().split()[-1] if line.strip() else ""]
        if "rife_v4.6.onnx" in col.replace("\\", "/")
    ]
    if wanted:
        _extract(models_arc, models_dir, seven, extra_args=wanted)
    else:
        _extract(models_arc, models_dir, seven)
    onnx_v1 = models_dir / "rife" / "rife_v4.6.onnx"
    onnx_v2 = models_dir / "rife_v2" / "rife_v4.6.onnx"
    if not onnx_v1.is_file() and not onnx_v2.is_file():
        nested = list(models_dir.rglob("rife_v4.6.onnx"))
        for found in nested:
            rel = "rife_v2" if "rife_v2" in found.as_posix() else "rife"
            dest = models_dir / rel / "rife_v4.6.onnx"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if found.resolve() != dest.resolve():
                shutil.copy2(found, dest)
    if not (models_dir / "rife" / "rife_v4.25.onnx").is_file() and not (models_dir / "rife_v2" / "rife_v4.25.onnx").is_file():
        m425 = cache / "rife_v4.25.7z"
        _download(f"{GITHUB_MODELS}/rife_v4.25.7z", m425, cb, "RIFE 4.25", (0.88, 0.93))
        _extract(m425, models_dir, seven)
    if not (models_dir / "rife" / "rife_v4.26.onnx").is_file() and not (models_dir / "rife_v2" / "rife_v4.26.onnx").is_file():
        m426 = cache / "rife_v4.26.7z"
        _download(f"{GITHUB_MODELS}/rife_v4.26.7z", m426, cb, "RIFE 4.26", (0.93, 0.96))
        _extract(m426, models_dir, seven)

    if not (models_dir / "rife_v2" / "rife_v4.6.onnx").is_file():
        v2_arc = cache / "rife_v2_v4.7z"
        _download(f"{GITHUB_MODELS}/rife_v2_v4.7z", v2_arc, cb, "RIFE v2 / v4", (0.90, 0.96))
        _extract(v2_arc, models_dir, seven)
        nested = list(models_dir.rglob("rife_v4.6.onnx"))
        for found in nested:
            if "rife_v2" in found.as_posix():
                dest = models_dir / "rife_v2" / "rife_v4.6.onnx"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if found.resolve() != dest.resolve():
                    shutil.copy2(found, dest)

    _ensure_python_pth(mpv_root)
    _progress(cb, "執行環境已就緒。請重新開啟 mpv。", 1.0)


def _ensure_python_pth(mpv_root: Path) -> None:
    pth = mpv_root / "python312._pth"
    if not pth.is_file():
        return
    text = pth.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    changed = False
    if "import site" not in text.replace("#import site", ""):
        # keep the zip + cwd, enable site
        new_lines = []
        for line in lines:
            if line.strip() == "#import site":
                new_lines.append("import site")
                changed = True
            else:
                new_lines.append(line)
        if "import site" not in "\n".join(new_lines):
            new_lines.append("import site")
            changed = True
        if changed:
            pth.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    plugins = "vs-plugins"
    if plugins not in text:
        with pth.open("a", encoding="utf-8") as fh:
            fh.write("vs-plugins\n")


def install_lua(mpv_root: Path) -> Path:
    src = resources_dir() / "zz-fluid-ipc.lua"
    dest = mpv_root / "scripts" / "zz-fluid-ipc.lua"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


FLUID_VF_TOGGLE = '@fluid:vapoursynth="~~/shaders/fluid_rife.vpy":4:1'


def ensure_input_binding(mpv_root: Path) -> None:
    """F3 is owned by zz-fluid-ipc.lua. Strip old input.conf toggles (:8:4)."""
    conf = mpv_root / "input.conf"
    if not conf.is_file():
        return
    text = conf.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("F3") and "fluid_rife" in raw:
            out.append("# F3 is bound in scripts/zz-fluid-ipc.lua (toggle 開/關)")
            changed = True
        else:
            out.append(raw)
    if changed:
        trailing = "\n" if text.endswith("\n") else ""
        conf.write_text("\n".join(out) + trailing, encoding="utf-8")


def autostart_path() -> Path:
    appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "FluidMotion.bat"


def set_autostart(enabled: bool, command: str) -> None:
    path = autostart_path()
    if not enabled:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'@echo off\nstart "" {command} --start-hidden\n', encoding="utf-8")
