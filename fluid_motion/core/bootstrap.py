from __future__ import annotations

import os
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from fluid_motion.core.proc import run_hidden
from fluid_motion.paths import download_dir, resources_dir

VSMLRT_TAG = "v15.16"
GITHUB_VSMLRT = f"https://github.com/AmusementClub/vs-mlrt/releases/download/{VSMLRT_TAG}"
GITHUB_MODELS = "https://github.com/AmusementClub/vs-mlrt/releases/download/external-models"
SEVENZR = "https://www.7-zip.org/a/7zr.exe"

Progress = Callable[[str, float], None]


def _progress(cb: Progress | None, message: str, ratio: float) -> None:
    if cb:
        cb(message, max(0.0, min(1.0, ratio)))


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
    """Download vs-mlrt TensorRT + RIFE 4.6 into the portable mpv tree."""
    mpv_root = Path(mpv_root)
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
    if not (models_dir / "rife" / "rife_v4.6.onnx").is_file() and not (models_dir / "rife_v2" / "rife_v4.6.onnx").is_file():
        v2_arc = cache / "rife_v2_v4.7z"
        _download(f"{GITHUB_MODELS}/rife_v2_v4.7z", v2_arc, cb, "RIFE v2 / v4", (0.90, 0.96))
        _extract(v2_arc, models_dir, seven)

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


def ensure_input_binding(mpv_root: Path) -> None:
    conf = mpv_root / "input.conf"
    marker = "Fluid Motion RIFE"
    line = (
        'F3 vf toggle @fluid:vapoursynth="~~/shaders/fluid_rife.vpy":8:4; '
        'show-text "Fluid Motion RIFE 4.6 TensorRT" 2500\n'
    )
    if conf.is_file() and marker in conf.read_text(encoding="utf-8", errors="replace"):
        return
    with conf.open("a", encoding="utf-8") as fh:
        fh.write("\n# Fluid Motion\n")
        fh.write(line)


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
