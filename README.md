# Fluid Motion

[English](#english) · [中文](#中文)

Realtime **RIFE 4.6** frame interpolation for **mpv**, accelerated with **TensorRT + CUDA**. A tray app in the style of SVP4: auto-detects mpv, injects a VapourSynth filter, hides in the Windows notification area.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/platform-Windows-lightgrey)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## English

### What it does

Fluid Motion is **not** a video player. It sits in the tray, finds running `mpv.exe`, and when interpolation is on it injects:

`vf add @fluid:vapoursynth=shaders/fluid_rife.vpy`

The `.vpy` script runs `vsmlrt.RIFE(model=46, backend=Backend.TRT)` inside mpv.

### Requirements

- Windows 10/11 x64
- NVIDIA GPU (TensorRT). RTX 20-series and newer recommended
- [mpv](https://mpv.io) with **VapourSynth** (portable packs that ship `vapoursynth.dll` + Python, e.g. many `C:\mpv` builds)
- `hwdec=auto-copy` in `mpv.conf` (copy-back is required for VapourSynth)

The first-run **Install TensorRT runtime** button downloads vs-mlrt TensorRT (~2.6 GB) and RIFE 4.6 ONNX into your mpv folder.

### Run from source

```bat
py -3 -m pip install -r requirements.txt
py -3 -m fluid_motion
```

Or `run.bat`. Closing the window hides to the tray.

### Build the exe

```bat
build.bat
```

Output: `dist\FluidMotion.exe` (one file). Do not run anything under `build\`.

### First use

1. Start Fluid Motion
2. Click **Install TensorRT runtime** (once)
3. Fully quit and reopen mpv
4. Play a video — the UI should show the player as connected
5. Enable interpolation

The first resolution compiles a TensorRT engine (a few minutes). Later plays at the same size reuse the cache (`%APPDATA%\FluidMotion\engines`).

mpv hotkey **F3** toggles the filter. Existing bindings (for example F2) are left alone.

### Pipeline

```
mpv  →  \\.\pipe\fluid-mpv-<pid>
     →  vf @fluid:vapoursynth
     →  vsmlrt.RIFE 4.6 + Backend.TRT (fp16, CUDA graphs)
```

### Credits

RIFE, vs-mlrt, mpv, VapourSynth, NVIDIA TensorRT — see [NOTICE](NOTICE). TensorRT/CUDA binaries are NVIDIA-licensed, not MIT.

### License

[MIT](LICENSE) © 2026 bosen12

---

## 中文

Windows 即時補幀管理器：用 **RIFE 4.6 + TensorRT / CUDA** 對 **mpv** 補幀。控制台可藏到系統匣，自動偵測已開啟的 mpv。

### 需求

- Windows 10/11 64 位
- NVIDIA 顯示卡（建議 RTX 20 以後）
- 含 **VapourSynth** 的 mpv（便攜包裡通常有 `vapoursynth.dll` 與內嵌 Python）
- `mpv.conf` 使用 `hwdec=auto-copy`

第一次在介面裡點 **安裝 TensorRT 執行環境**，會把 vs-mlrt TensorRT 與 RIFE 4.6 ONNX 裝進 mpv 目錄（約 2.6 GB）。

### 從原始碼執行

```bat
py -3 -m pip install -r requirements.txt
py -3 -m fluid_motion
```

或執行 `run.bat`。關閉視窗會藏到右下角；托盤可再打開、開關補幀、結束。

### 打包 exe

```bat
build.bat
```

產出 `dist\FluidMotion.exe`。不要執行 `build\` 底下的檔案。

### 第一次使用

1. 開啟 Fluid Motion
2. 點 **安裝 TensorRT 執行環境**
3. **完全退出並重開 mpv**
4. 播放影片，介面顯示已接上
5. 打開即時補幀

第一個解析度會編譯 TensorRT engine（可能數分鐘），之後同解析度走快取。

mpv **F3** 可切換濾鏡，不會覆蓋你現有的 F2 等快捷鍵。

### 授權

原始碼為 [MIT](LICENSE)。首次安裝下載的 TensorRT / CUDA 執行檔適用 NVIDIA 授權，見 [NOTICE](NOTICE)。
