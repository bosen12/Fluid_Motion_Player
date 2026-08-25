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
- Any mpv-based player — plain [mpv](https://mpv.io), mpv.net, or a host that
  embeds libmpv such as [AX Player](https://github.com/bosen12/AX_Player)

Nothing else needs preparing. Earlier versions also required an mpv that
already had VapourSynth, plus `hwdec=auto-copy` set by hand; both are now
handled for you:

- **VapourSynth** is installed when your mpv lacks it. Official mpv builds
  compile the bridge in but ship none of its runtime, so `vf=vapoursynth`
  fails on a stock install — Fluid Motion adds the missing pieces.
- **Copy-back decoding** is switched on over IPC while interpolation runs,
  and restored when it stops. VapourSynth cannot read a GPU-resident frame,
  so this used to fail silently whenever `mpv.conf` said otherwise.

**Install TensorRT runtime** downloads roughly 3.5 GB in total (vs-mlrt
TensorRT, RIFE models, and VapourSynth if needed) into the config directory
of the player you are running — start playback first so it can ask the
player where that is.

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

1. Start your player and begin playing something
2. Start Fluid Motion — the UI should show the player as connected
3. Click **Install TensorRT runtime** (once)
4. Fully quit and reopen the player, so it picks up the new scripts
5. Play a video and enable interpolation

Step 1 comes first on purpose: the installer asks the connected player for
its config directory rather than guessing at install paths, which is what
lets this work with an mpv in any location and with embedded hosts. With no
player running it falls back to the configured `mpv_root`, and refuses
rather than installing several GB somewhere no player will read.

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
- 任何以 mpv 為基礎的播放器——原生 [mpv](https://mpv.io)、mpv.net，或內嵌
  libmpv 的殼（例如 [AX Player](https://github.com/bosen12/AX_Player)）

其他都不用先準備。舊版還要求 mpv 本身已含 VapourSynth，並自行在 `mpv.conf`
設好 `hwdec=auto-copy`；這兩件事現在都自動處理：

- **VapourSynth**：mpv 沒有的話會自動安裝。官方 mpv 建置有把 bridge 編譯進去，
  但不附帶執行期,所以原封不動的安裝跑 `vf=vapoursynth` 會失敗——Fluid Motion
  會補上缺的部分。
- **copy-back 解碼**：補幀運作期間透過 IPC 切換，停止時還原。VapourSynth 無法
  讀取 GPU 常駐的影格，以前 `mpv.conf` 設定不同就會靜默失敗。

點 **安裝 TensorRT 執行環境** 總共約下載 3.5 GB（vs-mlrt TensorRT、RIFE 模型，
必要時加上 VapourSynth），裝進你**正在使用的播放器**的設定目錄——請先開始播放，
它才問得到那個目錄在哪。

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

1. 開啟播放器並開始播放
2. 開啟 Fluid Motion——介面應顯示已接上該播放器
3. 點 **安裝 TensorRT 執行環境**（只需一次）
4. **完全退出並重開播放器**，讓它載入新裝的腳本
5. 播放影片，打開即時補幀

第 1 步放在最前面是刻意的:安裝時會**直接問連線中的播放器**它的設定目錄在哪，
而不是猜測安裝路徑——這正是它能支援任意位置的 mpv 以及內嵌型播放器的原因。
沒有播放器在跑時會退回設定檔裡的 `mpv_root`，若該處也找不到 mpv 就直接中止，
而不是把好幾 GB 裝進一個沒有播放器會讀的地方。

第一個解析度會編譯 TensorRT engine（可能數分鐘），之後同解析度走快取。

mpv **F3** 可切換濾鏡，不會覆蓋你現有的 F2 等快捷鍵。

### 授權

原始碼為 [MIT](LICENSE)。首次安裝下載的 TensorRT / CUDA 執行檔適用 NVIDIA 授權，見 [NOTICE](NOTICE)。
