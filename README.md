# Fluid Motion

[English](#english) · [中文](#中文)

Realtime **RIFE** frame interpolation for **mpv**, accelerated with **TensorRT + CUDA** on NVIDIA or **ncnn + Vulkan** on AMD. A tray app in the style of SVP4: auto-detects mpv, selects the matching backend, injects a VapourSynth filter, and hides in the Windows notification area.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/platform-Windows-lightgrey)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## English

### What it does

Fluid Motion is **not** a video player. It sits in the tray, finds running `mpv.exe`, and when interpolation is on it injects:

`vf add @fluid:vapoursynth=shaders/fluid_rife.vpy`

The `.vpy` script runs `vsmlrt.RIFE(...)` inside mpv, on whichever inference
backend your GPU can use.

### Requirements

- Windows 10/11 x64
- A GPU, and the backend is picked for you:
  - **NVIDIA** — TensorRT. RTX 20-series and newer recommended. This is the
    path with real measurements behind it.
  - **AMD** — ncnn over Vulkan. ⚠️ **Untested on AMD hardware.** Nobody working
    on this project has an AMD card, so while the code path is complete and the
    install is verified, whether RIFE actually interpolates on an AMD GPU has
    no runtime evidence. If you try it, a report either way is genuinely
    useful — attach `%APPDATA%\FluidMotion\fluid_debug.log`.
  - A machine with both reports both; NVIDIA wins, and you can override it in
    the app.
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

The install button downloads into the config directory of the player you are
running — roughly 3.5 GB on NVIDIA (vs-mlrt TensorRT, RIFE models, and
VapourSynth if needed). On AMD it is far smaller: the ncnn plugin is a single
2.7 MB file, because Vulkan itself comes from your display driver — start playback first so it can ask the
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
3. Click **Install runtime** (once); Fluid Motion selects TensorRT or ncnn/Vulkan for the detected GPU
4. Fully quit and reopen the player, so it picks up the new scripts
5. Play a video and enable interpolation

Step 1 comes first on purpose: the installer asks the connected player for
its config directory rather than guessing at install paths, which is what
lets this work with an mpv in any location and with embedded hosts. With no
player running it falls back to the configured `mpv_root`, and refuses
rather than installing several GB somewhere no player will read.

That directory is also what decides whether a player can interpolate at all,
so readiness is reported **per player**: a player whose own config directory
has no runtime is shown as blocked, with what it is missing, and the filter
is never pushed into it. (It used to be pushed anyway whenever the
*configured* root happened to be complete -- mpv accepts the filter, fails to
construct it, and ends up with no video stream selected at all.)

The small IPC script that binds F3 and debounces seeks is installed into each
connected player's config directory as well, not only the configured one.
Restart that player once afterwards: mpv loads scripts only at launch, and
the UI says so on the player's card.

On NVIDIA, the first resolution compiles a TensorRT engine (a few minutes). Later plays at the same size reuse the cache (`%APPDATA%\FluidMotion\engines`). ncnn/Vulkan does not compile or use this engine cache.

mpv hotkey **F3** toggles the filter. Existing bindings (for example F2) are left alone.

### Pipeline

```
mpv  →  \\.\pipe\fluid-mpv-<pid>
     →  vf @fluid:vapoursynth
     →  vsmlrt.RIFE + Backend.TRT (NVIDIA) or Backend.NCNN_VK (AMD)
```

### Credits

RIFE, vs-mlrt, mpv, VapourSynth, NVIDIA TensorRT — see [NOTICE](NOTICE). TensorRT/CUDA binaries are NVIDIA-licensed, not MIT.

### License

[MIT](LICENSE) © 2026 bosen12

---

## 中文

Windows 即時補幀管理器：用 **RIFE** 對 **mpv** 補幀，依顯示卡自動選用推論後端。控制台可藏到系統匣，自動偵測已開啟的 mpv。

### 需求

- Windows 10/11 64 位
- 一張顯示卡,後端會自動選:
  - **NVIDIA** —— TensorRT（建議 RTX 20 以後）。這是有實測數據的路徑。
  - **AMD** —— ncnn / Vulkan。⚠️ **未經 AMD 實機驗證。** 這個專案沒有人有 AMD
    顯示卡,程式路徑完整、安裝流程也驗證過,但 RIFE 到底能不能在 AMD GPU 上
    補幀,沒有任何執行期證據。如果你試了,**不論成功或失敗都請回報**——附上
    `%APPDATA%\FluidMotion\fluid_debug.log` 即可。
  - 兩張都有的機器兩者都會偵測到,**以 NVIDIA 優先**,也可以在程式裡手動指定。
- 任何以 mpv 為基礎的播放器——原生 [mpv](https://mpv.io)、mpv.net，或內嵌
  libmpv 的殼（例如 [AX Player](https://github.com/bosen12/AX_Player)）

其他都不用先準備。舊版還要求 mpv 本身已含 VapourSynth，並自行在 `mpv.conf`
設好 `hwdec=auto-copy`；這兩件事現在都自動處理：

- **VapourSynth**：mpv 沒有的話會自動安裝。官方 mpv 建置有把 bridge 編譯進去，
  但不附帶執行期,所以原封不動的安裝跑 `vf=vapoursynth` 會失敗——Fluid Motion
  會補上缺的部分。
- **copy-back 解碼**：補幀運作期間透過 IPC 切換，停止時還原。VapourSynth 無法
  讀取 GPU 常駐的影格，以前 `mpv.conf` 設定不同就會靜默失敗。

按下安裝鈕會裝進你**正在使用的播放器**的設定目錄——請先開始播放，它才問得到
那個目錄在哪。NVIDIA 約下載 3.5 GB（vs-mlrt TensorRT、RIFE 模型，必要時加上
VapourSynth）；AMD 小很多,ncnn 外掛只有單一個 2.7 MB 的檔案,因為 Vulkan 本身
由顯示卡驅動提供。

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
3. 點 **安裝執行環境**（只需一次）；Fluid Motion 會依偵測到的 GPU 選擇 TensorRT 或 ncnn/Vulkan
4. **完全退出並重開播放器**，讓它載入新裝的腳本
5. 播放影片，打開即時補幀

第 1 步放在最前面是刻意的:安裝時會**直接問連線中的播放器**它的設定目錄在哪，
而不是猜測安裝路徑——這正是它能支援任意位置的 mpv 以及內嵌型播放器的原因。
沒有播放器在跑時會退回設定檔裡的 `mpv_root`，若該處也找不到 mpv 就直接中止，
而不是把好幾 GB 裝進一個沒有播放器會讀的地方。

那個目錄同時也決定了一個播放器到底能不能補幀，所以就緒狀態是**逐播放器**判斷的：
自己的設定目錄沒有執行環境的播放器會顯示為未就緒並列出缺什麼，濾鏡不會被送進去。
（以前只要**設定檔裡**那個 root 剛好是完整的就會照送——mpv 會收下濾鏡、然後建構失敗，
最後連影片軌都沒有了。）

負責綁定 F3 與 seek 防抖的 IPC 小腳本，現在也會裝進每個連線播放器自己的設定目錄，
不再只裝設定檔裡那一個。裝完請把該播放器重開一次：mpv 只在啟動時載入腳本，
播放器卡片上會提示。

NVIDIA 上第一個解析度會編譯 TensorRT engine（可能數分鐘），之後同解析度走快取；ncnn/Vulkan 不編譯也不使用這份 engine 快取。

mpv **F3** 可切換濾鏡，不會覆蓋你現有的 F2 等快捷鍵。

### 授權

原始碼為 [MIT](LICENSE)。首次安裝下載的 TensorRT / CUDA 執行檔適用 NVIDIA 授權，見 [NOTICE](NOTICE)。
