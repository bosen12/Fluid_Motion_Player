# Fluid Motion Player

A Windows **tray utility**, not a player. It injects realtime RIFE frame
interpolation into an already-running mpv-based host by adding a VapourSynth
filter over mpv's named-pipe IPC (`vf add @fluid:vapoursynth=...vpy`), using
TensorRT/CUDA. Styled after SVP4.

It is a companion to **AX Player** (`C:\projects\AX_Player`,
github.com/bosen12/AX_Player) and has first-class support for it, but it drives
any mpv host. Keep changes scoped to that role: it manages a filter, it does
not play anything.

## HANDOFF.md lives in the AX Player repo and covers this one too

The engineering journal for **both** projects is
`C:\projects\AX_Player\HANDOFF.md`. Read it before proposing work here. It
records what has already been investigated and deliberately *not* done, with
measurements — including several items that are Fluid Motion's:

- 「量過之後撤回的『效能問題』」 — perf claims withdrawn after measuring, kept
  on the page specifically so nobody lists them again.
- 「查過、判斷不值得動」 and §8.4 — e.g. `est_matrix` guessing the colour matrix
  from frame size (real, deprioritised because the owner doesn't watch HDR),
  the unused `script` parameter in `inject._vf_arg` (deliberate — a test
  documents the `~~/` path constraint through it), `bootstrap._download`
  leaving `.part` files (assessed as harmless).
- §8.5 — **open decisions waiting on the owner.** The big one: a 126 MB `mpv/`
  directory is committed here via Git LFS and *nothing reads it*
  (`mpv_root_candidates()` never looks inside the repo; the spec ships only
  `ui` and `resources`). Removing it needs a history rewrite and it is already
  pushed, so it is not a decision to make unilaterally.

The local `findings.md` / `progress.md` / `task_plan.md` are running notes, not
specifications.

## Commands

```bat
py -3.10 -m pytest tests -q
```

Use an explicit version. `python -m pytest` picks up 3.12 here, which has no
pytest. Both `py -3.10` and `py -3.14` have the full test and build
environment; CI runs 3.10 on `windows-latest`.

- Run from source: `run.bat` or `py -3.10 -m fluid_motion`
- Build: `build.bat` → `dist/FluidMotion.exe`, copied to the repo root

`build.bat` uses `py -3`, which resolves to **3.14** on this machine — that is
the interpreter releases are built with, and the artifact size differs
noticeably on 3.10. Note also that `requirements.txt` is entirely unpinned and
`build.bat` reinstalls at build time, so two builds of the same commit are not
guaranteed to match.

Tests use fakes (`_FakeIpc`) rather than a live mpv and need no GPU.
`tests/conftest.py` redirects `APPDATA` per test — without it a test overwrites
the real user's `config.json`, which has happened.

## Architecture

`Engine` in `core/watcher.py` is a **polling reconciliation loop**
(`TICK_SECONDS = 0.3`): discover mpv processes, connect over IPC, and drive the
filter toward what the settings ask for. Housekeeping (engine-cache scan,
runtime diagnosis, `nvidia-smi`) is throttled separately from the hot tick on
purpose — a previous version scanned directory trees while idle.

Concurrency is the part that bites. One `MpvIpc` is reached by **four threads**:
the watcher tick, the pywebview bridge (`set_enabled`, which ends in a second
full `tick()`), the hotkey loop, and the bootstrap thread. `_apply_lock` only
covers apply/remove; the connection itself now serialises commands
(`MpvIpc.command` takes `self._lock`, using the caller's timeout as the
queueing budget). `tick()` has no re-entrancy guard.

**A failed IPC read is not an answer.** `vf_is_fluid(None)` is `False`, which is
indistinguishable from "no filter loaded", so a busy mpv used to read as "the
filter fell off" and get another `vf add` — exactly when it could least afford
one. `snapshot_playback` therefore reports `vf_ok` separately, and a tick that
cannot read `vf` leaves the recorded state alone. Preserve that distinction if
you touch the snapshot or the tick's decision chain.

### Two backends, one of them unverified

`vs_script.resolve_backend` picks TensorRT (`Backend.TRT`) or ncnn over Vulkan
(`Backend.NCNN_VK`) from the `backend` setting and the adapters
`gpu.detect_adapters()` finds in the driver registry.

**Nobody working on this repo has AMD hardware.** The ncnn path is verified as
far as it can be — the plugin archive exists and holds exactly `vsncnn.dll`,
the pinned `vsmlrt.py` really does define `Backend.NCNN_VK`, `RIFE()` accepts
it, the generated script parses — and no further. That it *interpolates* on an
AMD GPU is untested. Three rules hold that risk down; keep them:

1. **NVIDIA wins every tie, and so does a failed probe.** Mixed machines are
   ordinary (the dev box reports an RTX 5070 Ti beside a Ryzen iGPU), and an
   empty vendor set means "could not tell", never "no NVIDIA". A machine that
   works today must not be moved onto the untested path by a reading that did
   not arrive.
2. **`diagnose()` demands the backend's own accelerator — both halves of it.**
   An AMD tree needs `vsncnn.dll` *and* the Vulkan loader, which is not
   installable: it comes from the display driver. Either missing reports not
   ready rather than ready-with-nothing-to-run, and the message names which
   one — mpv would otherwise say only "could not init VS".
3. **The NVIDIA output is pinned.** `test_an_nvidia_gpu_still_gets_exactly_the_
   tensorrt_backend_it_did_before` fixes the `Backend.TRT` call and its
   parameters. The rendered script is byte-identical to what shipped before the
   AMD work; if you change it, that is a decision, not a refactor.

TensorRT-only concepts are absent from the ncnn script rather than translated:
no `engine_folder` (it builds no engines, so the engine cache does not apply)
and no CUDA-graph flicker gate. `NCNN_STREAMS` is pinned to 1 because tuning it
needs a measurement nobody here can take. Installing follows the same split —
2.6 GB of CUDA for TensorRT, a 2.7 MB single-DLL archive for ncnn, which gets
Vulkan from the display driver.

Other invariants:

- The `.vpy` is written into, and loaded from, **the config dir of the player
  that will read it** — not `settings.mpv_root`. `inject._vf_arg` always emits
  `~~/shaders/fluid_rife.vpy` (relative, and `~~` is the player's own config
  dir); handing `apply()` a different root writes the script in one place and
  points the filter at another, and mpv only says "could not init VS".
- mpv reloads the `.vpy` on **every seek**, so it must never be readable
  half-written.
- Changing `hwdec` at runtime costs ~0.65 s of stalled playback; setting it to
  the value it already has is nearly free.
- The filter needs copy-back frames; VapourSynth is CPU-side and a GPU-resident
  frame cannot be fed to it.

## Layout

- `fluid_motion/core/watcher.py` — the `Engine` loop (largest, most central)
- `fluid_motion/core/inject.py` — vf injection/removal, hwdec state, fps and
  drop-rate telemetry
- `fluid_motion/core/mpv_ipc.py` — named-pipe client
- `fluid_motion/core/vs_script.py` — generates the RIFE `.vpy`; `target_multi`
  and `parse_fps` live here
- `fluid_motion/core/mpv_detect.py` — process discovery and per-player labels,
  including the `axplayer.exe` → "AX Player" mapping
- `fluid_motion/core/bootstrap.py` — TensorRT/VapourSynth installer, Lua and
  hotkey install
- `fluid_motion/api.py` — the `Bridge` exposed to JS. This is the security
  boundary. Validation lives in `Engine.update_settings`, and
  `tests/test_bridge.py` pins that the bridge keeps routing through it rather
  than writing onto the dataclass — that file did not exist when this section
  was written and the paragraph said so; it does now.
- `fluid_motion/ui/` — plain HTML/CSS/JS, no framework
- `docs/` — the published GitHub Pages site, not developer docs

## Web UI: treat everything external as hostile

The page is a pywebview WebView2 view holding `window.pywebview.api` — `quit()`,
`start_setup()` (a multi-GB download), `open_engine_cache()` (`os.startfile`).
A `media-title` XSS through `innerHTML` was real: mpv reports whatever the
container's metadata says, and for a stream that comes from the far end.

Route every interpolation of externally sourced data through `escapeHtml()`.
The guard test asserts that per-interpolation — an earlier version counted
`innerHTML` sites instead, which was already false of the sinks when it was
written and would pass a swap.

## Releases

The version is bumped **by hand** in two files, `fluid_motion/__init__.py` and
`pyproject.toml`, with no CI check — it once sat at 1.0.0 while tags were at
v1.4.6. Bump both, commit as `Bump to X.Y.Z`, tag `vX.Y.Z`, then
`gh release create` with `FluidMotion.exe`. Release titles follow
`Fluid Motion vX.Y.Z`. The shipped exe carries no version resource.

The distribution copy the owner keeps lives at `C:\Fluid_Motion`
(`FluidMotion.exe`, `README.md`, `LICENSE`, `NOTICE`).

## Conventions

- UI copy is Traditional Chinese; code and comments are English.
- Comments are narrative and explain **why** — especially why a constant has
  its value. Match that.
- **Diagnostics go through `fluid_motion/log.py`**, not the stdlib `logging`
  module — one file, one format, no configuration surface to get wrong. Same
  shape as AX Player's `debug_log.py`, rotation included, so the two projects'
  logs read alike.

  It exists because v1.6.0 shipped an AMD path nobody could test and asked
  users to report back; a report needs something to report. **Log state
  *changes*, never state** — the watcher ticks three times a second, and a
  line per tick buries the one that mattered. Verified: 300 steady ticks write
  zero lines. `_backend()` is where a backend change is recorded, because it
  is the one place the answer cannot be reached around.

  `tests/conftest.py` resets `log._log_path` for the same reason it redirects
  `APPDATA`: the path is cached in a module global, so without the reset
  whichever test logs first pins the directory for all the others.
- Don't commit `dist/`, `build/`, `*.exe`, `*.onnx`, or TensorRT engines.
