const PROFILES = [
  { id: "2x", label: "2×" },
  { id: "3x", label: "3×" },
  { id: "4x", label: "4×" },
  { id: "120", label: "120" },
  { id: "144", label: "144" },
  { id: "display", label: "螢幕" },
];

const MODELS = [
  { id: 426, label: "4.26" },
  { id: 425, label: "4.25" },
  { id: 46, label: "4.6" },
];

const SCENE_PRESETS = [
  { id: "live", label: "真人", value: 0.1 },
  { id: "anime", label: "動畫", value: 0.15 },
];

function modelLabel(id) {
  const hit = MODELS.find((m) => Number(m.id) === Number(id));
  return hit ? `RIFE ${hit.label}` : "RIFE";
}

const mock = {
  settings: {
    enabled: false,
    profile: "2x",
    scene_threshold: 0.1,
    rife_model: 426,
    autostart: false,
  },
  players: [],
  gpu: {
    name: "NVIDIA GeForce RTX 5070 Ti",
    utilization: 0,
    memory_used_mb: 0,
    memory_total_mb: 16303,
    power_w: 0,
    available: true,
  },
  runtime: {
    ready: false,
    checks: [
      { id: "mpv", label: "mpv", ok: true },
      { id: "vapoursynth", label: "VapourSynth", ok: true },
      { id: "tensorrt", label: "TensorRT + CUDA", ok: false },
      { id: "rife46", label: "RIFE 4.6", ok: false },
      { id: "rife425", label: "RIFE 4.25", ok: false },
    ],
  },
  error: "",
  bootstrap: { running: false, message: "", progress: 0 },
  player_count: 0,
  connected: 0,
  gpu_safe_mode: true,
  engine_cache: { count: 0, total_bytes: 0, path: "" },
  engine_compiling: false,
};

function api() {
  return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
}

async function call(name, ...args) {
  const bridge = api();
  if (!bridge || typeof bridge[name] !== "function") {
    if (name === "get_state") return mock;
    return mock;
  }
  return bridge[name](...args);
}

function $(id) {
  return document.getElementById(id);
}

function setPressed(el, on) {
  el.setAttribute("aria-pressed", on ? "true" : "false");
}

// media-title is not the file name: mpv reads it from the container's own
// metadata, so it carries whatever the file (or, for a stream, the far end)
// says. Measured with an mkv remuxed to carry
// title='<img src=x onerror=alert(1)>' -- mpv reports that string verbatim,
// and an <img onerror> written through innerHTML runs, with
// window.pywebview.api in scope. Everything else in this file uses
// textContent; this is the one place a string has to become markup.
function escapeHtml(value) {
  return String(value == null ? "" : value).replace(
    /[&<>"']/g,
    (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]
  );
}

function renderPlayers(players) {
  const root = $("players");
  if (!players.length) {
    root.innerHTML = `<p class="empty">尚未偵測到 mpv。播放影片後會自動接上。</p>`;
    return;
  }
  root.innerHTML = players
    .map((p) => {
      const res = p.width ? `${p.width}×${p.height}` : "";
      // p.label, not a hardcoded "mpv": with two players open both cards used
      // to be identical, and for an embedded host (AX Player) the label was
      // wrong on top of being ambiguous.
      const name = escapeHtml(p.label || p.name || "mpv");
      // The filter is written into and loaded from the player's *own* config
      // dir, so that is the directory whose readiness decides whether this
      // player can interpolate -- not the one the checklist panel shows.
      const blocked = p.connected && p.ready === false;
      const notes = [];
      if (blocked) {
        notes.push(
          `<div class="player-note" data-kind="blocked">此播放器的設定目錄尚未安裝執行環境：${escapeHtml(
            p.missing || ""
          )}<br><span class="player-dir">${escapeHtml(p.config_dir || "")}</span></div>`
        );
      } else if (p.connected && p.needs_restart) {
        notes.push(
          `<div class="player-note">已為此播放器安裝 F3 控制腳本，重新開啟播放器後生效</div>`
        );
      }
      return `<article class="player" data-active="${p.connected && p.interpolation}" data-blocked="${blocked}">
        <div class="player-name"><span>${name}</span><span>${p.connected ? "已連線" : "未連線"}</span></div>
        <div class="player-media">${escapeHtml(p.media || res || "pid " + p.pid)}</div>
        ${notes.join("")}
      </article>`;
    })
    .join("");
}

// Escaped even though every label in runtime.py is built from literals today:
// Check.detail is already assembled from str() of whatever the probe found (an
// mpv error string, a path), so the day a label carries one of those this stops
// being a formality. The guard test counts escapeHtml() calls against
// interpolations rather than counting innerHTML sites, so a raw hole added here
// fails the suite instead of waiting for the data to turn hostile.
function renderChecks(checks) {
  $("checks").innerHTML = (checks || [])
    .map(
      (c) => `<span class="check" data-ok="${escapeHtml(c.ok)}">${escapeHtml(c.label)}</span>`
    )
    .join("");
}

// Chip groups are built once and then only have aria-pressed flipped.
// Re-running innerHTML on every 900ms poll used to destroy the button under the
// cursor mid-click: with the mousedown target detached, the browser dispatches
// click on the container instead, closest("[data-profile]") returns null, and
// the handler silently returns -- the user's profile change just vanished.
function chipGroup(rootId, items, attr, extraClass) {
  const root = $(rootId);
  if (!root || root.dataset.built === "true") return root;
  root.innerHTML = items
    .map(
      (it) =>
        `<button type="button" class="chip${extraClass ? " " + extraClass : ""}" ${attr}="${it.id}" aria-pressed="false">${it.label}</button>`
    )
    .join("");
  root.dataset.built = "true";
  return root;
}

function markPressed(root, isActive) {
  if (!root) return;
  for (const btn of root.children) setPressed(btn, isActive(btn));
}

function renderProfiles(active) {
  const root = chipGroup("profiles", PROFILES, "data-profile");
  markPressed(root, (btn) => btn.dataset.profile === String(active));
}

function renderModels(active) {
  const root = chipGroup("models", MODELS, "data-model");
  markPressed(root, (btn) => Number(btn.dataset.model) === Number(active));
}

function renderScenePresets(value) {
  const items = SCENE_PRESETS.map((p) => ({ id: p.value, label: p.label }));
  const root = chipGroup("scene-presets", items, "data-scene-preset", "chip-sm");
  const current = Number(value);
  markPressed(root, (btn) => Math.abs(current - Number(btn.dataset.scenePreset)) < 0.005);
}

function renderCache(cache) {
  const stat = $("cache-stat");
  if (!stat) return;
  const info = cache || { count: 0, total_bytes: 0 };
  if (!info.count) {
    stat.textContent = "尚無快取";
    return;
  }
  const gb = info.total_bytes / 1e9;
  const size = gb >= 1 ? `${gb.toFixed(1)} GB` : `${(info.total_bytes / 1e6).toFixed(0)} MB`;
  stat.textContent = `${info.count} 個引擎 · ${size}`;
}

// Measured playback speed against realtime. Only ever reports a shortfall:
// mpv presents frames at the target rate and no faster, so a pipeline with
// room to spare looks identical to one that is exactly keeping up. Below 1.00
// is the case worth surfacing -- 4K interpolation sits around 0.3 on hardware
// that handles 1080p without noticing, and nothing else in the UI says so.
function renderRealtime(player, connected, compiling, settling) {
  const stat = $("realtime-stat");
  const bar = $("realtime-bar");
  const note = $("realtime-note");
  if (!stat || !bar || !note) return;

  const ratio = player && typeof player.realtime === "number" ? player.realtime : null;
  const drops = player && typeof player.drop_rate === "number" ? player.drop_rate : null;
  // Keeping the clock is not the same as keeping the picture: mpv can hold
  // realtime by discarding frames, which reads as a healthy 1.00 while the
  // playback visibly stutters. Measured at 120fps on real content, that was
  // 1.00x with 9.6% of frames thrown away.
  const dropping = drops !== null && drops >= 0.01;
  const measuring = connected > 0 && player && !player.paused;

  if (!connected) {
    stat.textContent = "—";
    bar.style.width = "0%";
    note.textContent = "沒有連上的播放器。";
  } else if (compiling) {
    stat.textContent = "編譯中";
    bar.style.width = "0%";
    note.textContent = "TensorRT 引擎編譯中,這段期間的速度不代表實際效能。";
  } else if (player && player.paused) {
    stat.textContent = "已暫停";
    bar.style.width = "0%";
    note.textContent = "播放暫停中,無法測量。";
  } else if (ratio === null || settling) {
    stat.textContent = "測量中…";
    bar.style.width = "0%";
    note.textContent = "播放速度相對於實際時間。1.00× 表示跟得上。";
  } else {
    const shown = Math.min(ratio, 1);
    stat.textContent = `${shown.toFixed(2)}×`;
    bar.style.width = `${Math.max(0, Math.min(1, shown)) * 100}%`;
    if (shown < 0.97) {
      note.textContent = `跟不上:只有 ${Math.round(shown * 100)}% 的實時速度。試試降低目標幀率、換較輕的模型,或關閉補幀。`;
    } else if (dropping) {
      // Realtime held by discarding frames. The rate looks perfect and the
      // picture stutters anyway, so this has to read as a problem.
      note.textContent = `維持實時,但正在丟棄 ${(drops * 100).toFixed(1)}% 的影格 —— 畫面會頓。試試降低目標幀率或換較輕的模型。`;
    } else {
      note.textContent = "跟得上實時播放。";
    }
  }

  const bad =
    measuring && ratio !== null && !settling && !compiling && (ratio < 0.97 || dropping);
  stat.classList.toggle("is-bad", Boolean(bad));
  bar.classList.toggle("is-bad", Boolean(bad));
}

function tickNumber(el, next) {
  if (el.textContent === next) return;
  el.textContent = next;
}

function render(state) {
  const enabled = Boolean(state.settings.enabled);
  const connected = state.connected || 0;
  $("live").dataset.on = connected > 0 ? "true" : "false";
  $("live-text").textContent = connected > 0 ? "已接上 mpv" : "等待 mpv";
  $("gpu-name").textContent = state.gpu.available ? state.gpu.name : "未偵測到 NVIDIA GPU";
  const gpuMode = $("gpu-mode");
  if (gpuMode) {
    if (!state.gpu.available) {
      gpuMode.textContent = "";
      gpuMode.removeAttribute("data-mode");
    } else {
      const safe = state.gpu_safe_mode !== false;
      gpuMode.textContent = safe ? "安全模式" : "加速模式";
      gpuMode.dataset.mode = safe ? "safe" : "fast";
    }
  }

  const player = (state.players || []).find((p) => p.connected) || (state.players || [])[0];
  const compiling = Boolean(state.engine_compiling);
  const settling = !compiling && Boolean(player && player.settling);
  const interpolating = Boolean(
    enabled && (state.players || []).some((p) => p.connected && p.interpolation)
  );
  const srcEl = $("src-fps");
  const dstEl = $("dst-fps");
  const dstBlock = dstEl.closest(".fps-block");
  const outLabel = !compiling && !settling && player && (player.output_fps || player.estimated_vfps);
  tickNumber(srcEl, connected && player && player.fps ? player.fps : "—");
  tickNumber(
    dstEl,
    connected === 0
      ? "—"
      : outLabel
        ? outLabel
        : compiling
          ? "編譯中"
          : settling || enabled
            ? "…"
            : "—"
  );
  const bad = !compiling && Boolean(player && player.fps_ok === false);
  dstBlock.classList.toggle("is-bad", bad);
  dstEl.classList.toggle("is-bad", bad);
  dstEl.setAttribute("aria-invalid", bad ? "true" : "false");

  const toggle = $("toggle");
  setPressed(toggle, enabled);
  toggle.querySelector(".toggle-label").textContent = !enabled
    ? "未啟用"
    : interpolating
      ? "補幀中"
      : connected > 0
        ? "套用中"
        : "待命";
  toggle.disabled = !state.runtime.ready && !enabled;
  toggle.dataset.state = state.runtime.ready ? "ready" : "error";

  renderPlayers(state.players || []);
  renderProfiles(state.settings.profile);
  renderModels(state.settings.rife_model);
  renderChecks(state.runtime.checks);

  $("scene").value = state.settings.scene_threshold;
  $("scene-val").textContent = Number(state.settings.scene_threshold).toFixed(2);
  renderScenePresets(state.settings.scene_threshold);
  renderRealtime(player, connected, compiling, settling);
  renderCache(state.engine_cache);

  const util = state.gpu.utilization || 0;
  $("gpu-bar").style.width = `${util}%`;
  $("gpu-stat").textContent = state.gpu.available
    ? `${util}% · ${state.gpu.memory_used_mb} / ${state.gpu.memory_total_mb} MB · ${Math.round(state.gpu.power_w)} W`
    : "—";

  const setupBtn = $("setup");
  const ready = Boolean(state.runtime.ready);
  setupBtn.hidden = ready && !state.bootstrap.running;
  setupBtn.disabled = Boolean(state.bootstrap.running);
  setupBtn.dataset.state = state.bootstrap.running ? "loading" : "idle";
  setupBtn.querySelector("span").textContent = state.bootstrap.running
    ? state.bootstrap.message || "安裝中"
    : "安裝 TensorRT 執行環境";

  const toast = $("toast");
  if (state.error && connected > 0) {
    toast.dataset.open = "true";
    toast.textContent = state.error;
  } else if (state.bootstrap.message && state.bootstrap.running) {
    toast.dataset.open = "true";
    toast.style.borderLeftColor = "var(--color-accent)";
    toast.textContent = state.bootstrap.message;
  } else {
    toast.dataset.open = "false";
    toast.style.borderLeftColor = "";
  }

  const engine = $("engine-line");
  engine.classList.toggle("is-bad", bad);
  engine.classList.toggle("is-settling", settling);
  engine.classList.toggle("is-compiling", compiling);
  if (!ready) {
    engine.textContent = "還缺 TensorRT 執行環境。安裝後請重新開啟 mpv。";
  } else if (compiling) {
    engine.textContent = "首次編譯 TensorRT 引擎中…可能需要數十秒到數分鐘,請稍候";
  } else if (settling) {
    engine.textContent = "切換中…正在套用新設定,請稍候";
  } else if (bad) {
    const target = player && player.target_fps ? player.target_fps : "";
    engine.textContent = target
      ? `輸出偏低 · 目標 ${target} · 實測 ${outLabel} — 效能不足`
      : "輸出偏低 — 效能不足";
  } else if (connected === 0) {
    engine.textContent = `${modelLabel(state.settings.rife_model)} · TensorRT · CUDA · 等待 mpv`;
  } else {
    engine.textContent = `${modelLabel(state.settings.rife_model)} · TensorRT · CUDA · 場景偵測已就緒`;
  }
}

// Bumped by every settings command. A get_state that was already in flight when
// the user changed something is answered from before that change, so dropping it
// keeps the poll from repainting stale settings over the command's own result.
let commandEpoch = 0;

async function command(name, ...args) {
  commandEpoch += 1;
  const state = await call(name, ...args);
  commandEpoch += 1;
  render(state);
  return state;
}

async function refresh() {
  const epoch = commandEpoch;
  try {
    const state = await call("get_state");
    if (epoch !== commandEpoch) return;
    render(state);
  } catch (err) {
    $("toast").dataset.open = "true";
    $("toast").textContent = String(err);
  }
}

function bind() {
  $("toggle").addEventListener("click", async () => {
    const pressed = $("toggle").getAttribute("aria-pressed") === "true";
    $("toggle").setAttribute("aria-pressed", pressed ? "false" : "true");
    await command("set_enabled", !pressed);
  });

  $("profiles").addEventListener("click", async (event) => {
    const btn = event.target.closest("[data-profile]");
    if (!btn) return;
    renderProfiles(btn.dataset.profile);
    await command("set_profile", btn.dataset.profile);
  });

  $("models").addEventListener("click", async (event) => {
    const btn = event.target.closest("[data-model]");
    if (!btn) return;
    renderModels(Number(btn.dataset.model));
    await command("set_model", Number(btn.dataset.model));
  });

  $("scene").addEventListener("input", () => {
    $("scene-val").textContent = Number($("scene").value).toFixed(2);
  });
  $("scene").addEventListener("change", async () => {
    await command("set_scene", Number($("scene").value));
  });
  $("scene-presets").addEventListener("click", async (event) => {
    const btn = event.target.closest("[data-scene-preset]");
    if (!btn) return;
    const value = Number(btn.dataset.scenePreset);
    renderScenePresets(value);
    await command("set_scene", value);
  });
  $("setup").addEventListener("click", async () => {
    render(await call("start_setup"));
  });
  $("cache-clear").addEventListener("click", async () => {
    render(await call("clear_engine_cache"));
  });
  $("cache-open").addEventListener("click", () => call("open_engine_cache"));
  $("btn-hide").addEventListener("click", () => call("hide"));
  $("btn-min").addEventListener("click", () => call("hide"));
  $("btn-quit").addEventListener("click", () => call("quit"));
}

// Polls unconditionally, including while the window sits hidden in the tray.
//
// v1.4.5 gated this on document.hidden, on the reasoning that the tray is
// this app's normal state and nobody can see the panel there. The gate never
// fired: pywebview's hide() hides the native window without telling the
// WebView2 document anything. Probed directly -- a 100ms counter reports
// document.hidden=false and visibilityState="visible" before hide(), after
// hide(), and after show(), and keeps advancing at the same rate throughout.
//
// Driving it from the Python side would work, but the saving does not justify
// it: an idle get_state() measures 0.200ms, which is 13.4ms of work per minute
// at this interval -- an order of magnitude below the figures that retired the
// other performance items. So the gate is gone rather than reimplemented.
document.addEventListener("DOMContentLoaded", () => {
  bind();
  render(mock);
  refresh();
  setInterval(refresh, 900);
});
