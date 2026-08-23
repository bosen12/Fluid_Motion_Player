const PROFILES = [
  { id: "2x", label: "2×" },
  { id: "3x", label: "3×" },
  { id: "60", label: "60" },
  { id: "120", label: "120" },
  { id: "144", label: "144" },
  { id: "display", label: "螢幕" },
];

const MODELS = [
  { id: 426, label: "4.26" },
  { id: 425, label: "4.25" },
  { id: 46, label: "4.6" },
  { id: 4251, label: "4.25 輕量" },
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
    trt_streams: 2,
    rife_model: 426,
    autostart: false,
    force_accel: false,
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

function renderPlayers(players) {
  const root = $("players");
  if (!players.length) {
    root.innerHTML = `<p class="empty">尚未偵測到 mpv。播放影片後會自動接上。</p>`;
    return;
  }
  root.innerHTML = players
    .map((p) => {
      const res = p.width ? `${p.width}×${p.height}` : "";
      return `<article class="player" data-active="${p.connected && p.interpolation}">
        <div class="player-name"><span>mpv</span><span>${p.connected ? "已連線" : "未連線"}</span></div>
        <div class="player-media">${p.media || res || "pid " + p.pid}</div>
      </article>`;
    })
    .join("");
}

function renderChecks(checks) {
  $("checks").innerHTML = (checks || [])
    .map((c) => `<span class="check" data-ok="${c.ok}">${c.label}</span>`)
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
  const srcEl = $("src-fps");
  const dstEl = $("dst-fps");
  const dstBlock = dstEl.closest(".fps-block");
  const outLabel = !compiling && !settling && player && (player.output_fps || player.estimated_vfps);
  tickNumber(srcEl, player && player.fps ? player.fps : "—");
  tickNumber(dstEl, outLabel ? outLabel : compiling ? "編譯中" : settling || enabled ? "…" : "—");
  const bad = !compiling && Boolean(player && player.fps_ok === false);
  dstBlock.classList.toggle("is-bad", bad);
  dstEl.classList.toggle("is-bad", bad);
  dstEl.setAttribute("aria-invalid", bad ? "true" : "false");

  const toggle = $("toggle");
  setPressed(toggle, enabled);
  toggle.querySelector(".toggle-label").textContent = enabled ? "補幀中" : "未啟用";
  toggle.disabled = !state.runtime.ready && !enabled;
  toggle.dataset.state = state.runtime.ready ? "ready" : "error";

  renderPlayers(state.players || []);
  renderProfiles(state.settings.profile);
  renderModels(state.settings.rife_model);
  renderChecks(state.runtime.checks);

  $("scene").value = state.settings.scene_threshold;
  $("scene-val").textContent = Number(state.settings.scene_threshold).toFixed(2);
  renderScenePresets(state.settings.scene_threshold);
  $("streams").value = state.settings.trt_streams;
  $("streams-val").textContent = String(state.settings.trt_streams);
  $("force-accel").checked = Boolean(state.settings.force_accel);
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
  if (state.error) {
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
  $("streams").addEventListener("input", () => {
    $("streams-val").textContent = $("streams").value;
  });
  $("streams").addEventListener("change", async () => {
    await command("set_streams", Number($("streams").value));
  });
  $("force-accel").addEventListener("change", async () => {
    await command("set_force_accel", $("force-accel").checked);
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

document.addEventListener("DOMContentLoaded", () => {
  bind();
  render(mock);
  refresh();
  setInterval(refresh, 900);
});
