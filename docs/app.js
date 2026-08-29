/* Fluid Motion — page interactions.
 *
 * Two pieces: the pipeline map (click a node, read what that stage does) and
 * the interpolation strip (pick a multiplier, watch where the generated
 * frames land). The strip's arithmetic is the real one -- multi-1 generated
 * frames between every pair of decoded frames, and the output rate is the
 * source rate times the multiplier. No number here is decorative.
 */
(function () {
  "use strict";

  /* ---------- the map ---------- */

  var NOTES = {
    mpv: {
      h: "mpv — 你原本就在用的播放器",
      p: [
        "Fluid Motion 不播放任何東西。它列舉正在執行的行程找 mpv，也認得內嵌 libmpv 的殼（那種行程名稱不叫 mpv.exe，所以改從它開的具名管道反推 pid）。",
        "找到之後，播放仍然完全由 mpv 負責 —— 解碼、送顯示、字幕、音軌，一件都沒有換手。"
      ]
    },
    pipe: {
      h: "named pipe — 指令走的路",
      p: [
        "每個 mpv 實例會開一條自己的具名管道，Fluid Motion 透過它下 JSON IPC 指令：問幀率、問設定目錄、加濾鏡、切解碼模式。",
        "管道名帶著 pid，所以同時開兩個播放器時不會把 A 的指令送去 B。"
      ]
    },
    vf: {
      h: "vf @fluid — 注入的那一段",
      p: [
        "實際下的指令就是這一句：vf add @fluid:vapoursynth=\"~~/shaders/fluid_rife.vpy\":4:1。",
        "最後那個 1 是 concurrent-frames，釘死不能放開 —— RIFE 是時序性的，並行請求會把畫面撕開。標籤 @fluid 讓它隨時能被單獨拆掉，不影響你自己的其他濾鏡。"
      ]
    },
    rife: {
      h: "RIFE + TensorRT — 真正在算的地方",
      p: [
        "vsmlrt 的 RIFE 在 TensorRT 後端上跑（fp16、靜態形狀）。第一次遇到某個解析度要編譯 engine，數十秒到數分鐘；之後同解析度走快取。",
        "VapourSynth 讀不到留在 GPU 上的畫面，所以補幀運作期間會透過 IPC 把解碼切成 copy-back，停止時還原。"
      ]
    },
    out: {
      h: "畫面 — 從 mpv 自己送出",
      p: [
        "補完的幀回到 mpv 的影像鏈，由 mpv 自己送顯示。沒有第二個視窗、沒有轉檔、沒有另一個播放器。",
        "所以 uosc 的控制列、thumbfast 的預覽、你的快捷鍵，全部照常運作。"
      ]
    }
  };

  var nodes = Array.prototype.slice.call(document.querySelectorAll(".node"));
  var note = document.getElementById("mapnote");

  function openNode(btn) {
    var key = btn.dataset.node;
    var data = NOTES[key];
    if (!data) return;
    nodes.forEach(function (n) { n.setAttribute("aria-pressed", String(n === btn)); });
    note.innerHTML = "";
    var h = document.createElement("h3");
    h.textContent = data.h;
    note.appendChild(h);
    data.p.forEach(function (text) {
      var p = document.createElement("p");
      p.textContent = text;
      note.appendChild(p);
    });
  }

  nodes.forEach(function (btn) {
    btn.addEventListener("click", function () { openNode(btn); });
  });

  /* ---------- the interpolation strip ---------- */

  var SOURCE_FPS = 24000 / 1001;   // 23.976 -- the rate target_multi() falls back to
  var SOURCE_FRAMES = 6;

  var strip = document.getElementById("strip");
  var segs = Array.prototype.slice.call(document.querySelectorAll(".seg"));
  var playBox = document.getElementById("play");
  var rSrc = document.getElementById("r-src");
  var rOut = document.getElementById("r-out");
  var rMade = document.getElementById("r-made");

  var multi = 2;
  var ticks = [];
  var head = 0;
  var raf = null;
  var last = 0;

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  function build() {
    strip.innerHTML = "";
    ticks = [];
    for (var i = 0; i < SOURCE_FRAMES; i++) {
      addTick("src");
      if (i < SOURCE_FRAMES - 1) {
        for (var g = 0; g < multi - 1; g++) addTick("made");
      }
    }
    head = 0;
    paint();
  }

  function addTick(kind) {
    var el = document.createElement("div");
    el.className = "tick tick--" + kind;
    el.dataset.lit = "false";
    strip.appendChild(el);
    ticks.push(el);
  }

  function paint() {
    for (var i = 0; i < ticks.length; i++) {
      ticks[i].dataset.lit = String(i === head);
    }
  }

  function readout() {
    var out = SOURCE_FPS * multi;
    var made = out - SOURCE_FPS;
    rSrc.textContent = SOURCE_FPS.toFixed(3);
    rOut.textContent = out.toFixed(2);
    rMade.textContent = made.toFixed(2);
  }

  function step(now) {
    if (!last) last = now;
    if (now - last > 190) {
      last = now;
      head = (head + 1) % ticks.length;
      paint();
    }
    raf = window.requestAnimationFrame(step);
  }

  function setPlaying(on) {
    if (raf) { window.cancelAnimationFrame(raf); raf = null; }
    if (on && !reduced.matches) {
      last = 0;
      raf = window.requestAnimationFrame(step);
    } else {
      // Stopped: light every decoded frame so the shape still reads.
      for (var i = 0; i < ticks.length; i++) {
        ticks[i].dataset.lit = String(ticks[i].classList.contains("tick--src"));
      }
    }
  }

  if (strip && segs.length) {
    segs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        multi = Number(btn.dataset.multi);
        segs.forEach(function (b) { b.setAttribute("aria-pressed", String(b === btn)); });
        build();
        readout();
        setPlaying(playBox.checked);
      });
    });

    if (playBox) {
      playBox.addEventListener("change", function () { setPlaying(playBox.checked); });
      if (reduced.matches) playBox.checked = false;
    }

    build();
    readout();
    setPlaying(playBox ? playBox.checked : false);

    // Stop the loop while the tab is in the background.
    document.addEventListener("visibilitychange", function () {
      setPlaying(document.visibilityState === "visible" && playBox && playBox.checked);
    });
  }

  /* ---------- latest tag ---------- */

  fetch("https://api.github.com/repos/bosen12/Fluid_Motion_Player/releases/latest", {
    headers: { Accept: "application/vnd.github+json" }
  })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || !data.tag_name) return;
      var chip = document.getElementById("ver-chip");
      if (chip) chip.textContent = data.tag_name;
    })
    .catch(function () { /* offline or rate-limited: keep the static value */ });
})();
