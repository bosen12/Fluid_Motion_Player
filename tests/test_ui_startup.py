"""The panel must not paint the design mock inside pywebview.

pywebview injects window.pywebview.api a moment after DOMContentLoaded --
measured in WebView2, absent at DOMContentLoaded every time and callable
13-114 ms later. app.js used to render(mock) from its DOMContentLoaded handler
and let call() answer with the mock whenever the bridge was missing, and the
next thing to ask was the 900 ms poll. Recorded against the real UI in real
WebView2, on a machine with the runtime installed and interpolation enabled:

    +54 ms   還缺 TensorRT 執行環境 / install button visible / toggle 未啟用
    +961 ms  the real state

With the fix the real state lands at +133 ms, ahead of WebView2's first paint
(~232 ms), so no frame of the mock is ever shown.

These run the real app.js under node with a stubbed window and document, and
drive the startup sequence itself rather than matching strings: a guard that
reads source text is exactly what §9.17 warned about.
"""
import json
import shutil
import subprocess

import pytest

from fluid_motion.paths import ui_dir

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node to run app.js")

# Runs app.js in a vm context. `render` and `bind` are replaced after load --
# they are plain global functions, and the DOMContentLoaded handler looks them
# up at call time -- so the sequence under test is app.js's own.
HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const appJs = process.argv[process.argv.length - 1];

function load({ embedded, readyAtLoad }) {
  const listeners = {};
  const docListeners = {};
  const win = { addEventListener: (t, f) => (listeners[t] = listeners[t] || []).push(f) };
  if (embedded) win.chrome = { webview: {} };
  const bridge = { get_state: async () => ({ real: true }) };
  if (readyAtLoad) win.pywebview = { api: bridge };
  const doc = {
    addEventListener: (t, f) => (docListeners[t] = docListeners[t] || []).push(f),
    getElementById: () => null,
  };
  const ctx = { window: win, document: doc, console, setInterval: () => 0, Promise };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(appJs, "utf8"), ctx);
  const mock = vm.runInContext("mock", ctx);
  const renders = [];
  ctx.bind = () => {};
  ctx.render = (state) => renders.push(state === mock ? "mock" : state && state.real ? "real" : "other");
  const ready = () => {
    win.pywebview = { api: bridge };
    for (const f of listeners["pywebviewready"] || []) f();
  };
  const start = () => {
    for (const f of docListeners["DOMContentLoaded"] || []) f();
  };
  return { ctx, renders, ready, start, mock };
}

const tick = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const out = {};

  // Inside WebView2, bridge not there yet: nothing may paint until it is.
  {
    const s = load({ embedded: true, readyAtLoad: false });
    s.start();
    await tick(30);
    out.embedded_before_ready = [...s.renders];
    const click = s.ctx.call("get_state");
    let clickAnswer = "pending";
    click.then((v) => (clickAnswer = v === s.mock ? "mock" : v.real ? "real" : "other"));
    await tick(30);
    out.click_before_ready = clickAnswer;
    s.ready();
    await tick(30);
    out.embedded_after_ready = [...s.renders];
    out.click_after_ready = clickAnswer;
  }

  // Bridge already injected when the script runs: straight to real state.
  {
    const s = load({ embedded: true, readyAtLoad: true });
    s.start();
    await tick(30);
    out.embedded_ready_at_load = [...s.renders];
  }

  // A plain browser previewing the design: the mock is the whole point there.
  {
    const s = load({ embedded: false, readyAtLoad: false });
    s.start();
    await tick(30);
    out.plain_browser = [...s.renders];
  }

  process.stdout.write(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def startup():
    proc = subprocess.run(
        [NODE, "-e", HARNESS, "--", str(ui_dir() / "app.js")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_nothing_is_painted_inside_webview2_before_the_bridge_arrives(startup):
    assert startup["embedded_before_ready"] == [], (
        "the panel painted before it had any real state -- that frame is the mock"
    )


def test_the_first_paint_inside_webview2_is_real_state(startup):
    assert startup["embedded_after_ready"] == ["real"]


def test_a_click_before_the_bridge_arrives_waits_instead_of_being_lost(startup):
    """Answering it with the mock would drop the user's action and repaint
    fake state over the panel."""
    assert startup["click_before_ready"] == "pending"
    assert startup["click_after_ready"] == "real"


def test_a_bridge_that_is_already_there_is_used_at_once(startup):
    assert startup["embedded_ready_at_load"] == ["real"]


def test_the_design_preview_in_a_plain_browser_still_gets_the_mock(startup):
    """No chrome.webview, no pywebview: the mock is what makes the page
    previewable at all (the fm-ui launch config serves it over http.server)."""
    assert startup["plain_browser"] == ["mock"]
