"""Quitting must take the app's child processes with it.

quit_app ends in os._exit(0) -- it has to, or pywebview and pystray leave a
headless process with no tray icon behind. os._exit joins no threads and runs
no finally blocks, and on Windows a child does not die with its parent. The
bootstrap thread is a daemon blocked inside run_hidden waiting on 7z, and
engine.stop() does not reach it: reproduced on this code path, parent gone,
extraction still running.

The wasted work is the smaller half. install_runtime's "one at a time" guard
is a per-process flag whose own comment reasons about "a second 3.5 GB
download into the same cache" extracting "over the first" -- it cannot see an
orphan from the *previous* session, so the next launch starts a second
extraction into the directory the first one is still writing.

run_hidden was subprocess.run, which keeps its Popen to itself; there is no
way to reach the child of a call still blocked inside it. It is now run()'s
own body with the handle registered, so the second half of this file pins the
parts of run()'s contract that are now this module's to keep.
"""
import subprocess
import sys
import threading
import time

import pytest

from fluid_motion.core import proc


def _spawn(seconds: float) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: proc.run_hidden(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"]
        ),
        daemon=True,
    )
    thread.start()
    return thread


def _wait_for_registration(count: int = 1, limit: float = 10.0) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        with proc._live_lock:
            if len(proc._live) >= count:
                return
        time.sleep(0.02)
    pytest.fail(f"no child registered within {limit}s")


def test_a_running_child_is_reachable_while_it_runs():
    """Guards the guard: if nothing is ever registered, the kill below reports
    zero and passes while every child survives."""
    thread = _spawn(30)
    try:
        _wait_for_registration()
        with proc._live_lock:
            assert len(proc._live) == 1
    finally:
        proc.terminate_children()
        thread.join(timeout=10)


def test_quitting_kills_a_child_that_is_still_running():
    thread = _spawn(60)
    _wait_for_registration()

    killed = proc.terminate_children()

    assert killed == 1, "the child was not killed"
    thread.join(timeout=10)
    assert not thread.is_alive(), (
        "run_hidden is still blocked -- the child outlived the kill, which is "
        "what os._exit would then orphan"
    )
    with proc._live_lock:
        assert not proc._live, "a dead child is still registered"


def test_a_finished_child_does_not_stay_registered():
    """A leaked entry per call would grow without bound in a process that runs
    nvidia-smi every few seconds, and terminate_children would walk all of it."""
    before = len(proc._live)
    proc.run_hidden([sys.executable, "-c", "pass"])
    assert len(proc._live) == before


# -- the parts of subprocess.run's contract this module now owns -------------


def test_output_is_captured_and_decoded_as_utf8():
    # The child writes UTF-8 bytes to the buffer instead of using print(), which
    # encodes with the child's own locale codepage -- cp1252 on the CI runner,
    # cp950 on a zh-TW box, and neither can encode U+0308. The child died of
    # UnicodeEncodeError there, so the test failed on the child's encoder rather
    # than on run_hidden's decoder, which is the thing under test.
    script = "import sys; sys.stdout.buffer.write('gefo\\u0308rce'.encode())"
    result = proc.run_hidden([sys.executable, "-c", script])
    assert result.returncode == 0
    assert "geförce" in result.stdout
    assert result.stderr == ""


def test_a_nonzero_exit_is_reported_rather_than_raised_by_default():
    result = proc.run_hidden([sys.executable, "-c", "raise SystemExit(3)"])
    assert result.returncode == 3


def test_check_still_raises():
    with pytest.raises(subprocess.CalledProcessError):
        proc.run_hidden([sys.executable, "-c", "raise SystemExit(3)"], check=True)


def test_timeout_still_raises_and_kills_the_child():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        proc.run_hidden([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert time.monotonic() - started < 15, "the timeout did not fire"
    with proc._live_lock:
        assert not proc._live, "the timed-out child is still registered"


def test_quit_app_reaps_before_it_calls_os_exit():
    """The check above pins terminate_children, not that quitting calls it.

    A mutation sweep found exactly that gap: replacing the call in quit_app
    with `killed = 0` restored the original bug and every test above stayed
    green -- HANDOFF §9.39's shape, the guard standing one layer below the
    thing it is guarding.

    quit_app is a closure inside main() that ends in os._exit(0), so it cannot
    be called from a test at all. Read out of the source instead, the way
    test_public_surfaces derives the backend list rather than restating it:
    what matters is that the reap is in there and that it happens *before* the
    exit, since os._exit runs nothing after itself.
    """
    import ast

    import fluid_motion.app as app_mod
    from pathlib import Path

    tree = ast.parse(Path(app_mod.__file__).read_text(encoding="utf-8"))
    quit_fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "quit_app"),
        None,
    )
    assert quit_fn is not None, "quit_app is gone or was renamed"

    reap_at = exit_at = None
    for node in ast.walk(quit_fn):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name == "terminate_children":
            reap_at = node.lineno
        elif name == "_exit":
            # The *first* one. Anything after it is unreachable, and taking the
            # last let a mutation that inserted an os._exit ahead of the reap
            # go through: reap_at was still below the later exit's line.
            exit_at = node.lineno if exit_at is None else min(exit_at, node.lineno)

    assert exit_at is not None, "quit_app no longer ends in os._exit"
    assert reap_at is not None, (
        "quit_app does not reap its children; os._exit joins no threads, so a "
        "7z left running by the bootstrap thread outlives the app"
    )
    assert reap_at < exit_at, "the reap is after os._exit, which never returns"


def test_the_names_quit_app_reaps_with_are_actually_bound():
    """The AST check above sees a *call*; it cannot see whether the name
    resolves.

    A mutation sweep deleted `from fluid_motion.core.proc import
    terminate_children` and the whole suite stayed green. At runtime that is a
    NameError inside quit_app -- an Exception, caught by the handler that
    exists so quitting can never be blocked -- so the reap silently does
    nothing and the orphan comes back exactly as before.

    Binding is checked here, and the handler now logs rather than passing, so
    the same class of breakage leaves a trace instead of a silence.
    """
    import fluid_motion.app as app_mod
    from fluid_motion.core import proc as proc_mod
    from fluid_motion import log as log_mod

    assert getattr(app_mod, "terminate_children", None) is proc_mod.terminate_children, (
        "quit_app's terminate_children does not resolve to the real one"
    )
    assert getattr(app_mod, "log_exc", None) is log_mod.log_exc, (
        "the reap's failure path cannot report anything"
    )


def test_the_reaps_failure_path_does_not_swallow_in_silence():
    """The handler is broad on purpose -- quitting must never be blockable --
    which is exactly why it must not be a bare `pass`.

    A mutation sweep deleted the terminate_children import and the suite
    stayed green. At runtime that is a NameError inside quit_app, an Exception,
    caught here and dropped: the reap silently does nothing and the orphan is
    back with no trace anywhere.

    Checked in the source, like the ordering above, because quit_app cannot be
    called. The first attempt at this test rebuilt the try/except in the test
    body and asserted over its own log call -- it passed with the handler
    reverted to `pass`, which is the tautology CLAUDE.md's contact-sheet
    example warns about, so it was deleted rather than kept.
    """
    import ast
    from pathlib import Path

    import fluid_motion.app as app_mod

    tree = ast.parse(Path(app_mod.__file__).read_text(encoding="utf-8"))
    quit_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "quit_app"
    )
    guarded = [
        node for node in ast.walk(quit_fn)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and getattr(call.func, "id", getattr(call.func, "attr", "")) == "terminate_children"
            for call in ast.walk(node)
        )
    ]
    assert len(guarded) == 1, f"expected one guarded reap, found {len(guarded)}"

    for handler in guarded[0].handlers:
        assert not all(isinstance(stmt, ast.Pass) for stmt in handler.body), (
            "the reap's failure path is a bare pass: a NameError there looks "
            "exactly like a successful reap that had nothing to kill"
        )
