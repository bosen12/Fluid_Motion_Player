"""The player-specific half of the pipeline: which directory decides.

The filter is written into, and loaded from, the config dir of the player
that will run it -- which is not necessarily settings.mpv_root. Everything
here is a consequence of those two having been allowed to disagree.
"""
from pathlib import Path

import pytest

from fluid_motion.config import Settings
from fluid_motion.core.mpv_detect import PlayerProcess, display_label
from fluid_motion.core.runtime import Check, RuntimeStatus, missing_labels

PLAYER_ROOT = Path("Z:/player-runtime")
CONFIGURED_ROOT = Path("Z:/configured-mpv")


def _status(root: Path, *, ready: bool) -> RuntimeStatus:
    ids = ("mpv", "vapoursynth", "scdetect", "vsmlrt", "tensorrt", "python")
    checks = [Check(name, name.title(), ready) for name in ids]
    checks += [Check(f"rife{n}", f"RIFE {n}", ready) for n in (46, 425, 426)]
    return RuntimeStatus(mpv_root=str(root), ready=ready, checks=checks)


class _FakeIpc:
    def __init__(self, props: dict | None = None):
        self.props = dict(props or {})
        self.commands: list[tuple] = []
        self.vf: list = []
        self.path = "fake-pipe"

    def close(self):
        pass

    def get(self, name, *, timeout=None):
        if name == "vf":
            return self.vf
        return self.props.get(name)

    def set(self, name, value):
        self.props[name] = value

    def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[:2] == ("vf", "add"):
            self.vf = [{"name": "vapoursynth", "label": "fluid"}]
        elif args[:2] == ("vf", "remove"):
            self.vf = []


@pytest.fixture()
def engine_with_player(monkeypatch, tmp_path):
    """One connected player whose own config dir is PLAYER_ROOT."""
    from fluid_motion.core import watcher as watcher_mod

    def _build(*, player_ready: bool, enabled: bool = True, vf=None):
        settings = Settings(enabled=enabled, profile="2x", mpv_root=str(CONFIGURED_ROOT))
        engine = watcher_mod.Engine(settings)
        ipc = _FakeIpc({"container-fps": 24, "estimated-vf-fps": 24, "display-fps": 60})
        if vf is not None:
            ipc.vf = vf
        monkeypatch.setattr(
            watcher_mod,
            "iter_mpv_processes",
            lambda: [PlayerProcess(pid=99, name="AXPlayer.exe", label="AX Player")],
        )
        monkeypatch.setattr(watcher_mod, "connect_pid", lambda p, extra=None, **kw: ipc)
        monkeypatch.setattr(watcher_mod, "player_config_dir", lambda _ipc: PLAYER_ROOT)
        # The configured root is fully installed; the player's own is not.
        monkeypatch.setattr(
            watcher_mod,
            "diagnose",
            lambda root, **_kw: _status(Path(root), ready=(Path(root) != PLAYER_ROOT or player_ready)),
        )
        monkeypatch.setattr(watcher_mod, "engine_cache_info", lambda: engine._engine_cache)
        monkeypatch.setattr(watcher_mod, "install_lua", lambda root: root)
        monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda root: None)
        engine._runtime = _status(CONFIGURED_ROOT, ready=True)
        return engine, ipc

    return _build


def _record_apply(monkeypatch):
    from fluid_motion.core import watcher as watcher_mod

    applied: list[Path] = []

    def fake_apply(ipc, settings, mpv_root, *, announce=False, info=None, pid=None, **_kw):
        applied.append(Path(mpv_root))
        ipc.command("vf", "add", "@fluid:vapoursynth")
        return Path(mpv_root) / "shaders" / "fluid_rife.vpy"

    monkeypatch.setattr(watcher_mod, "apply", fake_apply)
    return applied


# -- the gate -------------------------------------------------------------
def test_a_player_whose_own_config_dir_is_not_ready_is_never_filtered(
    monkeypatch, engine_with_player
):
    """settings.mpv_root being ready says nothing about the directory the
    filter is actually loaded from. Pushing it anyway is not a worse error
    message, it is a dead video: mpv accepts the vf, then fails to construct
    the filter ("Failed to load VapourSynth VSScript library") and ends up
    with no video or audio stream selected -- observed against a live player.
    """
    engine, ipc = engine_with_player(player_ready=False)
    applied = _record_apply(monkeypatch)

    engine.tick()

    assert engine._runtime.ready is True, "the configured root is still ready"
    assert applied == [], "the filter was pushed into a player that cannot load it"
    assert ipc.vf == []


def test_that_player_is_reported_with_the_directory_and_what_it_lacks(
    monkeypatch, engine_with_player
):
    engine, _ = engine_with_player(player_ready=False)
    _record_apply(monkeypatch)

    engine.tick()

    player = engine._players[0]
    assert player.ready is False
    assert player.config_dir == str(PLAYER_ROOT)
    assert player.missing, "a blocked player has to say what it is missing"


def test_a_ready_player_is_still_filtered_from_its_own_directory(
    monkeypatch, engine_with_player
):
    engine, _ = engine_with_player(player_ready=True)
    applied = _record_apply(monkeypatch)

    engine.tick()

    assert applied == [PLAYER_ROOT], "applied against the wrong directory"
    assert engine._players[0].ready is True


def test_a_leftover_filter_is_taken_off_a_player_that_cannot_run_it(
    monkeypatch, engine_with_player
):
    """The state today's bug leaves behind: @fluid on the graph of a player
    with no runtime. It has to come off, not be left there."""
    engine, ipc = engine_with_player(
        player_ready=False, vf=[{"name": "vapoursynth", "label": "fluid"}]
    )
    _record_apply(monkeypatch)

    engine.tick()

    assert ipc.vf == []


def test_enabling_names_the_player_that_cannot_take_it(monkeypatch, engine_with_player):
    engine, ipc = engine_with_player(player_ready=False, enabled=False)
    applied = _record_apply(monkeypatch)
    monkeypatch.setattr("fluid_motion.core.watcher.save_settings", lambda s: None)
    engine.tick()  # learn the player's readiness

    engine.set_enabled(True)

    assert applied == []
    assert "AX Player" in engine._error


# -- installing the IPC script into the player's own dir -------------------
def test_the_ipc_script_is_installed_into_the_players_own_config_dir(monkeypatch, tmp_path):
    """start() only ever installed it into settings.mpv_root, so a player
    running out of anywhere else never got it: no F3 binding (AX Player's own
    toolbar button sends F3, and a live instance reported no F3 binding at
    all), no post-seek hold-off, no stale-filter cleanup."""
    from fluid_motion.core import watcher as watcher_mod

    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path / "configured")))
    root = tmp_path / "player"
    installed: list[Path] = []
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: installed.append(Path(r)))
    monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda r: None)

    engine._ensure_player_scripts(root, pid=7)

    assert installed == [root]
    assert 7 in engine._needs_restart, "mpv only loads scripts at launch"


def test_an_out_of_date_script_is_replaced(monkeypatch, tmp_path):
    """This assertion used to read the other way -- "an existing script is left
    alone" -- and that froze every config dir on whichever version of the
    script it first received.

    The script is not static: six revisions between v1.0.0 and v1.4.5, and the
    v1.0.0 one is 192 characters against today's 4742. Among what it lacks is
    the per-player seek_hold file (d0b1e76); without that, a seek in one player
    tears the filter off every other connected player. Meanwhile start() has
    always rewritten settings.mpv_root's copy on every launch -- so the only
    directory that stayed current was the one AX Player does not use.
    """
    from fluid_motion.core import watcher as watcher_mod

    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path / "configured")))
    root = tmp_path / "player"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "zz-fluid-ipc.lua").write_text("-- an older build's", encoding="utf-8")
    installed: list[Path] = []
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: installed.append(Path(r)))
    monkeypatch.setattr(watcher_mod, "ensure_input_binding", lambda r: None)

    engine._ensure_player_scripts(root, pid=7)

    assert installed == [root], "a stale script stayed stale"
    assert 7 in engine._needs_restart, "mpv only loads scripts at launch"


def test_a_script_that_is_already_current_is_not_rewritten(monkeypatch, tmp_path):
    """The other half, and the one with the trap in it.

    install_lua writes through write_text, so on Windows the file on disk gets
    \\r\\n where the packaged resource has \\n -- 4932 bytes against 4764, for
    text that is identical. Compare the two as bytes and every check answers
    "differs": the script is rewritten every session and needs_restart is
    raised forever, so the panel permanently asks the user to restart a player
    that is already running exactly the right script.
    """
    from fluid_motion.core import watcher as watcher_mod
    from fluid_motion.core.bootstrap import install_lua

    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path / "configured")))
    root = tmp_path / "player"
    install_lua(root)  # the real writer, so the line endings are the real ones
    on_disk = (root / "scripts" / "zz-fluid-ipc.lua").read_bytes()

    installed: list[Path] = []
    monkeypatch.setattr(watcher_mod, "install_lua", lambda r: installed.append(Path(r)))

    engine._ensure_player_scripts(root, pid=7)

    assert installed == [], "rewrote a script that was already current"
    assert 7 not in engine._needs_restart, "asked for a restart with nothing to load"
    assert (root / "scripts" / "zz-fluid-ipc.lua").read_bytes() == on_disk


# -- naming the player -----------------------------------------------------
@pytest.mark.parametrize(
    "name, exe, expected",
    [
        ("mpv.exe", "", "mpv"),
        ("AXPlayer.exe", "", "AX Player"),
        ("mpvnet.exe", "", "mpv.net"),
        ("python.exe", r"C:\Python310\python.exe", "內嵌播放器"),
        ("embedded-mpv", "", "內嵌播放器"),
    ],
)
def test_a_player_is_called_what_it_actually_is(name, exe, expected):
    """Every card said "mpv", which is wrong for an embedded host and leaves
    two open players looking identical."""
    assert display_label(name, exe) == expected


# -- what a blocked player is missing --------------------------------------
def test_the_three_model_checks_read_as_one_missing_thing():
    checks = [
        Check("vapoursynth", "VapourSynth", False),
        Check("rife46", "RIFE 4.6", False),
        Check("rife425", "RIFE 4.25", False),
        Check("rife426", "RIFE 4.26", False),
    ]
    status = RuntimeStatus(mpv_root="Z:/x", ready=False, checks=checks)

    assert missing_labels(status) == ["VapourSynth", "RIFE 模型"]


def test_one_model_present_is_not_missing():
    checks = [
        Check("vapoursynth", "VapourSynth", True),
        Check("rife46", "RIFE 4.6", False),
        Check("rife425", "RIFE 4.25", True),
        Check("rife426", "RIFE 4.26", False),
    ]
    status = RuntimeStatus(mpv_root="Z:/x", ready=True, checks=checks)

    assert missing_labels(status) == []


# -- copy-back decoding surviving a crash ----------------------------------
def test_a_saved_decode_mode_outlives_the_process_that_saved_it(tmp_path, monkeypatch):
    """The map only ever lived in memory. The lua strips a stale @fluid when
    this app stops answering its heartbeat, but it has no idea what hwdec was
    before and never touches it -- so a Fluid Motion killed while
    interpolating left the player on copy-back for the rest of its life,
    paying the GPU->CPU transfer for a filter that is no longer there."""
    from fluid_motion.core import inject

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    monkeypatch.setattr(inject, "_process_started", lambda pid: None)
    ipc = _FakeIpc({"hwdec": "auto-safe"})

    inject.ensure_copyback_hwdec(ipc, pid=1234)

    assert ipc.props["hwdec"] == "auto-copy"
    # A fresh process (empty in-memory map) can still find it.
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    assert inject.stranded_hwdec() == {1234: "auto-safe"}


def test_a_reused_pid_does_not_get_somebody_elses_decode_mode(tmp_path, monkeypatch):
    from fluid_motion.core import inject

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    monkeypatch.setattr(inject, "_process_started", lambda pid: 1000.0)
    inject.ensure_copyback_hwdec(_FakeIpc({"hwdec": "auto-safe"}), pid=1234)

    # Same pid, different process.
    monkeypatch.setattr(inject, "_process_started", lambda pid: 9999.0)
    assert inject.stranded_hwdec() == {}


def test_restoring_clears_the_record(tmp_path, monkeypatch):
    from fluid_motion.core import inject

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    monkeypatch.setattr(inject, "_process_started", lambda pid: None)
    ipc = _FakeIpc({"hwdec": "auto-safe"})
    inject.ensure_copyback_hwdec(ipc, pid=1234)

    inject.restore_hwdec(ipc, pid=1234)

    assert ipc.props["hwdec"] == "auto-safe"
    assert inject.stranded_hwdec() == {}


def test_a_player_still_filtering_keeps_its_saved_mode_rather_than_losing_it(
    tmp_path, monkeypatch
):
    """Startup recovery must not throw the record away for a player that is
    legitimately still on copy-back -- that would strand it later instead."""
    from fluid_motion.core import inject, watcher as watcher_mod

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    monkeypatch.setattr(inject, "_process_started", lambda pid: None)
    inject.ensure_copyback_hwdec(_FakeIpc({"hwdec": "auto-safe"}), pid=1234)
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})  # a fresh process

    still_filtering = _FakeIpc({"hwdec": "auto-copy"})
    still_filtering.vf = [{"name": "vapoursynth", "label": "fluid"}]
    monkeypatch.setattr(watcher_mod, "connect_pid", lambda p, **kw: still_filtering)
    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path)))

    engine._recover_stranded_hwdec()

    assert still_filtering.props["hwdec"] == "auto-copy", "restored while still needed"
    assert inject.stranded_hwdec() == {1234: "auto-safe"}, "the only record was discarded"


def test_a_stranded_player_is_put_back_when_the_app_returns(tmp_path, monkeypatch):
    from fluid_motion.core import inject, watcher as watcher_mod

    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})
    monkeypatch.setattr(inject, "_process_started", lambda pid: None)
    inject.ensure_copyback_hwdec(_FakeIpc({"hwdec": "auto-safe"}), pid=1234)
    monkeypatch.setattr(inject, "_PREV_HWDEC", {})

    stranded = _FakeIpc({"hwdec": "auto-copy"})  # lua already stripped the filter
    monkeypatch.setattr(watcher_mod, "connect_pid", lambda p, **kw: stranded)
    engine = watcher_mod.Engine(Settings(mpv_root=str(tmp_path)))

    engine._recover_stranded_hwdec()

    assert stranded.props["hwdec"] == "auto-safe"
    assert inject.stranded_hwdec() == {}


def test_every_connected_player_gets_the_script_check(monkeypatch, engine_with_player):
    """Not just settings.mpv_root at start-up: the check has to run against
    whatever directory each connected player actually reports."""
    from fluid_motion.core import watcher as watcher_mod

    engine, _ = engine_with_player(player_ready=True)
    _record_apply(monkeypatch)
    checked: list[tuple] = []
    monkeypatch.setattr(
        watcher_mod.Engine,
        "_ensure_player_scripts",
        lambda self, root, pid: checked.append((Path(root), pid)),
    )

    engine.tick()

    assert checked == [(PLAYER_ROOT, 99)]


# -- an unreadable vf is not an absent filter ------------------------------
def test_a_player_whose_vf_cannot_be_read_is_left_alone(monkeypatch, engine_with_player):
    """`vf_is_fluid(None)` is False, which reads exactly like "no filter".

    The mpv that cannot answer is the one busy compiling a TensorRT engine, so
    treating silence as "the filter fell off" sends it a fresh vf add every
    APPLY_RETRY_BACKOFF -- precisely when it can least afford one. The control
    for this is test_a_ready_player_is_still_filtered_from_its_own_directory:
    same fixture, same readiness, vf readable and empty, and it *does* apply.
    """
    from fluid_motion.core.mpv_ipc import IpcError

    engine, ipc = engine_with_player(player_ready=True)
    applied = _record_apply(monkeypatch)

    def _vf_is_unreadable(name, *, timeout=None):
        if name == "vf":
            raise IpcError("mpv IPC timed out")
        return ipc.props.get(name)

    ipc.get = _vf_is_unreadable

    engine.tick()

    assert applied == [], "re-applied on the strength of a reading that never arrived"
    assert ipc.commands == [], "a vf command went to an mpv that was not answering"


# -- a pipe is not proof of a player --------------------------------------
def test_a_pipe_naming_a_dead_pid_does_not_invent_a_player(monkeypatch):
    """Embedded hosts are discovered from named pipes rather than a process
    name, and the fallback for "the process is there but will not say what it
    is" was reused for "the process is not there", inventing a player.

    Not only a wrong card in the UI. tick() derives allow_ambiguous from the
    player count, so one phantom entry takes a genuinely single-player machine
    off the ambiguous pipe names -- and for an mpv.conf that sets
    input-ipc-server=mpvsocket those are the only way in at all.
    """
    from fluid_motion.core import mpv_detect

    monkeypatch.setattr(mpv_detect, "list_win_pipes", lambda: ["fluid-mpv-999999"])
    monkeypatch.setattr(mpv_detect, "psutil", mpv_detect.psutil)

    assert mpv_detect.iter_mpv_processes() == []


def test_a_pipe_naming_a_live_process_is_still_a_player(monkeypatch):
    """The control: the discovery this protects has to keep working."""
    import os

    from fluid_motion.core import mpv_detect

    monkeypatch.setattr(
        mpv_detect, "list_win_pipes", lambda: [f"fluid-mpv-{os.getpid()}"]
    )

    found = mpv_detect.iter_mpv_processes()

    assert [p.pid for p in found] == [os.getpid()]


def test_a_foreign_pipe_name_cannot_stop_player_discovery(monkeypatch):
    """_embedded_player_pids walks the machine's whole pipe namespace.

    It keeps whatever matches by name, so the segment it converts to a pid is
    not the app's string. str.isdigit() is true for 128 characters int()
    rejects -- ①②③, ⑴⑵⑶, superscripts -- and a ValueError here leaves
    iter_mpv_processes entirely: every tick would fail the same way for as
    long as that pipe existed, so interpolation would stop over a name the
    scan merely walked past.
    """
    import os

    from fluid_motion.core import mpv_detect

    real = f"fluid-mpv-{os.getpid()}"
    monkeypatch.setattr(
        mpv_detect, "list_win_pipes", lambda: [r"mpvSockets\①", "fluid-mpv-②", real]
    )

    pids = mpv_detect._embedded_player_pids(set())

    assert pids == {os.getpid()}, "the real player must still be found beside them"
