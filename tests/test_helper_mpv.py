"""thumbfast's own mpv is not a player to inject into.

An operator sweep over the comparisons in this package killed 117 of 131
mutants. `_is_helper_mpv`'s two membership tests were among the survivors:
either one could be inverted and nothing went red.

They matter because a hover preview spawns a second mpv. Treat that as a
player and the watcher connects to it, sets hwdec to copy-back, and adds a
RIFE filter to a process whose entire job is to decode one frame and exit --
while the real player sits next to it. AX Player's sidebar produces one of
these per row hovered.
"""
import types

from fluid_motion.core.mpv_detect import _is_helper_mpv


def _proc(cmdline, *, raises=None):
    """Shaped like the psutil object process_iter hands over.

    `info` is the dict process_iter fills; `cmdline()` is the fallback call for
    the fields it was not asked for.
    """

    def cmdline_call():
        if raises is not None:
            raise raises
        return cmdline

    return types.SimpleNamespace(info={"cmdline": cmdline}, cmdline=cmdline_call)


def test_a_thumbfast_helper_is_not_a_player():
    assert _is_helper_mpv(_proc([r"C:\mpv\mpv.exe", "--script=thumbfast.lua", "clip.mkv"]))


def test_a_scriptless_helper_is_not_a_player_either():
    """--load-scripts=no is the other shape: no scripts means no
    zz-fluid-ipc.lua, so it could never be driven anyway."""
    assert _is_helper_mpv(_proc([r"C:\mpv\mpv.exe", "--load-scripts=no", "clip.mkv"]))


def test_a_real_player_is_not_mistaken_for_a_helper():
    """The half that matters more. A guard that answered True for everything
    would leave the watcher with no players at all -- and the panel would just
    say 尚未偵測到 mpv, which looks exactly like nothing being open."""
    assert not _is_helper_mpv(_proc([r"C:\mpv\mpv.exe", r"D:\Anime\ep01.mkv"]))
    assert not _is_helper_mpv(_proc([r"C:\Program Files\mpv\mpv.exe"]))


def test_a_process_that_will_not_say_is_treated_as_a_player():
    """cmdline() raises for a process that exited between the scan and the
    read, and for one this user may not query. Answering True there would drop
    a real player silently; answering False costs at most one failed connect,
    which the tick already handles.
    """
    import psutil

    assert not _is_helper_mpv(_proc(None, raises=psutil.AccessDenied()))
    assert not _is_helper_mpv(_proc(None, raises=OSError("gone")))
