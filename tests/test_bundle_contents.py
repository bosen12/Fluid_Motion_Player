"""What the release bundle leaves out, and why leaving it out is safe.

FluidMotion.spec excludes numpy. Nothing in this app, pywebview or pystray
imports it; it came in through PIL._typing, which imports numpy.typing under
`if TYPE_CHECKING:` -- never true at runtime, but PyInstaller's analysis does
not evaluate the condition -- and numpy's submodules then dragged in yaml and
charset_normalizer. The onefile went from 38.1 MB to 27.2 MB.

PIL itself stays: icon.py draws the tray icon with it, app.py opens it, and
pystray serialises it to ICO. That is what makes this exclude worth a test.
tray() treats ImportError as "no tray available" and returns quietly, so a
PIL that could not load without numpy would not crash anything -- the app
would start, and the tray icon would simply never appear. The source tree has
numpy installed, so no checkout and no other test could see it happen.
"""
import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import fluid_motion


def _repo() -> Path:
    # Through the package, never the cwd: two sibling projects live on this
    # machine, and a cwd-relative read once passed against the other one's file.
    return Path(fluid_motion.__file__).resolve().parent.parent


def _spec_excludes() -> list[str]:
    tree = ast.parse((_repo() / "FluidMotion.spec").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "excludes" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("FluidMotion.spec has no top-level `excludes = [...]`")


def _all_imports(source: str) -> set[str]:
    """Every module imported anywhere in the source, function bodies included."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _covers(module: str, excluded: str) -> bool:
    return module == excluded or module.startswith(excluded + ".")


def test_the_release_spec_leaves_out_numpy_and_keeps_pil():
    excludes = _spec_excludes()
    assert "numpy" in excludes, "the onefile grows back by ~11 MB of numpy nothing imports"
    assert not any(_covers("PIL", e) or _covers(e, "PIL") for e in excludes), (
        "PIL is excluded, and the tray icon is drawn and loaded with it"
    )


def test_nothing_the_app_ships_imports_an_excluded_module():
    """Derived from the spec, so a new exclude is checked without editing this.

    Function-level imports count: the frozen build has no copy to import
    lazily either, and tray()'s own import is exactly that kind.
    """
    excludes = _spec_excludes()
    shipped = sorted((_repo() / "fluid_motion").rglob("*.py")) + [_repo() / "packaging" / "launch.py"]
    assert len(shipped) > 10, f"found almost nothing to check under {_repo()}"

    offenders = [
        f"{path.relative_to(_repo())}: imports {module} (excluded: {excluded})"
        for path in shipped
        for module in _all_imports(path.read_text(encoding="utf-8"))
        for excluded in excludes
        if _covers(module, excluded)
    ]
    assert not offenders, "\n".join(["the source runs, the release would not:", *offenders])


# Runs in a child interpreter: blocking a module by setting sys.modules[name] to
# None makes every later `import name` raise ImportError, which is what the
# frozen build does -- and doing that here would poison every other test.
_TRAY_WITHOUT_NUMPY = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    sys.modules["numpy"] = None

    import fluid_motion.icon as icon

    out = Path(sys.argv[1])
    icon.resources_dir = lambda: out          # draw a fresh icon, not the shipped one
    png = icon.ensure_icon()                  # Image.new + ImageDraw, PNG and ICO save

    from PIL import Image

    image = Image.open(png)                   # what app.tray() does
    image.load()
    image.save(out / "tray.ico", format="ICO")  # what pystray does before LoadImage

    leaked = sorted(k for k, v in sys.modules.items() if v is not None and k.split(".")[0] == "numpy")
    print("OK" if not leaked else "LEAKED " + ",".join(leaked))
    """
)


def test_the_tray_icon_is_drawn_and_loaded_without_numpy(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _TRAY_WITHOUT_NUMPY, str(tmp_path)],
        cwd=_repo(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0 and result.stdout.strip() == "OK", (
        "with numpy unavailable -- as it is in the release -- the tray icon path "
        "failed; tray() would swallow this ImportError and the icon would just "
        f"not appear.\nstdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert (tmp_path / "icon.png").stat().st_size > 200
    assert (tmp_path / "icon.ico").stat().st_size > 200
    assert (tmp_path / "tray.ico").stat().st_size > 200
