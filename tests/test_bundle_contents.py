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


# What the frozen app was measured to load from PIL, plus the two format plugins
# its icon path goes through. None of these may be excluded, by name or by a
# parent package.
PIL_IN_USE = (
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL._imaging",
    "PIL._imagingmath",
    "PIL.PngImagePlugin",
    "PIL.IcoImagePlugin",
)


def test_the_release_spec_leaves_out_the_dead_weight():
    excludes = _spec_excludes()
    for name in ("numpy", "cryptography", "PIL._avif", "PIL._imagingft"):
        assert name in excludes, f"{name} is back in the onefile; nothing loads it"


def test_the_spec_keeps_the_pil_the_tray_icon_needs():
    excludes = _spec_excludes()
    hit = [(used, e) for used in PIL_IN_USE for e in excludes if _covers(used, e)]
    assert not hit, f"excluding {hit} takes out what the tray icon is drawn and loaded with"


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


def _import_time_imports(source: str) -> set[str]:
    """Imports that run when the module is imported: anything outside a
    function body. Class bodies and module-level if/try count."""
    found = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                found.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                found.add(child.module)
            visit(child)

    visit(ast.parse(source))
    return found


def test_pywebview_still_imports_cryptography_only_lazily():
    """cryptography is excluded on the strength of pywebview importing it only
    inside __generate_ssl_cert(). A release of pywebview that imported it at
    module level would make the frozen app die at `import webview` -- on
    start-up, for everyone -- while every test here still passed. Located with
    find_spec, which does not execute anything."""
    import importlib.util

    spec = importlib.util.find_spec("webview")
    assert spec is not None and spec.submodule_search_locations, "pywebview is not installed"
    package = Path(next(iter(spec.submodule_search_locations)))
    sources = sorted(package.rglob("*.py"))
    assert any("cryptography" in m for f in sources for m in _all_imports(f.read_text(encoding="utf-8"))), (
        "pywebview no longer imports cryptography at all; re-derive the exclude"
    )
    eager = [
        f"{f.relative_to(package)}: {m}"
        for f in sources
        for m in _import_time_imports(f.read_text(encoding="utf-8"))
        if _covers(m, "cryptography")
    ]
    assert not eager, "pywebview now imports cryptography at import time:\n" + "\n".join(eager)


# Runs in a child interpreter: blocking a module by setting sys.modules[name] to
# None makes every later `import name` raise ImportError, which is what the
# frozen build does -- and doing that here would poison every other test. The
# list to block is passed in from the spec, so a new exclude is exercised too.
_TRAY_WITH_EXCLUDES_MISSING = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    blocked = json.loads(sys.argv[2])
    for name in blocked:
        sys.modules[name] = None

    import fluid_motion.icon as icon

    out = Path(sys.argv[1])
    icon.resources_dir = lambda: out          # draw a fresh icon, not the shipped one
    png = icon.ensure_icon()                  # Image.new + ImageDraw, PNG and ICO save

    from PIL import Image

    Image.init()                              # every format plugin, as an unknown file would
    image = Image.open(png)                   # what app.tray() does
    image.load()
    image.save(out / "tray.ico", format="ICO")  # what pystray does before LoadImage
    assert "PNG" in Image.OPEN and "ICO" in Image.SAVE

    import webview                            # import-time safe without cryptography

    leaked = sorted(
        k for k, v in sys.modules.items()
        if v is not None and any(k == b or k.startswith(b + ".") for b in blocked)
    )
    print("OK" if not leaked else "LEAKED " + ",".join(leaked))
    """
)


def test_the_tray_icon_works_with_every_excluded_module_missing(tmp_path):
    import json

    blocked = [e for e in _spec_excludes() if e not in ("pytest", "unittest")]
    result = subprocess.run(
        [sys.executable, "-c", _TRAY_WITH_EXCLUDES_MISSING, str(tmp_path), json.dumps(blocked)],
        cwd=_repo(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0 and result.stdout.strip() == "OK", (
        f"with {blocked} unavailable -- as they are in the release -- the tray "
        "icon path failed; tray() would swallow this ImportError and the icon "
        f"would just not appear.\nstdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert (tmp_path / "icon.png").stat().st_size > 200
    assert (tmp_path / "icon.ico").stat().st_size > 200
    assert (tmp_path / "tray.ico").stat().st_size > 200
