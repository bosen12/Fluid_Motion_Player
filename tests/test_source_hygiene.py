"""Compile-time warnings, which are invisible on every run but the first.

CPython emits SyntaxWarning while *compiling* a module and then caches the
result in __pycache__, so a bad escape sequence announces itself once -- on
the run right after the file is edited -- and is silent from then on.

This repo is clean today; the net is here because the failure mode is that
nobody sees it. AX Player had two sitting in its tests, found only because a
suite run happened to follow a source change. Both were docstrings mentioning
a Windows path, and this codebase writes %APPDATA%\\FluidMotion paths into
prose constantly, so it is the same exposure.

"\\m" and "\\d" are invalid escape sequences that Python still resolves to
themselves today, with a warning saying "such sequences will not work in the
future"; the plan of record is for them to become SyntaxError.
"""
import pathlib
import warnings

SKIP_DIRS = {"__pycache__", "build", "dist", ".venv", "venv", ".git", "mpv"}


def _sources(root: pathlib.Path):
    for path in sorted(root.rglob("*.py")):
        if SKIP_DIRS.isdisjoint(path.parts):
            yield path


def test_no_module_compiles_with_a_warning():
    """Every .py in the repo, compiled fresh with warnings turned up.

    Anchored through this file rather than the cwd. A cwd-relative read in a
    test once opened a *different repo's* file and passed on it, and with two
    sibling projects on this machine that is not hypothetical.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "fluid_motion").is_dir(), f"anchored at the wrong tree: {root}"

    found = []
    for path in _sources(root):
        source = path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, str(path), "exec")
        for warning in caught:
            found.append(
                f"{path.relative_to(root)}:{warning.lineno} "
                f"{warning.category.__name__}: {warning.message}"
            )

    assert not found, "\n".join(["compile-time warnings:", *found])
