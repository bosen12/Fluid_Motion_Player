"""The README and the published site are surfaces too, and they drift.

Both projects have now been caught carrying a claim the code had moved past:
AX Player's site still described a thumbnail pool cap that had changed and gave
a reason its own HANDOFF disproved, and this one described a TensorRT-only app
five releases after the ncnn/AMD path shipped.

The backend list is derived from vs_script rather than written out here -- a
hand-kept list is the same drift one level up, which is how the XSS guard's
field list went stale.
"""
from pathlib import Path

import pytest

from fluid_motion.core.vs_script import BACKEND_NCNN, BACKEND_TRT, backend_label
from fluid_motion.paths import project_root

SURFACES = ("README.md", "docs/index.html")


def _read(name: str) -> str:
    return (project_root() / name).read_text(encoding="utf-8", errors="replace")


def test_the_surfaces_under_test_are_this_repos():
    """Anchored through paths.project_root(), which resolves from the package
    file. A cwd-relative read is how a build.bat test in the sibling project
    ended up opening a different repo's file and passing for it."""
    import fluid_motion

    repo = Path(fluid_motion.__file__).resolve().parent.parent
    for name in SURFACES:
        path = (project_root() / name).resolve()
        assert path.is_relative_to(repo), f"{path} is outside {repo}"
        assert path.is_file(), f"{path} is missing"


@pytest.mark.parametrize("surface", SURFACES)
def test_every_shipped_backend_is_named_on_the_public_surfaces(surface):
    """An installable backend nobody is told about may as well not ship.

    The ncnn path exists specifically to collect reports from AMD owners --
    log.py's own docstring says so: "it exists because v1.6.0 shipped an AMD
    path nobody could test and asked users to report back; a report needs
    something to report". The site listed "NVIDIA 顯示卡" as a flat
    requirement, so an AMD owner read it and left.
    """
    text = _read(surface)
    for backend in (BACKEND_TRT, BACKEND_NCNN):
        label = backend_label(backend)
        # "ncnn / Vulkan" is written with and without the spaces depending on
        # the sentence, so match on the distinctive half.
        needle = label.split(" / ")[0]
        assert needle in text, (
            f"{surface} never mentions the {label} backend, which the app ships"
        )


def test_quickstart_and_site_intro_do_not_describe_a_tensorrt_only_product():
    readme = _read("README.md")
    site = _read("docs/index.html")

    assert "Install TensorRT runtime" not in readme
    assert "安裝 TensorRT 執行環境" not in readme
    assert "Windows · NVIDIA · 不是播放器" not in site
    assert "RIFE + TensorRT。即時補幀" not in site
    assert "ncnn" in site and "AMD" in site


@pytest.mark.parametrize("surface", SURFACES)
def test_the_untested_amd_path_is_still_labelled_untested(surface):
    """Naming the backend without the caveat would be worse than omitting it.

    Nobody working on this repo has AMD hardware; the plugin archive, the
    generated script and the pinned vsmlrt API are verified, and that a frame
    actually comes out the other side is not. CLAUDE.md keeps three rules
    holding that risk down -- promising it on the front page is not one of them.
    """
    text = _read(surface)
    assert "未經 AMD 實機驗證" in text or "Untested on AMD" in text, (
        f"{surface} names the AMD backend without saying it is unverified"
    )


def test_the_sites_offline_version_fallback_is_not_stale():
    """docs/app.js fetches the newest tag from the GitHub API and writes it into
    the chip; the number in the HTML is what a visitor sees when that call does
    not come back -- offline, or rate-limited, which the unauthenticated API
    does easily.

    It was three minor releases behind when this test was written (v1.4.7 on a
    v1.6.4 app), and the sibling project's site was two behind, so this is
    drift with a rhythm rather than a one-off. Tied to __version__ because the
    release procedure already bumps that by hand in two files -- this makes the
    site ride along with a step that exists, instead of adding a third one to
    forget.
    """
    from fluid_motion import __version__

    html = _read("docs/index.html")
    assert 'id="ver-chip"' in html, "the chip this guards is gone"
    assert f'id="ver-chip">v{__version__}<' in html, (
        f"the offline fallback does not say v{__version__}; "
        "bump it in docs/index.html when bumping the version"
    )
