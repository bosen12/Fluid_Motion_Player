"""Dimmed text has to stay readable.

AX Player carries the same check and writing it found a real failure there.
This one found one here: --color-faint was oklch(52% ...), which measures
3.18:1 on --color-paper-3 -- under the 4.5:1 WCAG AA wants for text below
18.7px -- and it paints .fps-block dt and .player-dir at --text-xs, about 10px.

Computed from tokens.css rather than pinned to a value, so a future palette
change is judged on the thing that matters instead of on whether it matched a
literal.
"""
import math
import re

import pytest

from fluid_motion.paths import ui_dir

AA_SMALL_TEXT = 4.5

GROUNDS = ("color-paper", "color-paper-2", "color-paper-3")
# Every token this UI sets as a `color:` in styles.css.
FOREGROUNDS = ("color-ink", "color-muted", "color-faint", "color-accent",
               "color-live", "color-danger")


def _oklch_to_srgb(L: float, C: float, h_deg: float) -> tuple[float, float, float]:
    h = math.radians(h_deg)
    a, b = C * math.cos(h), C * math.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def gamma(u: float) -> float:
        u = max(0.0, min(1.0, u))
        return 12.92 * u if u <= 0.0031308 else 1.055 * (u ** (1 / 2.4)) - 0.055

    return gamma(r), gamma(g), gamma(bl)


def _luminance(rgb: tuple[float, float, float]) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg, bg) -> float:
    high, low = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _tokens() -> dict[str, tuple[float, float, float]]:
    text = (ui_dir() / "tokens.css").read_text(encoding="utf-8")
    found = re.findall(
        r"--(color-[\w-]+):\s*oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)", text
    )
    return {name: _oklch_to_srgb(float(L) / 100, float(C), float(h))
            for name, L, C, h in found}


def test_the_colour_conversion_is_right_before_anything_leans_on_it():
    """White on black is 21:1 by definition. Without this the whole file could
    be measuring a wrong colour space and reporting comfortable numbers."""
    assert _contrast(_oklch_to_srgb(1.0, 0, 0), _oklch_to_srgb(0.0, 0, 0)) == pytest.approx(21.0, abs=0.05)


def test_every_token_the_ui_paints_text_in_is_parsed():
    """A token renamed or moved out of oklch() would silently drop out of the
    check below, which would then pass by measuring nothing."""
    tokens = _tokens()
    for name in GROUNDS + FOREGROUNDS:
        assert name in tokens, f"{name} is no longer an oklch() token in tokens.css"


@pytest.mark.parametrize("foreground", FOREGROUNDS)
def test_dimmed_text_stays_readable(foreground):
    """4.5:1 on every ground it can appear over. --text-xs is 0.64rem, so the
    3:1 large-text allowance never applies to this UI's smallest labels."""
    tokens = _tokens()
    for ground in GROUNDS:
        ratio = _contrast(tokens[foreground], tokens[ground])
        assert ratio >= AA_SMALL_TEXT, (
            f"{foreground} on {ground} is {ratio:.2f}:1, under {AA_SMALL_TEXT}:1"
        )
