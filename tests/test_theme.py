"""The shared terminal visual language keeps text exact and widths stable."""
import pytest

from so3lr.cli import theme
from so3lr.cli.formatting import terminal_text


# Empty, a half cell, and clamping at both ends.
@pytest.mark.parametrize('ratio', [0, 0.26, 1.7, -3])
@pytest.mark.parametrize('ascii_only', [False, True])
def test_bar_has_exact_width_and_monotone_fill(ratio, ascii_only):
    bar = theme.bar(ratio, 20, ascii_only)
    assert len(bar.plain) == 20
    track = '-' if ascii_only else '─'
    full, half = divmod(round(max(0, min(1, ratio)) * 40), 2)
    assert len(bar.plain.rstrip(track)) == full + half
    if half and not ascii_only:
        assert bar.plain[full] == '╸'
    if ascii_only:
        assert bar.plain.isascii()


def test_controls_keep_visible_text_and_emphasize_only_keys():
    hint = '↑/↓: plots · e: energy · Esc/q: hide plots · Ctrl+C: interrupt MD'
    text = theme.controls(hint)
    assert text.plain == hint
    bold = [text.plain[span.start:span.end] for span in text.spans if span.style == theme.STRONG]
    assert bold == ['↑/↓', 'e', 'Esc/q', 'Ctrl+C']
    assert theme.controls(hint, ascii_only=True).plain == terminal_text(hint, True)
    assert theme.controls('Line one\nq: quit').plain == 'Line one\nq: quit'
