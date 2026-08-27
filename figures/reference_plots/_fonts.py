"""Shared font setup: Times New Roman, with a fallback that is actually a serif.

Every script here used to set `rcParams['font.family'] = 'Times New Roman'` on its own.
That is a trap: when the font is missing matplotlib does not fall back to another serif,
it falls back to **DejaVu Sans**, and the only sign is a `findfont` warning on stderr. A
figure that looks fine to the person who has the font ships in a sans-serif to everyone
else. This module removes that failure mode.

Times New Roman is not redistributable, so it is not in this repository. It is looked for
in the usual places and, if found, registered straight into matplotlib's font manager - no
system-wide install and no cache clearing needed. Drop `times.ttf`, `timesbd.ttf`,
`timesi.ttf` and `timesbi.ttf` into any of these and it will be picked up:

    $TIMES_FONT_DIR                       explicit override
    figures/fonts/                        beside the figure code
    ~/.fonts, ~/.local/share/fonts        the usual user font directories
    /usr/share/fonts/truetype/msttcorefonts
    /Library/Fonts, /System/Library/Fonts/Supplemental      (macOS)
    /mnt/c/Windows/Fonts, /media/sf_Windows/Fonts, C:\\Windows\\Fonts

Failing that, the order is Liberation Serif, then Nimbus Roman, then DejaVu Serif. The
first two are metrically compatible with Times New Roman, so line breaks and figure sizes
do not move; Liberation Serif is the closer match in glyph shape.
"""
import glob
import os

import matplotlib
from matplotlib import font_manager

FONT_DIRS = [
    os.environ.get('TIMES_FONT_DIR', ''),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts'),
    os.path.expanduser('~/.fonts'),
    os.path.expanduser('~/.local/share/fonts'),
    '/usr/share/fonts/truetype/msttcorefonts',
    '/Library/Fonts', '/System/Library/Fonts/Supplemental',
    '/mnt/c/Windows/Fonts', '/media/sf_Windows/Fonts', 'C:\\Windows\\Fonts',
]

PREFERENCE = ('Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif')

_resolved = None


def register():
    """Add any times*.ttf reachable from this machine to matplotlib's font manager."""
    for d in FONT_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, '[Tt]imes*.tt[fc]'))):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass        # .ttc collections are not readable by every build


def use(mathtext='cm', quiet=False):
    """Set rcParams to the best available serif and return its name."""
    global _resolved
    if _resolved is None:
        register()
        available = {f.name for f in font_manager.fontManager.ttflist}
        _resolved = next((f for f in PREFERENCE if f in available), 'serif')
        if _resolved != 'Times New Roman' and not quiet:
            print(f'[note] "Times New Roman" not found; using the metric-compatible '
                  f'"{_resolved}". Put times.ttf, timesbd.ttf, timesi.ttf and '
                  f'timesbi.ttf in figures/fonts/ (or point $TIMES_FONT_DIR at them) '
                  f'to reproduce the published figures glyph-for-glyph.')
    matplotlib.rcParams['font.family'] = _resolved
    matplotlib.rcParams['mathtext.fontset'] = mathtext
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42        # embed as TrueType, not Type 3
    return _resolved
