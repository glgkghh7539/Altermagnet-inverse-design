"""Shared figure writer: one render per file, and a crop box measured cheaply.

`bbox_inches='tight'` makes matplotlib render the whole canvas an EXTRA time at the save
dpi purely to find where the ink ends, once per output file. On a large canvas that extra
pass is most of the memory cost, and when it does not fit the OS kills the process with
SIGKILL before Python can raise anything - the symptom is a script that prints nothing and
exits 137. Measuring the crop box at 100 dpi gives the same answer in inches for 1/100 of
the pixels, and the resulting Bbox can be handed to savefig, which then skips its own
tight pass.

The figure is closed afterwards, so a script that writes several figures does not hold
them all.
"""
import gc
import os

import matplotlib.pyplot as plt

LAYOUT_DPI = 100      # dpi used only to measure the crop box
PAD_INCHES = 0.1      # matplotlib's own default padding for bbox_inches='tight'

# Output resolution, the same two-tier policy the figure notebook uses.
#   PNG_DPI         resolution of the png
#   PDF_RASTER_DPI  resolution of any `rasterized=True` layer embedded in the pdf.
#                   Text, axes and lines stay vector whatever this is, and a figure
#                   with no rasterized artist is unaffected by it entirely. 400 is
#                   above the 300 dpi journals ask for raster content, and costs a
#                   third of the memory of rendering the pdf at 800.
PNG_DPI = 800
PDF_RASTER_DPI = 400


def tight_bbox(fig):
    dpi0 = fig.dpi
    try:
        fig.set_dpi(LAYOUT_DPI)
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
    finally:
        fig.set_dpi(dpi0)
    return bb.padded(PAD_INCHES)


def save(fig, outdir, stem, dpi=None, pdf_raster_dpi=None, exts=('pdf', 'png'),
         close=True):
    """Write `stem.pdf` and `stem.png` into outdir and report what was written."""
    os.makedirs(outdir, exist_ok=True)
    dpi = PNG_DPI if dpi is None else dpi
    pdf_dpi = PDF_RASTER_DPI if pdf_raster_dpi is None else pdf_raster_dpi
    w, h = fig.get_size_inches()
    bbox = tight_bbox(fig)
    for ext in exts:
        d = pdf_dpi if ext == 'pdf' else dpi
        p = os.path.join(outdir, f'{stem}.{ext}')
        fig.savefig(p, dpi=d, bbox_inches=bbox)
        gc.collect()
        print(f'  written: {p}   ({d} dpi, {int(w * d):,} x {int(h * d):,} px, '
              f'{os.path.getsize(p) / 1e6:.1f} MB)')
    if close:
        plt.close(fig)
        gc.collect()
