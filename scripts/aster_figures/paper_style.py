"""Shared geometry and typography for the ASTER paper figures."""
from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MplPath

INK = '#24282D'
MUTED = '#666D75'
LINE = '#AEB6BF'
GRID = '#E5E9ED'
BLUE, BLUE_L = '#396E9D', '#EDF3F8'
TEAL, TEAL_L = '#218473', '#EAF4F0'
ORANGE, ORANGE_L = '#C17A36', '#FAF0E5'
PURPLE, PURPLE_L = '#7964A1', '#F1EDF6'
RED, RED_L = '#B95452', '#FAEEEE'
MASK, MASK_L = '#ABB2BB', '#F0F1F3'


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'mathtext.fontset': 'dejavusans',
        'axes.labelsize': 8.0, 'axes.titlesize': 9.0,
        'xtick.labelsize': 7.3, 'ytick.labelsize': 7.3,
        'legend.fontsize': 7.4, 'axes.linewidth': .65,
        'lines.linewidth': 1.4, 'patch.linewidth': .65,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'aster-paper-20261026-v2',
        'axes.unicode_minus': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    })


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, label, size=8.0, color=INK, ha='left', va='center', weight='normal', **kw):
    return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=1.15, **kw)


def panel(ax, x, y, letter, title):
    text(ax, x, y, f'({letter})', 9.1, weight='bold')
    text(ax, x+.27, y, title, 9.1, weight='bold')


def rect(ax, x, y, w, h, fill='white', edge=LINE, lw=.7, hatch=None, zorder=1, linestyle='-'):
    p = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                  linewidth=lw, hatch=hatch, zorder=zorder, linestyle=linestyle)
    ax.add_patch(p)
    return p


def node(ax, x, y, w, h, label, fill='white', edge=LINE, color=INK,
         size=8.0, weight='normal', lw=.75, hatch=None):
    p = rect(ax, x, y, w, h, fill, edge, lw, hatch)
    text(ax, x+w/2, y+h/2, label, size, color, ha='center', weight=weight)
    return p


def line(ax, x1, y1, x2, y2, color=LINE, lw=.7, ls='-', zorder=2):
    return ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                   linestyle=ls, solid_capstyle='round', zorder=zorder)[0]


def arrow(ax, x1, y1, x2, y2, color=INK, lw=.85, style='-|>', rad=0, ls='-', zorder=2):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=7.5, linewidth=lw, color=color,
        connectionstyle=f'arc3,rad={rad}', linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(p)
    return p


def bracket(ax, x1, x2, y, label, color=MUTED, up=True, size=7.2):
    d=.045 if up else -.045
    line(ax, x1, y, x1, y+d, color, .65)
    line(ax, x1, y+d, x2, y+d, color, .65)
    line(ax, x2, y+d, x2, y, color, .65)
    text(ax, (x1+x2)/2, y+(0.13 if up else -.13), label, size, color, ha='center')


def dot(ax, x, y, color=INK, r=.032):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color, linewidth=.5, zorder=3))


def cross(ax, x, y, color=MASK, size=.07, lw=.75):
    line(ax, x-size, y-size, x+size, y+size, color, lw, zorder=4)
    line(ax, x-size, y+size, x+size, y-size, color, lw, zorder=4)


def save(fig, outdir, stem, dpi=450):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        metadata = ({'Creator': 'ASTER paper figures', 'CreationDate': None, 'ModDate': None}
                    if ext == 'pdf' else
                    {'Creator': 'ASTER paper figures', 'Date': None}
                    if ext == 'svg' else None)
        payload = BytesIO()
        fig.savefig(payload, format=ext, dpi=dpi, metadata=metadata)
        # Readers never see an incomplete file while the figure set is rebuilt.
        with NamedTemporaryFile(dir=outdir, prefix=f'.{stem}-', suffix=f'.{ext}', delete=False) as f:
            f.write(payload.getvalue())
            temp_path = Path(f.name)
        temp_path.replace(outdir/f'{stem}.{ext}')
    plt.close(fig)
