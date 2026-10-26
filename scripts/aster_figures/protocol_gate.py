"""Compact counterfactual protocol and analytic gate figures for ASTER.

The protocol is a fictional example. The gate curves are exact formula
evaluations at declared probabilities, not observations of model behavior.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .paper_style import (
    configure, canvas, text, rect, line, dot, save,
    INK, MUTED, LINE, GRID, BLUE, BLUE_L, TEAL, TEAL_L,
    ORANGE, ORANGE_L, PURPLE,
)


def _action(ax, x, y, action, gold=False, edited=False):
    """Gold and intervention status use separate, redundant encodings."""
    width, height = .32, .27
    if edited:
        rect(ax, x-width/2-.055, y-.045, width+.11, height+.09,
             fill='white', edge=PURPLE, lw=.9, linestyle=(0, (2.2, 1.6)))
    rect(ax, x-width/2, y, width, height,
         fill=TEAL_L if gold else ORANGE_L,
         edge=TEAL if gold else ORANGE, lw=1.1 if gold else .65)
    text(ax, x, y+height/2, action, size=8.5, ha='center',
         color=INK, weight='bold' if gold else 'normal')


def _tree(ax, cx, title, subtitle, actions, threshold=6,
          gold=0, edited_actions=(), edited_threshold=False):
    text(ax, cx, 1.86, title, 9.1, ha='center', weight='bold')
    text(ax, cx, 1.66, subtitle, 7.5, ha='center', color=MUTED)

    # A shared binary decision node; only CS changes the threshold.
    x0, y0, width, height = cx-.315, 1.245, .63, .255
    if edited_threshold:
        rect(ax, x0-.045, y0-.037, width+.09, height+.074,
             edge=PURPLE, lw=.9, linestyle=(0, (2.2, 1.6)))
    rect(ax, x0, y0, width, height, BLUE_L, BLUE, .7)
    text(ax, cx, y0+height/2, rf'$t < {threshold}$?', 8.2, ha='center')

    xs = (cx-.345, cx+.345)
    for branch, x in enumerate(xs):
        chosen = branch == gold
        col, lw = (TEAL, 1.45) if chosen else (LINE, .75)
        line(ax, cx, y0, x, .99, col, lw)
        text(ax, cx+(-.25 if branch == 0 else .25), 1.15,
             'yes' if branch == 0 else 'no', 7.2,
             color=MUTED, ha='center')
        if chosen:
            dot(ax, x, .99, TEAL, .025)
        text(ax, x, .84, rf'$r_{branch+1}$', 8.2,
             color=INK, ha='center', weight='bold' if chosen else 'normal')
        line(ax, x, .735, x, .65, col, lw)
        _action(ax, x, .365, actions[branch], chosen,
                branch in edited_actions)


def fig03(outdir):
    configure()
    fig, ax = canvas(7.0, 2.25)
    text(ax, .12, 2.12, r'Fixed patient: $t=4$ months; no prior H', 8.0)
    text(ax, 6.88, 2.12, 'Prior-favored action: Y', 8.0, color=MUTED, ha='right')
    line(ax, .12, 2.005, 6.88, 2.005, GRID, .6)

    _tree(ax, .70, '(a) NC', 'Original', ('Y', 'X'))
    _tree(ax, 2.10, '(b) AC', 'Edit applicable action', ('Z', 'X'), edited_actions=(0,))
    _tree(ax, 3.50, '(c) IC', 'Edit sibling action', ('Y', 'Z'), edited_actions=(1,))
    _tree(ax, 4.90, '(d) SW', 'Swap actions', ('X', 'Y'), edited_actions=(0, 1))
    _tree(ax, 6.30, '(e) CS', 'Shift threshold', ('Y', 'X'), threshold=3,
          gold=1, edited_threshold=True)

    # A minimal shared legend; correctness and edits are not color-only.
    line(ax, 2.00, .115, 2.27, .115, TEAL, 1.45)
    dot(ax, 2.15, .115, TEAL, .025)
    text(ax, 2.37, .115, 'Gold path', 7.4, color=MUTED)
    rect(ax, 3.87, .055, .20, .12, edge=PURPLE, lw=.9,
         linestyle=(0, (2.2, 1.6)))
    text(ax, 4.17, .115, 'Edited component', 7.4, color=MUTED)
    save(fig, outdir, 'fig03_protocol')


def fig04(outdir):
    configure()
    fig = plt.figure(figsize=(7.0, 2.25))
    axes = [fig.add_axes([.075, .23, .395, .66]),
            fig.add_axes([.577, .23, .395, .66])]
    pairs = ((.90, .10), (.55, .45))
    colors = (PURPLE, TEAL)
    styles = ('-', (0, (4.0, 2.4)))

    for i, ax in enumerate(axes):
        x = np.linspace(0, 4 if i == 0 else 8, 301)
        for (ptop, pcomp), col, ls in zip(pairs, colors, styles):
            lam = x if i == 0 else x * (ptop-pcomp)
            multiplier = np.exp(lam * np.log(pcomp/ptop))
            ax.plot(x, multiplier, color=col, linestyle=ls, linewidth=1.5,
                    label=rf'$p=({ptop:.2f},\,{pcomp:.2f})$')
        ax.set_xlim(0, 4 if i == 0 else 8)
        ax.set_ylim(-.015, 1.035)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_xticks([0, 1, 2, 3, 4] if i == 0 else [0, 2, 4, 6, 8])
        ax.grid(axis='y', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color(LINE)
        ax.tick_params(pad=2.0)
        ax.set_xlabel(r'Effective strength $\lambda$' if i == 0
                      else r'Base strength $\lambda_0$', labelpad=3.5)
        ax.legend(loc='lower right', bbox_to_anchor=(.99, .11),
                  frameon=False, handlelength=2.35,
                  handletextpad=.55, labelspacing=.35, borderaxespad=0)
    axes[0].set_ylabel('Relative weight multiplier', labelpad=4.3)
    axes[0].set_title('(a) Fixed strength', loc='left', fontsize=9.1,
                      fontweight='bold', pad=6.0)
    axes[1].set_title('(b) Confidence-adaptive strength', loc='left', fontsize=9.1,
                      fontweight='bold', pad=6.0)
    save(fig, outdir, 'fig04_gate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'rendered')
    args = parser.parse_args()
    fig03(args.output_dir)
    fig04(args.output_dir)
