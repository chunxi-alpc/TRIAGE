"""Task and operator schematics; no simulated model observations."""
from pathlib import Path
from .paper_style import (
    configure, canvas, text, panel, rect, node, line, arrow, bracket, dot,
    cross, save, INK, MUTED, LINE, GRID, BLUE, BLUE_L, TEAL, TEAL_L,
    ORANGE, ORANGE_L, PURPLE, PURPLE_L, RED, RED_L, MASK, MASK_L,
)


def fig01(outdir):
    configure()
    fig, ax = canvas(7.0, 2.55)
    panel(ax, .12, 2.40, 'a', 'Fixed patient and rules')
    rect(ax, .13, 1.83, 2.18, .40, BLUE_L, BLUE, .7)
    text(ax, 1.22, 2.03, 'Relapse at 4 months; no prior H', 7.8, ha='center')

    # The counterfactual SW instance, also used in Fig. 3(d).
    line(ax, .32, 1.83, .32, .78, LINE, .8)
    line(ax, .32, 1.39, .58, 1.39, TEAL, 1.35)
    line(ax, .32, .78, .58, .78, LINE, .8)
    dot(ax, .32, 1.39, TEAL, .028)
    text(ax, .62, 1.39, r'$r_1$', 8.5, weight='bold')
    text(ax, .92, 1.39, r'$t<6\;\wedge\;\neg H$', 8.1)
    arrow(ax, 1.72, 1.39, 1.93, 1.39, TEAL, 1.0)
    node(ax, 1.97, 1.245, .30, .29, 'X', TEAL_L, TEAL, size=8.6, weight='bold', lw=1.0)
    text(ax, .62, .78, r'$r_2$', 8.5)
    text(ax, .92, .78, r'$t\geq 6$', 8.1)
    arrow(ax, 1.72, .78, 1.93, .78, LINE, .8)
    node(ax, 1.97, .635, .30, .29, 'Y', ORANGE_L, ORANGE, size=8.6)
    text(ax, 2.12, .47, 'familiar', 7.3, color=ORANGE, ha='center')
    line(ax, .64, .12, .88, .12, TEAL, 1.35)
    dot(ax, .76, .12, TEAL, .025)
    text(ax, .97, .12, 'Gold branch', 7.4, color=MUTED)

    line(ax, 2.48, .15, 2.48, 2.42, GRID, .7)
    # Matched read -> branch -> action lanes; each element has a task meaning.
    panel(ax, 2.69, 2.40, 'b', 'Action-visible')
    for x, label in ((3.10, 'Read'), (4.69, 'Select branch'), (6.34, 'Execute')):
        text(ax, x, 2.16, label, 7.4, color=MUTED, ha='center')
    node(ax, 2.77, 1.73, .66, .34, r'$C+P$', BLUE_L, BLUE, size=8.7)
    arrow(ax, 3.53, 1.90, 4.32, 1.90, BLUE, .9)
    node(ax, 2.84, 1.39, .22, .23, 'X', 'white', LINE, color=MUTED, size=7.8)
    node(ax, 3.14, 1.39, .22, .23, 'Y', ORANGE_L, ORANGE, size=7.8, weight='bold')
    arrow(ax, 3.43, 1.505, 4.37, 1.74, ORANGE, 1.0, rad=.15, ls=(0, (3, 2)))
    text(ax, 3.95, 1.30, 'Prior preference', 7.3, color=MUTED, ha='center')
    node(ax, 4.39, 1.73, .60, .34, r'$r_2$', ORANGE_L, ORANGE, size=9)
    arrow(ax, 5.08, 1.90, 6.05, 1.90, ORANGE, 1.0)
    node(ax, 6.15, 1.73, .38, .34, 'Y', ORANGE_L, ORANGE, size=9)
    text(ax, 4.69, 1.55, 'Inapplicable', 7.4, color=ORANGE, ha='center')
    cross(ax, 6.72, 1.90, ORANGE, .045, 1.0)

    panel(ax, 2.69, 1.03, 'c', 'ASTER')
    node(ax, 2.77, .44, .66, .34, r'$C+P$', BLUE_L, BLUE, size=8.7)
    arrow(ax, 3.53, .61, 4.32, .61, TEAL, 1.0)
    node(ax, 4.39, .44, .60, .34, r'$r_1$', TEAL_L, TEAL, size=9, lw=1.0)
    arrow(ax, 5.08, .61, 6.05, .61, TEAL, 1.0)
    node(ax, 6.15, .44, .38, .34, 'X', TEAL_L, TEAL, size=9, weight='bold', lw=1.0)
    # Action tokens remain in the context; their read edges are masked.
    node(ax, 2.84, .10, .22, .23, 'X', MASK_L, MASK, color=MUTED, size=7.8, hatch='///')
    node(ax, 3.14, .10, .22, .23, 'Y', MASK_L, MASK, color=MUTED, size=7.8, hatch='///')
    text(ax, 4.69, .23, 'BLIND', 7.4, color=TEAL, ha='center')
    text(ax, 5.66, .23, 'REVEAL', 7.4, color=TEAL, ha='center')
    dot(ax, 6.72, .61, TEAL, .035)
    save(fig, outdir, 'fig01_tlbs')


def _blind_mask(ax):
    labels = [r'$C_1$', r'$A_1$', r'$C_2$', r'$A_2$', r'$P$', r'$B$']
    x0, y0, cell = .59, 1.10, .238
    for j, label in enumerate(labels):
        text(ax, x0+(j+.5)*cell, y0+6*cell+.13, label, 7.8, ha='center')
    for i, label in enumerate(labels):
        y = y0+(5-i)*cell
        text(ax, x0-.12, y+cell/2, label, 7.8, ha='right')
        for j in range(6):
            x = x0+j*cell
            causal = j <= i
            action_query, action_key = i in (1, 3), j in (1, 3)
            blocked = causal and action_key and (not action_query or i != j)
            rect(ax, x, y, cell, cell,
                 MASK_L if blocked else BLUE_L if causal else 'white',
                 edge=GRID, lw=.55)
            if blocked:
                cross(ax, x+cell/2, y+cell/2, MASK, .042, .75)
            elif causal:
                dot(ax, x+cell/2, y+cell/2, BLUE, .013)
    # A2 -> A1 is blocked even after the phase switch (not clerk-dependent).
    rect(ax, x0+cell+.018, y0+2*cell+.018, cell-.036, cell-.036,
         fill='none', edge=INK, lw=.85)
    text(ax, .14, y0+3*cell, 'Queries', 7.4, color=MUTED, rotation=90, ha='center')
    text(ax, x0+3*cell, 2.83, 'Keys', 7.4, color=MUTED, ha='center')
    # Compact categorical legend, not an empirical attention heatmap.
    for x, label, mode in ((.18, 'Allowed', 'read'), (1.08, 'Blocked', 'block'), (1.99, 'Future', 'future')):
        rect(ax, x, -.18, .12, .12,
             BLUE_L if mode == 'read' else MASK_L if mode == 'block' else 'white', GRID, .65)
        if mode == 'read':
            dot(ax, x+.06, -.12, BLUE, .012)
        elif mode == 'block':
            cross(ax, x+.06, -.12, MASK, .027, .65)
        text(ax, x+.17, -.12, label, 7.2, color=MUTED)
    return x0+6*cell


def _token_tape(ax):
    x, y, w, gap = .52, 3.43, .46, .12
    tokens = [(r'$C_1$', BLUE_L, BLUE), (r'$A_1$', ORANGE_L, ORANGE),
              (r'$C_2$', BLUE_L, BLUE), (r'$A_2$', ORANGE_L, ORANGE),
              (r'$N$', 'white', LINE)]
    for label, fill, edge in tokens:
        node(ax, x, y, w, .29, label, fill, edge, size=8.2)
        x += w+gap
    bracket(ax, .52, x-gap, 3.79, 'Reusable guideline K/V prefix', size=7.6)
    arrow(ax, x+.015, y+.145, 3.74, y+.145, LINE, .8)
    for x, label, fill, edge in ((3.83, r'$P$', BLUE_L, BLUE),
                                (4.41, r'$Q$', 'white', LINE),
                                (5.15, r'$B$', BLUE_L, BLUE),
                                (6.26, r'$y_t$', TEAL_L, TEAL)):
        node(ax, x, y, .46, .29, label, fill, edge, size=8.2)
    arrow(ax, 4.96, y+.145, 5.07, y+.145, LINE, .8)
    arrow(ax, 5.71, y+.145, 6.17, y+.145, LINE, .8)
    text(ax, 4.35, 3.27, 'Patient + question', 7.2, color=MUTED, ha='center')
    text(ax, 5.38, 3.27, 'BLIND block', 7.2, color=MUTED, ha='center')
    text(ax, 6.49, 3.27, 'REVEAL', 7.2, color=MUTED, ha='center')


def fig02(outdir):
    configure()
    fig, ax = canvas(7.0, 4.30)
    ax.set_ylim(-.25, 4.05)
    _token_tape(ax)
    line(ax, .12, 3.08, 6.88, 3.08, GRID, .7)
    panel(ax, .12, 2.98, 'a', 'BLIND mask')
    panel(ax, 2.59, 2.98, 'b', 'Applicability')
    panel(ax, 4.72, 2.98, 'c', 'REVEAL bias')
    right = _blind_mask(ax)

    # The model reads at a fixed non-action B position; veto precedes writing ID.
    text(ax, 3.49, 2.67, r'Read at $B$', 7.5, color=MUTED, ha='center')
    node(ax, 2.94, 2.18, 1.10, .34, r'$p_{\mathrm{blind}}$', BLUE_L, BLUE, size=9)
    # Route the output from B, never from an action or guide position.
    rect(ax, .59, 1.10, 6*.238, .238, fill='none', edge=BLUE, lw=.85)
    line(ax, right+.035, 1.219, 2.20, 1.219, BLUE, .9)
    line(ax, 2.20, 1.219, 2.20, 2.35, BLUE, .9)
    arrow(ax, 2.20, 2.35, 2.83, 2.35, BLUE, .9)
    arrow(ax, 3.49, 2.10, 3.49, 1.97, INK, .85)
    node(ax, 2.94, 1.55, 1.10, .34, 'Soft veto*', PURPLE_L, PURPLE, size=8.0)
    arrow(ax, 3.49, 1.47, 3.49, 1.34, INK, .85)
    node(ax, 2.94, .94, 1.10, .32, r'$p(r\mid g)$', TEAL_L, TEAL, size=9)
    arrow(ax, 3.49, .86, 3.49, .73, INK, .85)
    node(ax, 2.84, .34, 1.30, .31, 'Write branch ID', 'white', TEAL, size=7.5)

    # Optional evidence only sees conditions and the patient, never actions.
    text(ax, .16, .495, r'$C,P$', 8.0, color=BLUE)
    arrow(ax, .58, .495, .79, .495, BLUE, .8)
    node(ax, .87, .34, 1.26, .31, 'Evidence clerk*', PURPLE_L, PURPLE, size=7.5)
    arrow(ax, .94, .69, .71, 1.03, PURPLE, .8, ls=(0, (3, 2)))
    text(ax, 1.20, .92, r'Joint $C,E$*', 7.5, color=PURPLE)
    text(ax, 1.20, .75, 'Post-prefix only', 7.2, color=MUTED)
    line(ax, 2.13, .495, 2.39, .495, PURPLE, .8, ls=(0, (3, 2)))
    line(ax, 2.39, .495, 2.39, 1.72, PURPLE, .8, ls=(0, (3, 2)))
    arrow(ax, 2.39, 1.72, 2.85, 1.72, PURPLE, .8, ls=(0, (3, 2)))
    text(ax, 1.50, .18, '* ASTER-full only', 7.2, color=MUTED, ha='center')

    # Gate entries are exact symbolic logit biases, not sampled attention values.
    text(ax, 5.85, 2.82, r'Example: $p_1>\max(p_2,p_{\mathrm{NONE}})$', 7.3, color=MUTED, ha='center')
    labels = [r'$C_1$', r'$A_1$', r'$C_2$', r'$A_2$', r'$P$', r'$B$']
    x0, cell = 5.20, .272
    for j, label in enumerate(labels):
        text(ax, x0+(j+.5)*cell, 2.67, label, 7.8, ha='center')
        node(ax, x0+j*cell, 2.18, cell, .28,
             r'$b_2$' if j==3 else '0',
             RED_L if j==3 else 'white', RED if j==3 else GRID,
             color=RED if j==3 else MUTED, size=7.5, lw=.6)
    text(ax, 4.72, 2.32, 'Gate', 7.5, weight='bold')
    text(ax, 6.83, 2.00, 'All layers / heads', 7.2, color=MUTED, ha='right')
    line(ax, 4.04, 1.10, 4.39, 1.10, TEAL, .9)
    line(ax, 4.39, 1.10, 4.39, 2.32, TEAL, .9)
    arrow(ax, 4.39, 2.32, 4.63, 2.32, TEAL, .9)
    arrow(ax, 4.39, 1.10, 4.63, 1.10, TEAL, .9)
    text(ax, 5.85, 1.74, r'$b_r=\lambda_g\log(p_r/p_{\max})$', 8.9, ha='center')
    text(ax, 5.85, 1.48, r'$\lambda_g=\lambda_0\,(p_{(1)}-p_{(2)})$', 8.2, ha='center')

    text(ax, 4.72, 1.10, 'Emph.', 7.5, weight='bold')
    for j, label in enumerate([r'$I_1$', r'$C_1$', r'$A_1$', r'$E_1$']):
        node(ax, 5.24+j*.40, .96, .31, .28, label, TEAL_L, TEAL, size=8.0, lw=.65)
    text(ax, 6.83, .79, r'Bounded $+\beta$ in selected layers', 7.2, color=MUTED, ha='right')
    node(ax, 5.15, .34, 1.67, .31, 'Generate decision path', 'white', TEAL, size=7.5)
    arrow(ax, 4.22, .495, 5.06, .495, TEAL, .9)
    line(ax, 6.88, 2.32, 6.94, 2.32, INK, .7)
    line(ax, 6.88, 1.10, 6.94, 1.10, INK, .7)
    line(ax, 6.94, 2.32, 6.94, .495, INK, .7)
    arrow(ax, 6.94, .495, 6.88, .495, INK, .7)
    text(ax, 4.84, .14, 'Causal mask + cross-action isolation persist', 7.2, color=MUTED, ha='center')
    save(fig, outdir, 'fig02_aster')


if __name__ == '__main__':
    out = Path(__file__).resolve().parent/'rendered'
    fig01(out)
    fig02(out)
