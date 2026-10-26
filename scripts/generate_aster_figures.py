#!/usr/bin/env python3
"""Generate ASTER manuscript figures without empirical or placeholder results.

Requirements: Python 3.10+, matplotlib, numpy, Pillow.
Set ASTER_CJK_FONT to a Noto Sans CJK SC font if it is not installed.
Outputs are SVG (vector paths), PDF (embedded fonts), and 220 dpi PNG.
The charts in Figure 4 are analytic illustrations, not experiment results.
"""
from pathlib import Path
import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
INK = '#17263B'
MUTED = '#5B6C7A'
LINE = '#D8E1EA'
BLUE, BLUE_L = '#2563A6', '#EDF4FB'
TEAL, TEAL_L = '#168A84', '#ECF8F5'
ORANGE, ORANGE_L = '#CF7523', '#FFF4E8'
PURPLE, PURPLE_L = '#7952B3', '#F3EFFA'
RED, RED_L = '#BC4E50', '#FCF0F0'
GREY_L = '#F4F6F9'


def configure():
    candidates = [os.environ.get('ASTER_CJK_FONT', ''),
                  str(ROOT / 'assets/NotoSansCJKsc-Regular.otf'),
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    font = next((p for p in candidates if p and Path(p).is_file()), None)
    if font:
        fm.fontManager.addfont(font)
        name = fm.FontProperties(fname=font).get_name()
    else:
        names = {f.name for f in fm.fontManager.ttflist}
        name = next((n for n in ['Noto Sans CJK SC', 'Source Han Sans SC',
                                'Microsoft YaHei', 'SimHei'] if n in names), None)
        if not name:
            raise RuntimeError('A Chinese font is required. Set ASTER_CJK_FONT to NotoSansCJKsc-Regular.otf.')
    plt.rcParams.update({'font.family': name, 'font.size': 12,
        'axes.unicode_minus': False, 'pdf.fonttype': 3,
        'ps.fonttype': 3, 'svg.fonttype': 'path',
        'mathtext.fontset': 'dejavusans', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(0, h))
    ax.axis('off')
    return fig, ax


def txt(ax, x, y, text, size=13, color=INK, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.6, **kw)


def box(ax, x, y, w, h, fill='white', edge=LINE, lw=1.25, radius=.12, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                       linewidth=lw, edgecolor=edge, facecolor=fill, **kw)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.6, style='-|>', rad=0, dashed=False):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                       mutation_scale=14, linewidth=lw,
                       color=color, connectionstyle=f'arc3,rad={rad}',
                       linestyle='--' if dashed else '-')
    ax.add_patch(p)
    return p


def heading(ax, w, h, num, title, subtitle):
    txt(ax, .55, h-.53, f'图 {num}  {title}', 23, weight='bold')
    txt(ax, .56, h-1.04, subtitle, 11.5, MUTED)
    ax.plot([.55, w-.55], [h-1.38, h-1.38], color=LINE, lw=1)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=220, bbox_inches=None,
                    metadata={'Creator': 'ASTER reproducible scientific figures'} if ext == 'pdf' else None)
    plt.close(fig)


def fig01(out):
    fig, ax = canvas(15, 9)
    heading(ax, 15, 9, 1, '先判断适用分支，再读取处置',
            '待检验现象：熟悉的处置是否反向影响分支判断？以下仅为虚构规则与假设路径。')
    box(ax, .55, 5.65, 8.9, 1.98, BLUE_L, BLUE)
    txt(ax, .85, 7.30, '同一份条件化指南', 15, BLUE, 'bold')
    txt(ax, .85, 6.72, 'R1   复发距缓解 < 6 个月，且既往未用 H', 13.3)
    txt(ax, .85, 6.12, 'R2   复发距缓解 ≥ 6 个月', 13.3)
    box(ax, 7.30, 6.44, 1.82, .52, TEAL_L, TEAL)
    box(ax, 7.30, 5.87, 1.82, .52, ORANGE_L, ORANGE)
    txt(ax, 8.21, 6.70, '处置 X', 13, TEAL, 'bold', ha='center')
    txt(ax, 8.21, 6.13, '处置 Y', 13, ORANGE, 'bold', ha='center')
    arrow(ax, 6.48, 6.70, 7.25, 6.70, BLUE)
    arrow(ax, 6.48, 6.13, 7.25, 6.13, BLUE)
    box(ax, 9.75, 5.65, 4.70, 1.98, TEAL_L, TEAL)
    txt(ax, 10.05, 7.30, '固定患者事实', 15, TEAL, 'bold')
    txt(ax, 10.05, 6.77, '缓解后 4 个月复发；既往未用 H', 12.3)
    txt(ax, 10.05, 6.18, '按给定规则：R1 → X', 14.3, TEAL, 'bold')

    box(ax, .55, 1.20, 6.75, 4.05, 'white', LINE)
    box(ax, 7.70, 1.20, 6.75, 4.05, 'white', LINE)
    txt(ax, .85, 4.85, 'A  明判：可能出现的诱饵路径', 15, ORANGE, 'bold')
    txt(ax, 8.00, 4.85, 'B  ASTER：拟议的信息流控制', 15, PURPLE, 'bold')

    box(ax, .95, 3.36, 2.25, .93, ORANGE_L, ORANGE)
    box(ax, 4.60, 3.36, 2.25, .93, RED_L, RED)
    txt(ax, 2.075, 3.82, '熟悉处置 Y', 15, ORANGE, 'bold', ha='center')
    txt(ax, 5.725, 3.82, '错误分支 R2', 15, RED, 'bold', ha='center')
    arrow(ax, 3.27, 3.82, 4.52, 3.82, ORANGE, 2, dashed=True)
    txt(ax, 3.89, 4.18, '先验吸引？', 10.8, MUTED, ha='center')
    txt(ax, .95, 2.76, '待检验：处置的合理性是否改变条件判断？', 12.3)
    txt(ax, .95, 2.20, '示意性错误输出：R2 → Y', 13, RED)
    txt(ax, .95, 1.65, '虚线表示研究假设，不表示已证实的因果关系。', 10.7, MUTED)

    coords = [(8.05, 1.72, '读取条件\n与病历', BLUE_L, BLUE),
              (10.30, 1.66, '盲判分支\nR1', PURPLE_L, PURPLE),
              (12.49, 1.60, '揭示处置\nR1 → X', TEAL_L, TEAL)]
    for x, w, text, fill, edge in coords:
        box(ax, x, 3.27, w, 1.10, fill, edge)
        txt(ax, x+w/2, 3.82, text, 12.9, edge, 'bold', ha='center')
    arrow(ax, 9.82, 3.82, 10.23, 3.82, PURPLE)
    arrow(ax, 12.01, 3.82, 12.42, 3.82, PURPLE)
    txt(ax, 8.08, 2.72, '判断阶段：屏蔽全部处置及其间接泄漏', 12.4)
    txt(ax, 8.08, 2.18, '执行阶段：按分支概率压低竞争处置', 12.4)
    txt(ax, 8.08, 1.65, '该路径说明设计目标；能否改善表现须实测。', 10.7, MUTED)
    txt(ax, .57, .64, '概念示意；规则与病例均为虚构，不构成医疗建议。图中未使用原稿的预填结果。', 10.7, MUTED)
    save(fig, out, 'fig01_tlbs')


def fig02(out):
    fig, ax = canvas(16, 10)
    heading(ax, 16, 10, 2, 'ASTER：按阶段控制可见性与处置权重',
            '方法设计图。分支判断与答案生成共用上下文；证据员为可选模块，不能读取处置。')
    # Inputs.
    box(ax, .55, 5.94, 4.12, 2.65, 'white', LINE)
    txt(ax, .85, 8.20, '输入与角色标注', 15, BLUE, 'bold')
    labels = [('编号', BLUE_L, BLUE), ('条件', BLUE_L, BLUE), ('处置', ORANGE_L, ORANGE),
              ('中性文本', GREY_L, MUTED), ('病历', TEAL_L, TEAL), ('问题', TEAL_L, TEAL)]
    for k, (label, fill, edge) in enumerate(labels):
        x = .86 + (k % 3)*1.19
        y = 7.22 - (k//3)*.61
        box(ax, x, y, 1.04, .45, fill, edge, lw=.9, radius=.07)
        txt(ax, x+.52, y+.225, label, 10.8, edge, ha='center')
    txt(ax, .86, 6.21, '处置角色覆盖选项与问题中的处置名称。', 9.9, MUTED)

    box(ax, .55, 4.33, 4.12, 1.30, TEAL_L, TEAL)
    txt(ax, .85, 5.24, '可选证据员（仅 ASTER-full）', 12.8, TEAL, 'bold')
    txt(ax, .85, 4.76, '仅看条件与病历：状态、置信度、证据位置', 10.3)

    # Blind stage.
    box(ax, 5.15, 4.33, 4.94, 4.26, PURPLE_L, PURPLE)
    txt(ax, 5.48, 8.20, '阶段 1  处置盲判', 16, PURPLE, 'bold')
    box(ax, 5.48, 6.98, 4.28, .70, 'white', PURPLE, lw=.9)
    txt(ax, 7.62, 7.33, '非处置查询 → 处置键：遮蔽', 12, PURPLE, ha='center')
    txt(ax, 5.48, 6.52, '条件与病历证据在原位联合高亮', 12.2)
    txt(ax, 5.48, 6.04, '逐组读出盲判分布（先不写编号）', 12.2)
    txt(ax, 7.62, 5.48, r'$p_{\mathrm{blind}}(r\mid g)$', 16, PURPLE, ha='center')
    txt(ax, 5.48, 4.84, '软否决（可选）→ 写入校正后编号', 11.1)
    arrow(ax, 4.72, 7.38, 5.08, 7.38, BLUE)
    arrow(ax, 4.72, 4.98, 5.08, 4.98, TEAL)

    # Reveal stage.
    box(ax, 10.57, 4.33, 4.88, 4.26, 'white', PURPLE)
    txt(ax, 10.89, 8.20, '阶段 2  按适用性揭示', 16, PURPLE, 'bold')
    txt(ax, 10.89, 7.58, '对数门控：压低竞争处置', 12.8, PURPLE, 'bold')
    txt(ax, 13.01, 6.98, r'$B_r=\lambda_g\log(p_r/p_{\max})$', 14.5, PURPLE, ha='center')
    txt(ax, 13.01, 6.39, r'$\lambda_g=\lambda_0\,(p_{(1)}-p_{(2)})$', 14, PURPLE, ha='center')
    txt(ax, 10.89, 5.78, '窗口内有界强调所选规则及其证据', 11.5, TEAL)
    box(ax, 10.89, 4.67, 4.23, .63, TEAL_L, TEAL, lw=1)
    txt(ax, 13.00, 4.985, '生成答案或自由文本决策路径', 12.3, TEAL, 'bold', ha='center')
    arrow(ax, 10.13, 6.32, 10.53, 6.32, PURPLE)
    txt(ax, 10.33, 6.75, r'$p$', 13, PURPLE, ha='center')

    # Access matrix. Avoid claiming bidirectional access: causal restriction noted.
    box(ax, .55, 1.15, 14.90, 2.81, 'white', LINE)
    txt(ax, .85, 3.60, '可见性与隔离约束', 14, weight='bold')
    x0, y0 = .89, 1.51
    widths = [4.22, 3.18, 3.18, 3.18]
    rows = [
        ['查询位置 / 读取对象', '非处置', r'处置 $A_r$', r'处置 $A_s$（$s\ne r$）'],
        ['盲判阶段：非处置查询', '可读', '遮蔽', '遮蔽'],
        [r'处置查询 $A_r$（所有阶段）', '可读', '可读', '隔离'],
        ['揭示阶段：作答查询', '可读', '按该分支概率门控', '按该分支概率门控'],
    ]
    rh = .41
    for r, vals in enumerate(rows):
        x = x0
        y = y0+(3-r)*rh
        for c, val in enumerate(vals):
            fill = GREY_L if r == 0 else ('white' if c == 0 else (PURPLE_L if val in ('遮蔽','隔离') else TEAL_L))
            ax.add_patch(Rectangle((x, y), widths[c]-.05, rh-.025, facecolor=fill, edgecolor='none'))
            txt(ax, x+.12 if c == 0 else x+(widths[c]-.05)/2, y+(rh-.025)/2,
                val, 10.4, INK if c == 0 else MUTED,
                weight='bold' if r == 0 else 'normal', ha='left' if c == 0 else 'center')
            x += widths[c]
    txt(ax, .89, 1.27, '所有可读关系仍受因果掩码限制；r 与 s 为同组不同分支。最高分支为 NONE 或并列时，本组不做门控。', 9.3, MUTED)
    txt(ax, .57, .68, '缓存前提：指南位于病历之前，且指南编码不依赖患者。概率相近时减弱门控；自信错误仍可能被强化。', 10.4, MUTED)
    save(fig, out, 'fig02_aster')


def fig03(out):
    fig, ax = canvas(16, 10)
    heading(ax, 16, 10, 3, '五卷配对评测与同信息比较',
            '虚构阈值示例：患者 t = 4 个月；原条件为 R1：t < 6，R2：t ≥ 6。Y 为原处置，Z 为替换处置。')
    widths = [1.63, 3.10, 1.60, 1.60, 2.03, 4.14]
    labels = ['评测卷', '条件改动', 'R1 处置', 'R2 处置', '金标分支与处置', '检验目标']
    rows = [
        ['NC 原规则', '无', 'Y', 'X', 'R1 → Y', '原生遵循'],
        ['AC 适用改写', '无', 'Z', 'X', 'R1 → Z', '适用处置与先验冲突'],
        ['IC 不适用改写', '无', 'Y', 'Z', 'R1 → Y', '不适用改写的阴性对照'],
        ['SW 处置交换', '无', 'X', 'Y', 'R1 → X', '原处置成为兄弟分支诱饵'],
        ['CS 条件漂移', '阈值 6 → 3；R2 适用', 'Y', 'X', 'R2 → X', '条件翻转与旧分支诱饵'],
    ]
    x0, top, rh = .60, 8.38, .61
    for i, row in enumerate([labels]+rows):
        x = x0
        y = top-(i+1)*rh
        for j, val in enumerate(row):
            fill = BLUE_L if i == 0 else ('white' if i%2 == 0 else GREY_L)
            if j == 4 and i: fill = TEAL_L
            ax.add_patch(Rectangle((x,y),widths[j]-.035,rh-.03,facecolor=fill,edgecolor='none'))
            txt(ax, x+.13 if j in (0,1,5) else x+(widths[j]-.035)/2,
                y+(rh-.03)/2,val,11.0,BLUE if i == 0 else (TEAL if j == 4 else INK),
                'bold' if i == 0 else 'normal', ha='left' if j in (0,1,5) else 'center')
            x += widths[j]
    txt(ax, .63, 4.45, '图 1 对应 SW。前四卷保持患者、问题与条件一致；CS 需验证归属翻转。所有派生卷按同一底稿配对。', 11, MUTED)

    specs = [(.60, BLUE, BLUE_L, '诊断对照',
              ['盲判与明判：仅输出分支编号',
               '诱饵：熟悉 / 陌生 / 中性代号',
               '控制词元长度、顺序与选项',
               '按闭卷置信度分层']),
             (5.65, PURPLE, PURPLE_L, '同信息基线',
              ['盲判信息：附注、裁剪、高亮、重排',
               '加入证据员：使用相同原子状态',
               '选项映射存在时比较 Map / Fuse',
               '匹配调参预算、成本与原生性能']),
             (10.70, TEAL, TEAL_L, '划分与报告',
              ['按决策树或指南隔离开发 / 测试',
               '锁定测试集后重算全部派生指标',
               '报告伤害率、覆盖率与噪声敏感性',
               '同任务、同输出复用评分；报告区间'])]
    for x, edge, fill, title, lines in specs:
        box(ax, x, 1.30, 4.70, 2.76, fill, edge)
        txt(ax, x+.29, 3.62, title, 15.0, edge, 'bold')
        for i, line in enumerate(lines):
            txt(ax, x+.29, 3.10-i*.45, line, 10.5)
    txt(ax, .62, .68, '此图定义实验设计，不预设效应方向。数据规模、显著性、跨模型复现、延迟与成本均需实测。', 10.6, MUTED)
    save(fig, out, 'fig03_protocol')


def fig04(out):
    fig = plt.figure(figsize=(15, 8.6))
    back = fig.add_axes([0, 0, 1, 1], xlim=(0, 15), ylim=(0, 8.6))
    back.axis('off')
    heading(back, 15, 8.6, 4, '对数门控的解析行为',
            '仅假定两个分支的概率，展示公式作用；曲线不是实验结果，也不表示准确率提升。')
    txt(back, 7.5, 6.72,
        r'$B_{\mathrm{comp}}=\lambda\log(p_{\mathrm{comp}}/p_{\mathrm{top}}),\qquad'
        r'\exp(B_{\mathrm{comp}})=(p_{\mathrm{comp}}/p_{\mathrm{top}})^{\lambda}$',
        16, PURPLE, ha='center')
    a = fig.add_axes([.075, .255, .40, .455])
    b = fig.add_axes([.565, .255, .40, .455])
    cases = [(.9, .1, PURPLE, '概率间隔较大：0.90 / 0.10'),
             (.55, .45, TEAL, '概率间隔较小：0.55 / 0.45')]
    x = np.linspace(0,4,401)
    x0 = np.linspace(0,8,401)
    for pt, pc, col, lab in cases:
        a.plot(x, (pc/pt)**x, color=col, lw=2.8, label=lab)
        b.plot(x0, (pc/pt)**(x0*(pt-pc)), color=col, lw=2.8, label=lab)
    for ax, right in [(a,False),(b,True)]:
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(LINE)
        ax.set_ylim(-.025,1.045)
        ax.set_yticks(np.linspace(0,1,6))
        ax.grid(axis='y', color=LINE, lw=.7, zorder=0)
        ax.tick_params(length=0, pad=7, labelsize=10)
        ax.set_xlim(0,8 if right else 4)
        ax.set_xticks(np.arange(0,9,2) if right else np.arange(0,5,1))
        ax.set_xlabel(r'基础强度 $\lambda_0$' if right else r'有效强度 $\lambda$', fontsize=12, labelpad=10)
        ax.set_ylabel('竞争处置相对权重的乘子', fontsize=11.5, labelpad=11)
        ax.legend(loc='upper right' if not right else 'lower right', frameon=False, fontsize=10.1,
                  bbox_to_anchor=(1, .85 if not right else .07))
    a.set_title('A  固定门控强度', loc='left', fontsize=14, pad=15, color=INK)
    b.set_title(r'B  自适应强度：$\lambda=\lambda_0(p_{(1)}-p_{(2)})$',
                loc='left', fontsize=13, pad=15, color=INK)
    # Formula-derived markers, deliberately not experiment points.
    base = 2
    for pt, pc, col, _ in cases:
        y=(pc/pt)**(base*(pt-pc))
        b.scatter([base],[y],s=38,color=col,zorder=5)
        b.annotate(f'{y:.3f}',(base,y),xytext=(10,10 if pt<.8 else 13),
                   textcoords='offset points',fontsize=10.5,color=col)
    txt(back, .67, 1.25, '解析含义', 12, weight='bold')
    txt(back, 2.10, 1.25, '概率相近时，自适应门控更温和；最高概率分支的门控偏置为 0。', 11.3)
    txt(back, .67, .77, '边界', 12, weight='bold')
    txt(back, 2.10, .77, '这里固定概率、查询与键；不包含后续表征变化。权重单调不推出准确率单调。', 11.3, MUTED)
    save(fig, out, 'fig04_gate')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'docs/ASTER/figures')
    args=parser.parse_args()
    configure()
    for make in (fig01, fig02, fig03, fig04):
        make(args.output_dir)
    metadata = {
        'kind': 'scientific_schematic_and_analytic_illustration',
        'empirical_results': False,
        'input': 'user manuscript supplied on 2026-10-26',
        'figures': ['fig01_tlbs', 'fig02_aster', 'fig03_protocol', 'fig04_gate'],
        'figure3_example': {
            'shared_with_figure1': True,
            'figure1_volume': 'SW',
            'native_treatment': 'Y',
            'replacement_treatment': 'Z',
            'volumes': {
                'NC': {'R1': 'Y', 'R2': 'X', 'gold': 'R1->Y'},
                'AC': {'R1': 'Z', 'R2': 'X', 'gold': 'R1->Z'},
                'IC': {'R1': 'Y', 'R2': 'Z', 'gold': 'R1->Y'},
                'SW': {'R1': 'X', 'R2': 'Y', 'gold': 'R1->X'},
                'CS': {'R1': 'Y', 'R2': 'X', 'gold': 'R2->X', 'threshold_change': '6->3'},
            },
            'score_cache_scope': ['task_case', 'guideline', 'judge_configuration', 'output'],
        },
        'figure4_assumptions': {'p_pairs': [[.9,.1],[.55,.45]],
            'quantity': '(p_comp/p_top)^lambda',
            'adaptive_lambda': 'lambda0*(p_top-p_comp)',
            'fixed_probability_query_key': True},
        'font': 'Noto Sans CJK SC',
        'formats': ['svg','pdf','png'],
    }
    (args.output_dir/'figure_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
