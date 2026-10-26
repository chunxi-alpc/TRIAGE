#!/usr/bin/env python3
"""Rebuild the four ASTER paper figures and their full-size review PDF.

Run from the repository root:
    python scripts/generate_aster_figures.py
No experiment data, network access, or external fonts are required.
"""
from pathlib import Path
import argparse
import json
import importlib.metadata

from aster_figures.overview import fig01, fig02
from aster_figures.protocol_gate import fig03, fig04
from aster_figures.proof import make_proof

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / 'docs' / 'ASTER' / 'figures'


def metadata():
    figures = [
        ('fig01_tlbs', 2.55, 'fictional task and hypothesized failure path'),
        ('fig02_aster', 4.30, 'attention mask and symbolic bias operators'),
        ('fig03_protocol', 2.25, 'five paired counterfactual guideline variants'),
        ('fig04_gate', 2.25, 'analytic relative weight multipliers'),
    ]
    return {
        'revision': 2,
        'kind': 'scientific_schematic_and_analytic_illustration',
        'empirical_results': False,
        'input': 'ASTER manuscript method specification; no experiment metrics',
        'figures': [{'stem': name, 'size_inches': [7.0, height], 'content': kind}
                    for name, height, kind in figures],
        'typography': {'font': 'DejaVu Sans', 'minimum_label_pt': 7.2,
                       'body_label_pt': 8.0, 'panel_heading_pt': 9.1,
                       'pdf_font_type': 'embedded TrueType', 'svg_editable_text': True},
        'formats': ['svg', 'pdf', 'png'],
        'png_dpi': 450,
        'proof': {'file': 'figures_review.pdf', 'pages': 4,
                  'figure_width_inches': 7.0, 'captions': 'captions.en.md'},
        'figure2_scope': {
            'matrix': 'representative role tokens; exact categorical mask, not attention values',
            'blind_readout': 'fixed non-action B before branch ID is written',
            'clerk_paths': ['post-prefix condition/evidence union highlighting', 'soft veto'],
            'reveal_example': 'p1 > max(p2, pNONE); emphasis additionally requires p1 >= tau',
            'persistent_constraints': ['causal mask', 'cross-action-fragment isolation'],
            'cache': 'fixed patient-independent guideline prefix only',
        },
        'figure3_example': {
            'patient': {'relapse_months': 4, 'prior_H': False},
            'figure1_volume': 'SW', 'prior_favored_action': 'Y',
            'volumes': {
                'NC': {'R1': 'Y', 'R2': 'X', 'gold': 'R1->Y', 'threshold': 6},
                'AC': {'R1': 'Z', 'R2': 'X', 'gold': 'R1->Z', 'threshold': 6},
                'IC': {'R1': 'Y', 'R2': 'Z', 'gold': 'R1->Y', 'threshold': 6},
                'SW': {'R1': 'X', 'R2': 'Y', 'gold': 'R1->X', 'threshold': 6},
                'CS': {'R1': 'Y', 'R2': 'X', 'gold': 'R2->X', 'threshold': 3},
            },
        },
        'figure4_assumptions': {
            'p_pairs': [[0.90, 0.10], [0.55, 0.45]],
            'none_probability': 0,
            'quantity': '(p_comp/p_top)^lambda',
            'adaptive_lambda': 'lambda0*(p_top-p_comp)',
            'fixed_probability_and_pre_gate_scores_and_other_biases': True,
            'not_an_attention_share_or_performance_measure': True,
        },
        'renderer_versions': {name: importlib.metadata.version(name)
                              for name in ('matplotlib', 'numpy', 'Pillow', 'reportlab', 'pypdf')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for draw in (fig01, fig02, fig03, fig04):
        draw(args.output_dir)
    make_proof(args.output_dir, FIGURE_DIR / 'captions.en.md')
    (args.output_dir / 'figure_metadata.json').write_text(
        json.dumps(metadata(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Created 4 SVG, 4 PDF, 4 PNG, a four-page PDF proof, and metadata in {args.output_dir}')


if __name__ == '__main__':
    main()
