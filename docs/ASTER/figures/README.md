# ASTER paper figures

> **版本提示（2026-10-27）：本目录保存 2026-10-26 图稿。** 图 2 的阶段掩码／证据员与图 4 的对数门控曲线已退出当前主方法；不要将其插入新稿作为现行算法图。图 1 与图 3 仍可说明任务与五卷协议。当前三模块流程图见[论文工作稿](../manuscript.zh.md)与[项目首页](../../../README.md)。以下尺寸、公式、图注和复现命令均按旧版解释。

[Mainline index](../README.md) · [Working manuscript](../manuscript.zh.md) · [English captions](captions.en.md) · [LaTeX inclusion](figures.tex)

The four figures cover the proposed failure mode, ASTER's computational operators, the paired counterfactual protocol, and an analytic gate relationship. Figures 1-3 are design schematics; Figure 4 evaluates declared mathematical inputs. They contain no measured ASTER outcomes.

## Files and publication dimensions

All figure text is English. Each source figure is 7.0 inches wide.

| Figure | Content | Width × height | Preview | Vector files |
|---|---|---|---|---|
| 1 | Treatment-led branch selection, fictional SW case | 7.0 × 2.55 in | [PNG](fig01_tlbs.png) | [PDF](fig01_tlbs.pdf) · [SVG](fig01_tlbs.svg) |
| 2 | BLIND mask, applicability readout, and REVEAL controls | 7.0 × 4.30 in | [PNG](fig02_aster.png) | [PDF](fig02_aster.pdf) · [SVG](fig02_aster.svg) |
| 3 | NC / AC / IC / SW / CS with aligned sibling trees | 7.0 × 2.25 in | [PNG](fig03_protocol.png) | [PDF](fig03_protocol.pdf) · [SVG](fig03_protocol.svg) |
| 4 | Fixed and confidence-adaptive logarithmic gate multipliers | 7.0 × 2.25 in | [PNG](fig04_gate.png) | [PDF](fig04_gate.pdf) · [SVG](fig04_gate.svg) |

[**Four-page figure-and-caption proof: figures_review.pdf**](figures_review.pdf) places one figure and its English caption on each page, retaining the figure's actual 7.0-inch width. It is the fifth PDF produced alongside the four individual figures. Inspect it at 100% scale when assessing type size and spacing.

![ASTER operators](fig02_aster.svg)

## Reading the diagrams

**Figure 1.** The action-visible and ASTER routes share the patient facts, conditions, and action tokens. The selection-stage mask changes. The example is the SW variant of Figure 3: $t=4$ months with no prior H supports $r_1\to X$, while familiar Y is attached to the inapplicable sibling $r_2$. The displayed error and intended ASTER route are hypothetical.

**Figure 2.** The matrix is a representative-token view of the operator, not a complete list of prompt positions. Its row is the query and its column is the key. The specification covers all non-action queries in BLIND, including neutral N, question Q, rule-ID, and other positions omitted from the matrix. Any action mention in a question, option, or footnote is tagged as an action. The outlined A-to-A cell marks isolation across distinct action fragments; this isolation persists in both phases together with the causal mask.

The highlighted B row is a fixed non-action readout position before writing the branch ID. Purple components belong to ASTER-full: the clerk sees only conditions and patient text, provides atomic states/confidence/evidence, supports joint in-place highlighting of condition and evidence spans for post-prefix queries, and supplies the violations used by the soft veto. Joint highlighting does not specify one-to-one condition-evidence attention edges. ASTER-self omits these components. The adjusted distribution determines the written branch ID and branches to both Gate and Emph. The shown case assumes $p_1>\max(p_2,p_{\mathrm{NONE}})$; ties or a top NONE disable both controls, and emphasis additionally requires $p_{\max}\ge\tau$.

The gate targets rule-action keys across all layers and heads. Emphasis is bounded and restricted to the selected layer window; $I,C,A,E$ denote the selected rule ID, conditions, action, and supporting evidence. Default ASTER does not automatically gate action mentions in question/option spans during REVEAL. Cache reuse applies only to a causal guideline prefix with fixed content, token positions, role annotations, and patient-independent computation. Patient-dependent highlighting and probabilities must not modify that prefix.

**Figure 3.** Only the specified action or threshold changes between aligned trees. The shared no-H condition is suppressed in the miniature trees. The five gold routes are $r_1\to Y$, $r_1\to Z$, $r_1\to Y$, $r_1\to X$, and $r_2\to X$. Y is a stipulated prior-favored action for this illustration; model-specific preference still requires measurement in the evaluation protocol.

**Figure 4.** The y-axis is the multiplicative change to the competing-to-top unnormalized action-weight ratio. The two input distributions are declared analytic examples, with zero NONE mass. The curves do not measure task performance, empirical robustness, or the branch's share of all context attention.

The complete captions are in [captions.en.md](captions.en.md) and are mirrored in [figures.tex](figures.tex). Each caption is under 200 words.

## Format and reproduction

| Output | Specification |
|---|---|
| PDF | Vector artwork with embedded TrueType fonts |
| SVG | Vector shapes and editable text |
| PNG | 450 dpi raster export |
| Language and fonts | English labels; no Chinese font dependency |
| Source layout | 7.0-inch width; figure-specific heights listed above |

From the repository root, install the figure dependencies and regenerate:

```bash
python -m pip install -r requirements-figures.txt
python scripts/generate_aster_figures.py
```

The [generator](../../../scripts/generate_aster_figures.py) writes the four figure sets and the four-page review PDF to this directory. It does not consume experiment result tables. The analytic parameters and export settings are recorded in [figure_metadata.json](figure_metadata.json). Dependency declarations are in [requirements-figures.txt](../../../requirements-figures.txt).

For a two-column LaTeX manuscript compiled from the repository root, add to the preamble:

```latex
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\graphicspath{{docs/ASTER/figures/}}
```

Then include the four `figure*` environments:

```latex
\input{docs/ASTER/figures/figures.tex}
```

Each uses `\includegraphics[width=\textwidth]{...pdf}`. Paths are relative to the compilation directory; a manuscript compiled from `paper/` needs `../docs/ASTER/figures/` instead. The source width is 7.0 inches, while the conference template controls the final `\textwidth`; inspect the rendered manuscript at its actual scale.

## Verified design references

These original papers informed operator placement, matched-case comparison, and aligned pathway layouts. They are design references, not evidence for ASTER's unmeasured claims. Page numbers are 1-based PDF page numbers in the linked versions. Main-conference publications are distinguished from workshop records and preprints.

| Original paper and status | Inspected figure | Design element used here |
|---|---|---|
| **Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs (PASTA)**, Zhang et al. **ICLR 2024 main conference, poster.** [Official record](https://openreview.net/forum?id=xZDWO0oejD) · [author PDF, arXiv v2](https://arxiv.org/pdf/2311.02262v2) | **Figure 1, p. 2** | Show the attention intervention at concrete token positions. ASTER Figure 2 locates its masks and biases at query-key entries rather than leaving them inside a generic model box. |
| **Spectral Attention Steering for Prompt Highlighting (SEKA)**, Li et al. **ICLR 2026 main conference.** [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/32ddf8ee55540539eb46083cd72361e8-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/32ddf8ee55540539eb46083cd72361e8-Paper-Conference.pdf) | **Figure 2, p. 4** | Separate the operator's inputs, modified tensor, and attention computation. ASTER Figure 2 distinguishes key-specific gating from bounded emphasis; it does not depict SEKA's key-editing operator as its own. |
| **Taming Knowledge Conflicts in Language Models (JuICE)**, Li, Chen, and Tong. **ICML 2025 main conference.** [Official proceedings](https://proceedings.mlr.press/v267/li25c.html) · [official PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25c/li25c.pdf) | **Figure 4, p. 5** | Use aligned computation stages and explicit transfer arrows to show what is carried between operations. ASTER Figure 2 carries a branch distribution into REVEAL and distinguishes that transfer from the reusable guideline prefix. |
| **MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models**, Li et al. **Inspected arXiv v1 (2025), with a NeurIPS 2025 GenAI4Health workshop poster record.** [Original PDF](https://arxiv.org/pdf/2505.11613v1) · [official workshop record](https://openreview.net/forum?id=Si6BICL2J9) | **Figure 2, p. 5**; **Figure 1, p. 3** | Couple a patient case to a visible guideline path and action. ASTER Figures 1 and 3 retain sibling-branch topology and mark the gold path. The workshop record is not a NeurIPS main-track designation. |
| **Evaluating Counterfactual Sensitivity to Patient Information in Medication-Safety Reasoning (MedPIC-Bench)**, Hou et al. **Inspected arXiv v1 preprint (2026).** [Original PDF](https://arxiv.org/pdf/2608.03028v1) · [original HTML](https://arxiv.org/html/2608.03028v1) | **Figure 1, p. 1**; **Figure 2, p. 3** | Preserve common case structure and localize the changed component. ASTER Figure 3 shares a patient and tree layout while marking only the action or threshold edit. Its template does not establish conference acceptance. |
| **Benchmarking Clinical Decision Pathway Adherence in Large Language Models (MEGA-CDP)**, Chen et al. **Inspected arXiv v1 preprint (2026).** [Original PDF](https://arxiv.org/pdf/2608.26592v1) · [original HTML](https://arxiv.org/html/2608.26592v1) | **Figure 3, p. 6**; **Figure 2, p. 3** | Align alternative pathways under a shared case so the differing branch is visible. ASTER Figure 1 separates selecting a branch from executing its action. |

## Figures that await empirical results

Blind/visible comparisons, action-familiarity effects, causal interventions, noise robustness, clerk coverage, ablations, and latency/cost plots require actual run artifacts and a locked test set. They are not generated from the manuscript's former placeholder values. If the measurements contradict a claim, revise the claim and its figure rather than preserving the expected conclusion.

