# 方法修订图表（2026-11-22）

图1为当前节点示例与2×2配对角色表。图2使用主稿附录表C2的历史16维配置重绘；数据逐行保留在 ../fig02_pilot_tradeoff.csv，原结果见 [归档指标](https://github.com/chunxi-alpc/TRIAGE/blob/08fc9dc6ef65236f697c10278648a024c89541a0/reports/aster_cancel_story_20261109/meeting/real_completion/metrics.json)。

图2横轴为历史的 A/P RMS 归一化预算。第3.4节修订方法使用 C/F 中位绝对分数差定标，新图3待获得对应实测后绘制。

render.py 与 fig02_spec.json 可复现两图，依赖 matplotlib 和 numpy。plot_publication_figure.py 来自 engineering-figure-banana 技能的数值绘图工具。可用 --font 指定本机中文字库；字体文件不纳入仓库。程序导出 SVG、PDF 和 PNG，SVG使用字形路径以保持显示一致。绘图在服务器 gcx261014 环境执行。

```bash
python render.py --font /path/to/CJK-font.ttc
```
