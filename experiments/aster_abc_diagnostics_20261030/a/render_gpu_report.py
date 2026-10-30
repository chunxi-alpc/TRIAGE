"""Attach completed GPU control evidence to the A report; never claim pending work ran."""
import argparse
import json
from pathlib import Path

def metric_text(metric):
    if not metric or metric.get('estimate') is None:return 'NA'
    result=f'{metric["estimate"]:.5f}'
    if metric.get('ci95'):result+=f' [{metric["ci95"][0]:.5f}, {metric["ci95"][1]:.5f}]'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--report',default='experiments/aster_abc_diagnostics_20261030/a/audit.md');p.add_argument('--copy-to');args=p.parse_args()
    gpu=Path(args.gpu);status=json.loads((gpu/'status.json').read_text())
    if status.get('status')!='complete':raise RuntimeError('GPU controls are incomplete; keep report pending')
    s=json.loads((gpu/'summary.json').read_text());lines=['## GPU实测结果','',
        f'已完成 {s["completed_forward_passes"]}/{s["planned_forward_passes"]} 次前向，{s["pairs"]} 对、{s["sources"]} 来源；技术失败 {len(s["technical_failures"])}。实际用时 {s.get("elapsed_seconds",0):.2f} 秒，峰值显存 {s.get("peak_memory_gb",0):.2f} GB。Qwen3-8B固定权重、bf16/eager，无训练/生成。',
        '', '以下为来源簇配对描述（2来源，一律不能视为总体特异性通过）。数字后为来源bootstrap 95%区间。',
        '', '|KV头|改写差分余弦|改写投影相对变化|处置交换差分余弦|处置交换投影相对变化|无关编号/决定事实投影比|', '|---|---|---|---|---|---|']
    heads=sorted({k.split('/')[0] for k in s['metrics']})
    for head in heads:
        a=s['metrics'].get(head+'/paraphrase',{});b=s['metrics'].get(head+'/action_swap',{});c=s['metrics'].get(head+'/irrelevant_a_minus_b',{})
        lines.append('|'+head+'|'+'|'.join([metric_text(a.get('direction_cosine')),metric_text(a.get('relative_projection_change')),metric_text(b.get('direction_cosine')),metric_text(b.get('relative_projection_change')),metric_text(c.get('nuisance_to_decisive_ratio'))])+'|')
    lines+=['', '等长数字记录号提供无关事实对照；年龄句改写与前置说明可能改变上游词元位置，处置交换还改变目标词元内容/数量，这些混杂已在records.jsonl保留。该小规模诊断只检验现有年龄规则，非年龄候选仍未裁定；不能据此宣称通用临床语义方向或最终任务收益。']
    lines+=['', '位置对照：', '', '|KV头|位置对照差分余弦|位置对照投影相对变化|原始谱投影减均值投影|', '|---|---|---|---|']
    for head in heads:
        position=s['metrics'].get(head+'/position_control',{});base=s['metrics'].get(head+'/base',{})
        lines.append('|'+head+'|'+'|'.join([metric_text(position.get('direction_cosine')),metric_text(position.get('relative_projection_change')),metric_text(base.get('spectral_minus_mean_projection'))])+'|')
    parts_path=gpu.parent/'gpu_token_parts/summary.json'
    if parts_path.exists():
        parts=json.loads(parts_path.read_text())
        lines+=['', '### 处置交换的编号/处置词元分解（事后CPU诊断，无额外GPU前向）', '', '|KV头|编号差分余弦|编号投影相对变化|处置差分余弦|处置投影相对变化|', '|---|---|---|---|---|']
        for head in heads:
            rid=parts['metrics'].get(head+'/action_swap/rule_id',{});act=parts['metrics'].get(head+'/action_swap/action',{})
            lines.append('|'+head+'|'+'|'.join([metric_text(rid.get('direction_cosine')),metric_text(rid.get('relative_projection_change')),metric_text(act.get('direction_cosine')),metric_text(act.get('relative_projection_change'))])+'|')
        lines+=['', '交换效应主要集中在处置词元：编号投影变化−12.54%至+4.83%，处置投影增加20.69%至151.38%。目标词元组成和上下文改变共同影响整体指标；不能把整体不稳全部归为规则适用性失败，也不能声称处置映射不变性成立。编号/处置应分开报告。']
    lines+=['', '当前样本支持年龄决定事实投影强于无关记录号、对本次等义改写相对稳定；整体处置交换没有满足预设±20%等价区间，原因含处置文本组成/上下文。谱步骤尚无独立增益证据。2来源一律描述；通用临床语义特异性仍需新来源人工裁定实验。']
    if s['technical_failures']:
        lines+=['','技术失败：']+[f'- {x}' for x in s['technical_failures']]
    report=Path(args.report);text=report.read_text().split('## GPU实测结果')[0]+'\n'.join(lines)+'\n';report.write_text(text)
    if args.copy_to:Path(args.copy_to).write_text(text)
    print(json.dumps({'status':'complete','report':str(report),'completed_forward_passes':s['completed_forward_passes']}))
if __name__=='__main__':main()
