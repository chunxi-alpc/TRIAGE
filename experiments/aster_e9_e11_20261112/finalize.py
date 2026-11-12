from pathlib import Path
from .common import *
from .summarize import main as summarize
def main():
 if read(OUT/'e9_status.json').get('state')!='complete':raise RuntimeError('E9 is still running')
 summarize()
 p=REPO/'docs/ASTER/experiments.zh.md';s=p.read_text()
 old="| E9 | OV路径与M1机制分析 | **待执行** | 已有选头和激活修补证据；近期文献提供机制假设 | 比较选中头/随机头，开展头输出修补、OV干预及QK冻结 |"
 new="| E9 | OV路径与M1机制分析 | **冻结E1受控机制补充已完成** | 71对开发样本、双向、6条件、选中与层匹配随机头，共1,704条干预；历史50种子结果直接复用 | 见[机制报告](../../reports/aster_e9_e11_20261112/E9_REPORT.zh.md)；分别解释头选择、输出投影与注意力路由，不将分数变化当作任务提升 |"
 if old in s:s=s.replace(old,new)
 else:assert new in s
 old="| 是否依赖模型家族或后训练？ | E6、E8 | 五模型独立拟合；Gemma2 Base/IT与Qwen3 Base/最终后训练检查点 | 待新协议评价 |"
 # E6/E8 are outside this task and remain unchanged.
 old="| M1选择是否具有任务相关机制？ | E9 | 选中头与层匹配随机头；头输出激活修补、OV路径、QK冻结 | 待执行 |"
 new="| M1选择是否具有任务相关机制？ | E9 | 选中头与层匹配随机头；输出修补、OV值路径、QK冻结、注意力权重和输出投影对照 | 冻结E1的71对开发样本补充已完成，见E9报告；不等同于任务性能改善 |"
 if old in s:s=s.replace(old,new)
 a=read(OUT/'e9_statistics.json');b=read(OUT/'e11_statistics.json')
 head=next(x for x in a['contrasts'] if x['contrast']=='head_selection' and x['kind']=='all' and x['metric']=='abs_score_change_difference')
 marker='### 12.6 E9机制与E11效率补充（2026-11-12，Codex06）'
 if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
 s+=f"""
\n{marker}

本次只完成E9缺失机制条件与E11可运行方法，不训练新参数，不重复内部消融，不安排E10新实验，未修改论文正文。

[E9报告](../../reports/aster_e9_e11_20261112/E9_REPORT.zh.md)：固定E1的51头、16维参数；开发集A/P/C/F分别40/3/18/10对，20来源，按固定ID顺序取样。输出修补、OV值路径、QK冻结、仅注意力权重和两种匹配输出投影均保存干预前后原模型及E1最终分数。选中头减随机头的绝对分数变化按来源等权为{head['mean']:.4f}，95%区间[{head['ci95_low']:.4f}, {head['ci95_high']:.4f}]。该结果说明分数敏感性，不能直接作为任务收益；纠正和新增错误另列。

[E11报告](../../reports/aster_e9_e11_20261112/E11_REPORT.zh.md)：原模型、E1主方法、LEACE、LoFiT、LoRA、CAD在同一GPU 1、同一输入、批量1下完成512/2048 token两档测试，每档预热3次后测20次，共240次测量。记录预填充、端到端延迟、吞吐、显存、KV、逻辑前向和底层调用、训练参数、额外存储。CAD额外参考前向已计入。LEACE/LoFiT/LoRA为历史检查点的效率测量，不代替当前E1有效性审计。

复用已有extraction_performance.json、evaluation_performance.json，以及历史h2/h3、50种子随机头/随机子空间结果，没有重跑这些实验。E11的审计通过ITI和新增代表方法按用户安排等待Codex00/01及后续方法完成，只补缺行。

配置、预测、日志、统计、参数快照和复现命令均在[交付目录](../../reports/aster_e9_e11_20261112/README.zh.md)。E9起初使用GPU 0，其他任务开始共享后将未完成记录续跑到GPU 5；E11始终在GPU 1测量。
"""
 p.write_text(s,encoding='utf-8')
 write(OUT/'artifact_manifest.json',dict(workspace=str(REPO),files=[
  dict(path=str(f.relative_to(OUT)),bytes=f.stat().st_size)
  for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ('e9.lock','artifact_manifest.json')],
  experiment_scope=['E9','E11'],manuscript_modified=False))
 print(json.dumps(dict(E9='complete',E9_rows=a['prediction_rows'],E11='six methods complete; audited ITI and new representative pending',E11_measurements=b['measurements']),ensure_ascii=False))
if __name__=='__main__':main()
