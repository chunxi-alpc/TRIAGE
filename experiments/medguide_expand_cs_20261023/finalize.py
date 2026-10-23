"""Validate identities and rule execution, export prompts privately and report publicly."""
import argparse,collections,hashlib,json
from pathlib import Path
from prepare import sha,write,execute,atom,evaluate,build

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--private',required=True);p.add_argument('--output',required=True);p.add_argument('--dev',required=True);a=p.parse_args();out=Path(a.output);private=Path(a.private);m=json.loads((out/'manifest.json').read_text());cohort=[json.loads(s) for s in (private/'cohort.jsonl').read_text().splitlines()];byid={r['id']:r for r in cohort};cs=[json.loads(s) for s in (out/'cs_selected.jsonl').read_text().splitlines()];paths=json.loads((Path(a.source)/'all_paths.json').read_text());preds=[json.loads(s) for f in sorted(out.glob('closed_book_*.jsonl')) for s in f.read_text().splitlines()];dev={json.loads(s)['id'] for s in Path(a.dev).read_text().splitlines()};manifests=[json.loads(f.read_text()) for f in out.glob('closed_book_*.manifest.json')]
 assert len(cohort)==800 and len(byid)==800 and len(preds)==800 and len({r['id'] for r in preds})==800
 assert {r['id'] for r in preds}==set(byid) and not set(byid)&dev
 assert len(manifests)==2 and all(r['complete'] and r['input_sha256']==sha(private/'cohort.jsonl') for r in manifests)
 for r in preds:assert r['gold']==byid[r['id']]['original_answer'] and r['stratum']==('K' if r['prediction']==r['gold'] else 'U')
 interventions=[]
 for r in cs:
  source=byid[r['id']];original=paths[r['tree']];j=r['depth'];prefix=r['source_path'][:j]
  for variant,t in [('original',r['old_threshold']),('CS',r['new_threshold']),('preserve',r['preserving_threshold'])]:
   execution=execute(original,r['source_path'],j,r['nodes'],r['patient_value'],t);label=next(k for k,v in source['options'].items() if v==execution['action'])
   altered=[]
   for path in original:
    q=path.copy()
    if q[:j]==prefix and q[j] in r['nodes']:
     spec=atom(q[j]);start,end=spec['span'];q[j]=q[j][:start]+spec['op']+f'{t:g}'+(' '+spec['unit'] if spec['unit'] else '')+q[j][end:]
    altered.append(q)
   # Evaluate rewritten scalar expressions separately from the selected gold.
   selected_nodes={q[j] for q in altered if q[:j]==prefix and atom(q[j]) and evaluate(atom(q[j])['op'],r['patient_value'],atom(q[j])['threshold'])}
   assert len(selected_nodes)==1
   rules='\n\n'.join('IF the clinical sequence is: '+' -> '.join(q[:-1])+'\nTHEN the next step is: '+q[-1]+'.' for q in altered)
   interventions.append({'id':r['id'],'variant':variant,'gold_answer':label,'threshold':t,'prompt_text':'Use the supplied decision rules for this offline hypothetical task.\n\nDecision rules:\n'+rules+'\n\n'+source['closed_book_prompt'],'execution_scope':r['execution_scope']})
  assert r['label_flipped'] and r['branch_flipped'] and r['preserving_label_ok']
 write(private/'cs_prompts.jsonl',interventions);write(out/'cs_variant_manifest.jsonl',[{k:r[k] for k in ('id','variant','gold_answer','threshold','execution_scope')} for r in interventions]);write(out/'strata.jsonl',sorted(preds,key=lambda r:r['id']))
 # Refresh discovery only, preserving inference cohort hash and frozen selection.
 from prepare import build
 import pyarrow.parquet as pq
 rows=pq.read_table(Path(a.source)/'train.parquet').to_pylist();inventory,pairs,allcs,audit=build(rows,paths);write(out/'threshold_nodes.jsonl',inventory);m['threshold_nodes']=len(inventory)
 status={'cases':800,'K':sum(r['stratum']=='K' for r in preds),'U':sum(r['stratum']=='U' for r in preds),'invalid_predictions':sum(r['prediction'] is None for r in preds),'trees_with_cases':len({r['tree'] for r in cohort}),'trees_without_source_rows':m['trees_without_selected_rows'],'cs_cases':len(cs),'cs_trees':dict(collections.Counter(r['tree'] for r in cs)),'cs_K':sum(next(p for p in preds if p['id']==r['id'])['stratum']=='K' for r in cs),'cs_label_flips':sum(r['label_flipped'] for r in cs),'cs_preserve_ok':sum(r['preserving_label_ok'] for r in cs),'cs_variants':len(interventions),'cohort_sha256':sha(private/'cohort.jsonl'),'cs_prompts_sha256':sha(private/'cs_prompts.jsonl'),'development_overlap':0,'duplicate_ids':0,'peak_gib_per_shard':[x['peak_gib'] for x in manifests],'seconds_per_shard':[x['seconds'] for x in manifests]}
 m['K_definition']='Qwen3-8B parsed greedy closed-book answer equals original answer';m['result']=status;(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));ext=json.loads((out/'external_summary.json').read_text())
 assert ext['medcalc']['computed']==20 and ext['medcalc']['within_source_tolerance']==20 and ext['medcalc']['preserve_ok']==20 and not ext['medcalc']['errors']
 if 'pairing_failures' in ext['driftmed']:assert not ext['driftmed']['pairing_failures']
 text=f'''# MedGUIDE 扩样与 CS 阈值实验

基于 PR #23 HEAD `3e4fac744bf63564cd32045f15c102390ac57c28`，独立分支 `codex/medguide-expand-cs-20261023`。全部计算在 115 服务器的 gcx261014 环境完成。未覆盖原 PR 结果。

## 样本与 K 组

原始表 7,747 例；对作者发布的 55 棵树逐一分层，最终抽取 800 个唯一 ID，实际覆盖 {status['trees_with_cases']} 棵树。`bre_19`、`cns_23` 在原表没有病例，无法从这份表补齐。每树原始及抽样数量见 manifest.json。先纳入阈值 CS 候选病例（85 例，其中 1 例随后因患者只给出范围而未纳入可执行 CS），再按固定种子 20261023 在各树轮流补足；选择不使用模型输出。排除既有 40 例开发集，重叠和重复 ID 均为 0。不是按原始树频率比例抽样，不能把总体比例当作原分布估计。

Qwen3-8B 闭卷实测：K{status['K']}、U{status['U']}；K 定义沿用 PR23：原问题自由生成答案解析后与原金标一致。800 例均完成，无法解析 {status['invalid_predictions']} 例。GPU 0、1 各 400 例，峰值显存分别 {status['peak_gib_per_shard'][0]:.2f}、{status['peak_gib_per_shard'][1]:.2f} GiB，用时分别 {status['seconds_per_shard'][0]:.1f}、{status['seconds_per_shard'][1]:.1f} 秒。本轮使用 SDPA，旧实验使用 eager，可能有浮点差异；本轮 K 独立测定，不拼接旧结果。没有重跑旧 A/B/C/D/CAD 全套推理，不把 K 的达标解释为干预有效。

## CS 自动发现与重算

扫描全部 55 棵树的条件路径，发现 {m['threshold_nodes']} 个含数值比较或时间/尺寸表达式的上下文节点，提取 {len(pairs)} 对可自动执行的互补单阈值条件。扫描不局限于最后一层。节点清单保留树、前缀和原文；多条件、区间、动作中的数字只记录，不自动当作完整逻辑。

沿真实来源患者路径找到 {len(cs)} 例可执行的 CS，全部纳入 800 例，均属于 `nscl_3`；其中 K{status['cs_K']}。从患者原文提取唯一尺寸值，保留证据位置；新阈值设在患者值另一侧，并同步改写同前缀下的互补条件。规则引擎重新选择有效路径和选项标签，{status['cs_label_flips']} 例分支及标签翻转；阈值仍留在原侧的对照 {status['cs_preserve_ok']} 例标签保留。已生成原始、CS、保留三种版本共 {len(interventions)} 条可推理题及其金标。公开逐题金标清单，完整提示存放服务器 private/cs_prompts.jsonl，哈希见 summary.json，可用 finalize.py 重建。

这是**以原路径已知条件为依据的规则重算**：未改节点沿用来源路径事实；改写后如需患者没有提供的下游条件，或存在多个不同可选动作，就明确记为无法唯一重算。不是从患者自然语言独立推断整棵树。胰腺复发时间分支同一条件下有多种动作，不能直接把任一兄弟动作当作唯一新金标；因此本轮没有得到时间阈值 CS。完整逐病例原因见 cs_audit.jsonl，不能把这一结果概括成跨疾病、跨树的 CS 成效。阈值改写是合成实验，不能当作临床指南更新。

## 外部来源与可比性

MedCalc-Bench 固定到 `83a1a56d6b53f8d4c0e03f0ba9699becf524a595`，读取用户指定 ncbi v1.2 测试集 1,100 例。20 例问题明确包含 BMI 条件；另 20 例仅要求调整体重公式，分开记录，未硬套 BMI 分支。20/20 例原数值重算落在来源允许区间，17 例修改 BMI 25 分界后分支与数值改变，20/20 例保留对照通过。BMI<18.5 的低体重分支保持原规则，因此 3 例不因这一阈值变化而翻转。公式、单位换算、来源行号、Note ID、原金标和新数值均保存；衍生记录遵循来源 CC-BY-SA-4.0。这是数值计算任务，独立报告，不混入 MedGUIDE 选项准确率。数据卡指向更新的 Verified 版本，本轮不混用版本。

DriftMed 官方数据卡说明为正确建议及人工构造的错误版本，不能直接称为真实历史阈值更新。已固定目标修订 `5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e` 并提供下载和数值差异提取脚本；实际成功获取 ConflictMed_v2 的 {ext['driftmed'].get('scenario_rows',0)} 条场景，按官方偏移配对规则验证 {ext['driftmed'].get('validated_pairs',0)} 对，诊断与阈值调整 {ext['driftmed'].get('diagnostic_threshold_pairs',0)} 对，其中 {ext['driftmed'].get('pairs_with_numeric_changes',0)} 对含数字变化。逐对来源 ID、类别、数字、文本哈希见 driftmed_threshold_candidates.jsonl。advice_pairs 文件未下载成功，已记录失败原因；没有把场景正文公开转发。数字变化也可能来自背景文字，仍是待提取可执行谓词的候选，不计作已验证 CS，更不当作真实历史指南更新。详见 external_sources.json 与 external_summary.json。

## 验证与复现

tests.txt 包含规则比较边界、互补条件、患者证据、歧义、缺失下游条件、标签翻转与保留等测试。finalize.py 还检查 800 ID 完整性、开发集排除、预测与金标对应关系、两分片输入哈希、CS 实际结果、改写后互补分支唯一性。各输出 SHA256 在 artifact_hashes.json。

服务器工作目录：`/home/gcx/TRIAGE-worktrees/medguide-expand-cs`；原始数据：`/home/gcx/aster_medguide_20261022/private`；本轮私有文件及运行日志：`/home/gcx/medguide_expand_cs_20261023/private`。运行步骤见 README.zh.md 与 reproduce.sh。GPU 任务已结束，无需继续监控。
'''
 (out/'REPORT.zh.md').write_text(text);(out/'artifact_hashes.json').write_text(json.dumps({f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='artifact_hashes.json'},indent=2));print(json.dumps(status))
if __name__=='__main__':main()
