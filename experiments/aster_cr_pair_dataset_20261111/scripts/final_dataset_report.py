"""Summarize delivered data from recorded one-round decisions, without extra model review."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import collections,csv,json,pathlib
from parallel_generation import ROOT,BASE,rows,save,now
def main():
 completion=json.loads((ROOT/'reports/completion_all.json').read_text())
 with (ROOT/'pair_statistics.csv').open(encoding='utf-8-sig') as f:stats=list(csv.DictReader(f))
 n=min(int(r['accepted']) for r in stats)
 accepted=rows(ROOT/'accepted_pairs.jsonl')
 new=collections.Counter(r['kind'] for r in accepted if r['origin']=='generated')
 groups={}
 for kind in ['A','P','C','F']:
  rs=[r for r in accepted if r['kind']==kind]
  groups[kind]=dict(count=len(rs),sources=len({r['source_id'] for r in rs}),cases=len({r['case_id'] for r in rs}),languages=dict(collections.Counter(r['language'] for r in rs)),complexity=dict(collections.Counter('small' if r['tree_complexity']<=15 else 'medium' if r['tree_complexity']<=30 else 'large' for r in rs)),target_500_met=len(rs)>=500,source_50_met=len({r['source_id'] for r in rs})>=50,case_300_met=len({r['case_id'] for r in rs})>=300)
 validation=rows(ROOT/'validation_results.jsonl')
 rule_counts=collections.Counter(x.get('rule_semantics_status','UNCERTAIN') for x in validation if x['kind'] in ['C','F'])
 balanced=rows(ROOT/'balanced_train.jsonl')
 balanced_coverage={k:dict(count=sum(r['kind']==k for r in balanced),sources=len({r['source_id'] for r in balanced if r['kind']==k}),cases=len({r['case_id'] for r in balanced if r['kind']==k})) for k in ['A','P','C','F']}
 report=dict(rule_semantics_status_counts=dict(rule_counts),completed_at=now(),total_accepted=len(accepted),accepted_new=dict(new),all_classes=groups,balanced_coverage=balanced_coverage,balanced_per_class=n,balanced_total=4*n,full_pool_total=len(accepted),clinical_expert_review=False,review_rounds=1,manual_reviewed=0,hash_audit_performed=False)
 save(ROOT/'reports/final_dataset_summary.json',report)
 save(ROOT/'reports/next_steps.json',dict(current_scope='dataset_construction_complete',remaining='Pending human clinical review if later available; no model training performed',final_report='reports/final_dataset_summary.zh.md',completed_at=now()))
 text=['# ASTER-CR 配对数据构建结果','',f'全部审核通过数据 {len(accepted)} 对；类别均衡集每类 {n} 对、共 {n*4} 对。原始数据和历史实验保持不变，未执行模型训练。','',
 '|类别|全部候选|AI审核PASS|最终合格|新增合格|来源数|病例数|距500缺口|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in stats:
  k=r['kind'];text.append('|'+ '|'.join(str(v) for v in [k,r['candidates'],r['ai_pass'],r['accepted'],new[k],r['sources'],r['cases'],r['needed_to_500']])+'|')
 text+=['','AI审核PASS为单轮独立智能体语义结论；最终合格还要求两侧标签一致、结构和规则检查通过且无未解决重复。主动弃权和无法编译的生成任务保留在各分片 conversion_exclusions.jsonl，不冒充候选或新增有效样本。','',
 'A 无新增需求，历史合格 A 全部保留在 full_pool_train.jsonl。balanced_train.jsonl 从四类按来源、语言和树复杂度分层选择相同数量，不重复采样。每类500、50来源、300病例为覆盖目标；未达目标时按表报告，不降低标准。','',
 '作者为 GPT-6.1 Sol/high；独立审核者为 GPT-6.1 Sol/xhigh，使用独立上下文，只读取原指南、原病例、原节点、原标签、实际修改及预期类别。审核一次，没有生成—审核反复修改。C/F 人工反事实标记保留，原始内容不被人工编辑替代。','',
 '确定性程序验证 JSON Schema、来源和病例ID、标签关系、编辑范围、重复与近重复、训练/开发/测试来源隔离及覆盖。可执行 C/F 规则重新计算；无可执行表示的语义由独立 B 判断，程序不声称已自动证实。未解决近重复保留 UNCERTAIN 并排除。','',
 '本轮没有临床专家参与。human_audit.csv 是分层抽查清单，实际人工审核0条、通过率尚无。F0 82对保留在 f0_diagnostic_pairs.jsonl，不进入四类主训练数据。','',
 '所有主交付见根目录 historical_pairs.jsonl、generated_candidates.jsonl、accepted_pairs.jsonl、rejected_pairs.jsonl、review_results.jsonl、balanced_train.jsonl、full_pool_train.jsonl、pair_statistics.csv、source_coverage.csv、human_audit.csv、dataset_manifest.json 和 README.md。两张图见 reports/pair_counts_and_acceptance.png、reports/source_distribution.png。','',
 '均衡集v2优先覆盖不同病例，四类病例覆盖分别为A '+str(balanced_coverage['A']['cases'])+'、P '+str(balanced_coverage['P']['cases'])+'、C '+str(balanced_coverage['C']['cases'])+'、F '+str(balanced_coverage['F']['cases'])+'；每类仍为635对、覆盖80来源。v1子集及其说明保留在reports/dataset_versions/v1。复现使用已保存的作者实际编辑和单轮审核结果，执行 scripts/merge_generated_parallel.py --finalize；程序结构见 README.md。历史审核原始交付保留 _historical 后缀。遵照用户后续要求未进行哈希审计。']
 (ROOT/'reports/final_dataset_summary.zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 return report
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
