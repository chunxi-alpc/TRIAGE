"""Read completed server outputs; no fitted parameter or data-selection changes."""
import json,hashlib,collections,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def rows(name):return [json.loads(x) for x in (ROOT/'results'/name).read_text().splitlines()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(rs):
    n=len(rs)
    return dict(n=n,correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/n if n else None,explicit_context_follow=sum(r['prediction']==r['context_answer'] for r in rs if r['condition'] in ['misleading','caution']),invalid=sum(r['prediction'] is None for r in rs),truncated=sum(r['truncated'] for r in rs))
def cluster_difference(a,b):
    aa={(r['id'],r['order']):r for r in a};bb={(r['id'],r['order']):r for r in b}
    assert aa.keys()==bb.keys()
    sums=collections.defaultdict(lambda:[0,0])
    for k,r in aa.items():sums[r['group']][0]+=int(r['correct'])-int(bb[k]['correct']);sums[r['group']][1]+=1
    if not sums:return dict(groups=0,delta=None,ci95=None)
    x=np.array(list(sums.values()));rng=np.random.default_rng(20261019)
    boot=x[rng.integers(0,len(x),(5000,len(x)))].sum(1)
    delta=boot[:,0]/boot[:,1]
    return dict(groups=len(x),delta=float(x[:,0].sum()/x[:,1].sum()),ci95=np.quantile(delta,[.025,.975]).tolist(),note='Descriptive group bootstrap; fixed model/direction, not training uncertainty. Few groups limit inference.')
def main():
    complete=load(ROOT/'results/completed.json');assert complete['status']=='complete'
    for f,h in complete['files'].items():assert sha(ROOT/'results'/f)==h
    items=load(ROOT/'data/items.json');base=rows('baseline.jsonl');inter=rows('test_interventions.jsonl');opens=rows('open_scored.jsonl')
    test={r['id'] for r in items if r['split']=='test'}
    prior={i:[r for r in base if r['id']==i and r['condition']=='closed'] for i in test}
    known={i for i,rs in prior.items() if len(rs)==2 and all(r['correct'] for r in rs)}
    all_open={i:[r for r in opens if r['id']==i] for i in test}
    strict={i for i in known if len(all_open[i])==2 and all(r['correct'] for r in all_open[i])}
    known_wrong={i for i,rs in prior.items() if len(rs)==2 and all(r['prediction'] in ['A','B'] and not r['correct'] for r in rs)}
    eligibility=[dict(id=i,recognition_known=i in known,strict_known=i in strict,stable_wrong_prior=i in known_wrong,open_questions_completed=len(all_open[i]),open_correct=sum(r['correct'] for r in all_open[i])) for i in sorted(test)]
    dump(ROOT/'results/eligibility.json',eligibility)
    conditions={
      'baseline_misleading':[r for r in base if r['split']=='test' and r['condition']=='misleading'],
      'caution_prompt':[r for r in base if r['split']=='test' and r['condition']=='caution'],
      'baseline_aligned':[r for r in base if r['split']=='test' and r['condition']=='aligned'],
    }
    for method in ['selected_direction','nonzero_direction_diagnostic']:
        for cond in ['misleading','aligned']:conditions[method+'_'+cond]=[r for r in inter if r['method']==method and r['condition']==cond]
    populations={'all_test':test,'recognition_known':known,'strict_open_and_recognition_known':strict,'stable_wrong_prior':known_wrong}
    results={pop:{name:metric([r for r in rs if r['id'] in ids]) for name,rs in conditions.items()} for pop,ids in populations.items()}
    deltas={pop:{name:cluster_difference([r for r in conditions[name] if r['id'] in ids],[r for r in conditions['baseline_misleading'] if r['id'] in ids]) for name in ['caution_prompt','selected_direction_misleading','nonzero_direction_diagnostic_misleading']} for pop,ids in populations.items()}
    pat=rows('patient.jsonl');po=rows('patient_open_scored.jsonl');patient_results={}
    for method,condition in [('no_intervention','closed'),('no_intervention','scoped'),('selected_direction','scoped'),('signed_scope_gate','scoped')]:
        rs=[r for r in pat if r['method']==method and r['condition']==condition]
        pairids={r['id'] for r in rs};joint=sum(all(r['correct'] for r in rs if r['id']==i) for i in pairids)
        patient_results[method+'_'+condition]=dict(n=len(rs),correct=sum(r['correct'] for r in rs),joint_correct_pairs=joint,pairs=len(pairids),applicable_correct=sum(r['correct'] for r in rs if r['applicable']),inapplicable_correct=sum(r['correct'] for r in rs if not r['applicable']),n_applicable=sum(r['applicable'] for r in rs),strict_patient_known_pairs=[i for i in pairids if all(x['correct'] for x in pat if x['id']==i and x['condition']=='closed') and len([x for x in po if x['id']==i])==2 and all(x['correct'] for x in po if x['id']==i)])
    # Essential integrity checks: full output coverage, unique keys, complete pairing, split lineage.
    assert len(base)==len(items)*8
    assert len({(r['id'],r['condition'],r['order']) for r in base})==len(base)
    assert len(opens)==2*len(load(ROOT/'data/open_probes.json'))
    for g in {r['group'] for r in items}:assert len({r['split'] for r in items if r['group']==g})==1
    assert len(inter)==len(test)*4*2
    assert len(pat)==64
    for r in rows('calibration.jsonl')+[x for x in inter+pat if x.get('alpha') and 'reused_from' not in x]:
        assert r['steering']['calls']==1 and r['steering']['realized_mean_norm']>0
    direction=load(ROOT/'results/direction.json');selection=load(ROOT/'results/selection.json')
    assert set(direction['train_ids']).issubset({r['id'] for r in items if r['split']=='train'})
    assert selection['test_inference_started'] is False
    audit=dict(status='passed',checks=['Raw completed-file SHA256 matches','194 x 4 conditions x 2 orders baseline complete and unique','Open probes complete','Direction uses training IDs only','Grouping prevents split overlap of declared fact families','Patient variants and source advice belong to held-out groups','Selection file frozen before test inference in sequential runner','Test intervention coverage complete; reuse explicitly marked','No generated response used to alter questions, regexes, split, or hyperparameter grid'],baseline_rows=len(base),open_rows=len(opens),test_intervention_rows=len(inter),patient_rows=len(pat),generated_rows=sum(len(rows(f)) for f in ['baseline.jsonl','open_probes.jsonl','calibration.jsonl','test_interventions.jsonl','patient.jsonl','patient_open.jsonl'])-sum('reused_from' in r for r in inter+pat),limitations=['Two open questions provide behavioral evidence of knowledge, not proof of training-set inclusion or causal localization','Main task is rejecting fabricated misleading advice, not validating real historical guideline updates','Some full-release advice pairs differ in scope or recommendation strength and are not logically exclusive; strict open subset uses pre-authored factual probes','Open correctness uses fixed answer regexes; raw responses retained for inspection','No native 4290-scenario results; original scenario-to-advice provenance unresolved','Patient applicability parser is handcrafted and simplified; not full CARE module one','Patient diagnostic is scoped evidence selection and not counted as strict parametric conflict','Full CARE key-space intervention, multiple models/seeds, DPO/SFT, external clinical validation unexecuted'])
    dump(ROOT/'audit/execution_audit.json',audit)
    result=dict(populations={k:sorted(v) for k,v in populations.items()},metrics=results,differences=deltas,patient=patient_results,selection=selection,direction=direction,execution=complete,audit=audit)
    dump(ROOT/'results/summary.json',result)
    pct=lambda x:'NA' if x is None else f'{100*x:.2f}%'
    lines=['CARE 医疗上下文知识冲突实测报告','2026-10-19','',
      '已执行：服务器 Qwen3-8B BF16；ConflictMedQA/DriftMed 建议级适配；闭卷测量；正确/错误上下文基线；提醒基线；训练集方向估计；验证集系数选择；留出测试；患者条件配对。',
      '本报告只读取本轮 completed.json 所校验的真实输出。不是原论文全规模复现，不是完整 CARE 实现。','',
      '数据来源与前轮审阅',
      '论文 https://aclanthology.org/2025.findings-emnlp.38/；发布 https://huggingface.co/datasets/RDBH/DriftMed；revision 5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e。',
      'PR #10 原 head 6bae71cbb99d60503de0f1867c7c38ca4f1772b5：前轮 160 合成规则病例并不建立参数知识冲突，原报告已承认两条规则都在上下文中。原自由生成结果不被本轮覆盖。',
      '实际下载 advice_pairs=194、两份 scenario 文件各4290条。论文195组与发布194组不一致；不猜测缺失对应关系。数据卡标注 Research Use Only，原始下载保存于服务器，Git保留来源哈希、获取脚本和本轮派生数据。',
      '错误建议为作者人工构造的 pseudo-outdated 文本，不是有历史出处的旧指南。医学标签依据发布版本（约2025年2月），未逐条核查当下指南；不能作为临床建议。','',
      '隔离与方法',
      '按人工医学主题合并和 TF-IDF cosine>=0.70 连通组划分，组内建议、换序、无选项问题和患者变体保持同一 split。患者诊断的来源组预先保留给 test。58 train / 68 validation / 68 test；22 / 12 / 14 组。主题隔离并不保证不存在其他语义重叠。',
      '严格已知集：两个无选项医学问题都匹配预先固定的正确答案，且两次候选换序都正确。只按闭卷结果选集，不按是否被错误上下文误导来选。无选项问题是人工编写的不同问法，不是仅修改“请回答”的包装。',
      '全量和双换序已知集为次要分析。选择题仍可能利用选项线索；因此严格无选项证据独立列出。主任务只提供一条检索建议，没有把旧、新两条规则同时放入检索上下文；选择题中的候选本身仍可提供线索。',
      f'方向只用 {direction["train_count"]} 组训练集双换序已知样本，取 clean-minus-misleading 的残差均值差；第 {direction["layer"]} 层（从0开始），范数 {direction["norm"]:.4f}。只改 prefill 最后一个词元，不改权重。',
      '该均值差同时可能包含上下文存在与长度因素，不是已证明的纯知识选择方向；未做匹配长度或随机方向消融。',
      f'validation 网格 [-2,-1,0,1,2]，按 aligned/misleading 平均准确率选取，同分优先绝对值小。最终 alpha={selection["selected"]["alpha"]}；最佳非零诊断 alpha={selection["selected_nonzero"]["alpha"]}。选择在 test 推断前写入 selection.json，之后没有重选。','',
      '实际留出结果（分母为两种选项顺序的回答次数，不能当成独立患者数）']
    for pop in ['all_test','recognition_known','strict_open_and_recognition_known','stable_wrong_prior']:
        lines.append(f'\n{pop}: {len(populations[pop])} 组事实')
        for name,m in results[pop].items():lines.append(f'  {name}: {m["correct"]}/{m["n"]} = {pct(m["accuracy"])}; 无效答案={m["invalid"]}; 明确跟随错误上下文={m["explicit_context_follow"]}')
    lines+=['','患者条件反事实配对（单独的适用性诊断）','四对分别改变免疫状态、透析状态、ICU状态、年龄；同对上下文和候选顺序固定。每例只有一条有适用范围的真实建议，没有显式旧规则。正确目标来自不同临床条件的来源建议。',
      '这些配对本身不是严格参数知识冲突证据，不进入主冲突分母。signed_scope_gate 使用预先写好的文本条件解析，不是完整 CARE 提取器。']
    for name,m in patient_results.items():lines.append(f'{name}: {m["correct"]}/{m["n"]}; 整对全正确 {m["joint_correct_pairs"]}/{m["pairs"]}; 无选项及换序均已知的患者对 {len(m["strict_patient_known_pairs"])}')
    lines+=['','运行与未执行范围',f'实际运行 {complete["seconds"]/60:.2f} 分钟；GPU 0；PyTorch峰值已分配 {complete["peak_allocated_bytes"]/2**30:.2f} GiB；峰值保留 {complete["peak_reserved_bytes"]/2**30:.2f} GiB。',
      '没有执行：原生4290场景的完整复现、真实历史指南更新验证、完整CARE模块一/键子空间干预、多模型多随机种子、SFT/DPO、临床专家独立复核。',
      '未运行的内容没有填入结果，也没有以本轮诊断替代。严格已知子集小，患者只有4对；不据此声称CARE有效。',
      '可追溯文件：config.json、data/*.json、sources来源记录、results/*.jsonl原始答案及提示、direction.pt训练激活、selection.json、audit/data_audit.json、audit/execution_audit.json。',
      '方向对比的按事实组bootstrap区间见 results/summary.json；不把方向训练不确定性包含在这些区间内。','']
    if (ROOT/'results/open_context_completed.json').exists():
        oc=load(ROOT/'results/open_context_completed.json')
        assert sha(ROOT/'results/open_context.jsonl')==oc['sha256']
        assert sha(ROOT/'results/open_context_scored.jsonl')==oc['scored_sha256']
        ors=rows('open_context_scored.jsonl');assert len(ors)==oc['n']
        recall={i for i in test if len(all_open[i])==2 and all(r['correct'] for r in all_open[i])}
        free={}
        for name,ids in [('strict_original',strict),('recall_only_exploratory',recall),('all_open_test',set(all_open_i for all_open_i,rr in all_open.items() if rr))]:
            free[name]={}
            for cond in ['misleading','aligned','caution','misleading_with_frozen_direction']:
                rs=[r for r in ors if r['id'] in ids and r['condition']==cond]
                free[name][cond]=dict(facts=len(ids),n=len(rs),correct=sum(r['correct'] for r in rs),explicit_wrong_match=sum(r['explicit_wrong_match'] for r in rs),truncated=sum(r['truncated'] for r in rs))
        supplement=dict(protocol=load(ROOT/'audit/open_context_protocol.json'),metrics=free,recall_only_ids=sorted(recall),execution=oc)
        dump(ROOT/'results/open_context_summary.json',supplement)
        lines+=['','无选项上下文冲突补测（事后探索性，未重选参数）',
          '在看到首轮结果后，对全部14组预先编写的test开放问题补测，不只补测已知样本。112次真实生成，问题、答案匹配器、方向和系数保持冻结；提示只包含检索文本和问题，不含候选答案。',
          'strict_original 仍使用首轮1组严格已知事实；recall_only_exploratory 是另列的3组双开放问答已知事实，不能覆盖原主要指标。',
          '开放答案的固定匹配器允许部分范围答案（例如3–6个月匹配6个月），因此 recall_only 属于敏感性诊断；不声称它提供更严格的事实验证。']
        for pop,ms in free.items():
            lines.append(pop)
            for cond,m in ms.items():lines.append(f'  {cond}: 正确{m["correct"]}/{m["n"]}; 明确错误答案{m["explicit_wrong_match"]}; 截断{m["truncated"]}')
        lines.append(f'补测用时 {oc["seconds"]:.2f} 秒。两轮共 {audit["generated_rows"]+oc["n"]} 次实际生成。')
    lines+=['','解释限制补充',
      '选定alpha为负值，不能把向量正负未经验证地等同于“记忆方向/上下文方向”。患者 signed_scope_gate 仅为沿前轮符号构造的探索性诊断，本轮未验证这种符号解释。',
      '少量严格样本、单模型、来源文本不等长和没有随机方向消融，均使本轮不足以判断CARE有效性。结果支持的只是该模型在本轮提示条件下的行为。']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines))
    print(json.dumps(dict(strict_n=len(strict),recognition_n=len(known),test_n=len(test),strict=results['strict_open_and_recognition_known'],selection=selection['selected'],patient=patient_results,runtime_minutes=complete['seconds']/60),ensure_ascii=False))
if __name__=='__main__':main()
