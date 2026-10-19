#!/usr/bin/env python3
"""Integrate the clinical follow-up without relabeling synthetic O2/O3 evidence."""
from __future__ import annotations
import datetime, hashlib, json, pathlib
import pandas as pd
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[1]
R=ROOT/'results'; C=R/'obs1_clinical'; D=ROOT/'data/clinical'
def dump(p,x): pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def table(d):
    lines=['|方法|适用准确率（95% CI）|不适用准确率（95% CI）|误用率（95% CI）|']
    lines+=['|---|---|---|---|']
    for method in ['baseline','seka','pasta']:
        values=[]
        for metric in ['A_app','A_non','M_non']:
            r=d[(d.method==method)&(d.metric==metric)].iloc[0]
            values.append(f'{r.estimate:.1%} [{r.ci_low:.1%}, {r.ci_high:.1%}]')
        lines.append('|'+method+'|'+'|'.join(values)+'|')
    return '\n'.join(lines)
def main():
    s=json.loads((C/'summary.json').read_text()); cm=json.loads((C/'manifest.json').read_text())
    metrics=pd.read_csv(C/'metrics.csv'); contrasts=pd.read_csv(C/'contrasts.csv')
    primary=metrics[metrics.analysis_subset=='eligible_cases']; count=s['screening']
    source_cards=json.loads((D/'rule_cards.json').read_text())
    # A readable per-case audit table joins the source gold, screening and all
    # three actual held-out predictions; no example is chosen by method benefit.
    candidates=[json.loads(x) for x in (D/'candidates.jsonl').read_text().splitlines()]
    screening=pd.read_csv(C/'prior_screen.csv').set_index('case_id')
    predictions=pd.read_csv(C/'predictions.csv'); predictions=predictions[predictions.role=='test']
    pivot=predictions.pivot(index='case_id',columns='method',values='prediction')
    review=[]
    for row in candidates:
        pr=screening.loc[row['case_id']]; item={k:row[k] for k in ['case_id','pair_id','group_id','rule_id','value','subset','gold_label','source_url','source_section','patient','guidance']}
        item.update(eligible=bool(pr.eligible),complete_pair_eligible=bool(pr.complete_pair_eligible),
            closedbook_label=row['option_map'][pr.prior_prediction],paraphrase_label=row['option_map'][pr.paraphrase_prediction])
        for method in ['baseline','seka','pasta']: item[method+'_label']=row['option_map'][pivot.loc[row['case_id'],method]]
        review.append(item)
    pd.DataFrame(review).to_csv(D/'case_review.csv',index=False)
    names={'glycemia':'血糖分类','adiposity':'成人BMI','renal':'CKD GFR分类','blood_pressure':'血压目标','venous_thromboembolism':'Wells概率分类',
      'pneumonia':'肺炎风险评分','sore_throat':'咽痛抗菌药推荐措辞','atrial_fibrillation':'房颤抗凝推荐措辞','bone_density':'DXA T-score范围',
      'anaemia':'贫血程度','copd':'COPD气流受限程度','hypoglycemia':'低血糖分级'}
    source_lines=['# 临床规则来源与核验说明\n',
      '真实临床规则的忠实转述，模拟患者，无真实患者数据。核验由助手基于公开一手参考资料完成，未声称经过独立医生复核。',
      '临床阈值、人群、单位、边界和必要例外已核对；模型概率不参与金标产生。不以高置信度作为资格条件。\n']
    for fam in sorted(names):
        cards=[r for r in source_cards if r['family']==fam]
        c=cards[0]
        source_lines.append(f"- **{names[fam]}**：[公开一手来源]({c['source']})；依据：{'；'.join(dict.fromkeys(x['section'] for x in cards))}。")
    source_lines.extend(['\n补充交叉核验：',
      '- GFR分层另与 [KDIGO 2024 CKD guideline](https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf) 的GFR分类核对。',
      '- WHO贫血分层另与 [WHO 2024 原始指南](https://www.who.int/publications/i/item/9789240088542) Table 3核对。',
      '- ADA低血糖定义核对 [ADA 2026原始表6.4](https://diabetesjournals.org/care/article-pdf/49/Supplement_1/S132/848848/dc26s006.pdf)。',
      '\n边界标点修复只改变高亮span元数据；修复前后患者、指南正文、选项和闭卷输入均相同。初次筛选与错误日志保留在results/obs1_clinical/attempt1_span_boundary。'])
    (D/'source_review.md').write_text('\n'.join(source_lines))
    if (C/'attempt1_span_boundary/candidates.jsonl').exists():
        before=[json.loads(x) for x in (C/'attempt1_span_boundary/candidates.jsonl').read_text().splitlines()]
        after=[json.loads(x) for x in (D/'candidates.jsonl').read_text().splitlines()]
        assert len(before)==len(after)
        fields=['prompt','closedbook_prompt','paraphrase_prompt','gold_answer','option_map','patient','guidance']
        assert all(all(a[k]==b[k] for k in fields) for a,b in zip(before,after))
        a=pd.read_csv(C/'attempt1_span_boundary/prior_screen.csv'); b=pd.read_csv(C/'prior_screen.csv')
        assert a.set_index('case_id').eligible.equals(b.set_index('case_id').eligible)
    figure_check=[]
    for stem in ['O1_clinical_faithfulness_misuse','O1_clinical_screening','O1_clinical_subset_sensitivity']:
        for ext in ['png','pdf','svg']:
            p=ROOT/'figures'/f'{stem}.{ext}'; assert p.is_file() and p.stat().st_size>1000
            if ext=='png': assert min(Image.open(p).info['dpi'])>=299
            figure_check.append(str(p.relative_to(ROOT)))
    dump(C/'deliverables.json',{'figure_files':figure_check,'source_review':'data/clinical/source_review.md',
        'boundary_fix_changed_model_input':False,'boundary_fix_changed_eligibility':False})
    initial_report=R/'observations_summary_initial_synthetic.md'
    if not initial_report.exists(): initial_report.write_text((R/'observations_summary.md').read_text())
    old=json.loads((R/'analysis_summary.json').read_text())
    h=old['obs2']['selected_head']; z=old['obs3']['primary']
    delta=[]
    cd=contrasts[contrasts.analysis_subset=='eligible_cases']
    for method in ['seka','pasta']:
        a=cd[(cd.method==method)&(cd.metric=='A_app')].iloc[0]
        m=cd[(cd.method==method)&(cd.metric=='M_non')].iloc[0]
        delta.append(f"- {method.upper()}：Δ适用准确率 {a.estimate:+.1%}，95% CI [{a.ci_low:+.1%}, {a.ci_high:+.1%}]；Δ误用率 {m.estimate:+.1%}，95% CI [{m.ci_low:+.1%}, {m.ci_high:+.1%}]。")
    final_report=f"""# CARE 三项观察：2026-10-16更新报告

**本次已补齐真实临床规则的观察一。** 上一版“临床合格样本0、不可检验”已被本次真实模型实测取代；初始虚构规则结果另存observations_summary_initial_synthetic.md。

## 观察一：真实临床规则先验冲突

12个临床规则家族、24条规则，192对、384例模拟病例。规则来自公开一手临床资料，未经虚假改写制造冲突；患者并非真实患者记录。
采用宽松筛选：来源核验、两个闭卷表述选择稳定、与规则金标冲突；不设高置信度门槛、不按干预收益挑样本。
**合格 {count['eligible']} 例：适用 {count['eligible_app']}、不适用 {count['eligible_non']}；两侧都合格 {count['complete_eligible_pairs']} 对（{2*count['complete_eligible_pairs']} 例）。**
主要检验状态：{s['subsets']['eligible_cases'].get('tradeoff')}。

{table(primary)}

配对方法变化：

{chr(10).join(delta)}

区间为10000次临床规则家族bootstrap 95% CI；同一规则的患者变体不能当作独立规则。合格病例主分析覆盖{s['subsets']['eligible_cases']['n_families']}个家族。
另报告完整配对及384例未筛选全体病例；详见obs1_clinical/clinical_report.md、metrics.csv、contrasts.csv。
固定Qwen3-4B-Base，三折按家族划分；每折训练6家族、开发2家族、测试4家族。官方SEKA投影与PASTA干预均实际执行，选项、输入、高亮span及层头保持一致。
本次模型阶段 {cm['seconds']/60:.2f} 分钟，GPU {cm['physical_gpu']}，BF16峰值显存 {cm['peak_memory_allocated_GiB']:.2f} GiB。模型revision与官方源码提交均见obs1_clinical/manifest.json。

## 观察二：原虚构规则机制实验，未改称临床结果

60组、360条件输入，36层×8 KV头；训练/开发/测试按组为36/12/12。
开发集选定层{int(h['layer'])}、KV头{int(h['kv_head'])}（从0计）。适用性测试AUROC {h['applicability_test_auc']:.3f}，
95% CI [{h['applicability_auc_ci'][0]:.3f}, {h['applicability_auc_ci'][1]:.3f}]。
这说明在这些受控输入下存在可解码信息，不能声称Key完全不编码适用性；措辞、长度及数字变化仍构成混杂。数据及图保留在obs2和原O2图。

## 观察三：原虚构规则固定query机制实验

主分析有效记录{z['n_valid']:,}条，来自12个独立规则组。实际BF16局部logit反向变化比例 {z['actual_bf16_opposite_rate']:.2%}，
95% CI [{z['actual_bf16_ci_low']:.2%}, {z['actual_bf16_ci_high']:.2%}]。
固定原始query，核对QK-norm、RoPE和4:1 GQA映射；公式恒等式在float32中检查，实际BF16舍入单独保存。
这不是完整多层临床决策的因果效应。详见obs3、原O3图和初始报告。

## 局限

- 临床观察一使用真实规则及模拟患者，无真实患者队列、无独立医生复核。来源核验、必要例外和边界检查不等于临床有效性验证。
- 稳定闭卷错误是操作性的“先验冲突”；受控选项偏好也可能参与，不能直接证明记忆中的错误医学知识。
- 仅12个规则家族，部分评分值和措辞重复；CI按家族计算，但不包含重新拟合投影的不确定性。
- 规则经忠实压缩，许多问题直接给出计算好的评分；不是完整病历推理或完整指南覆盖。
- 只有一个Base模型、固定层头及强度网格，不能把阴性结果外推为所有临床任务都无权衡。
- O2/O3仍是虚构规则机制实验，不能因本次O1补齐就将三项全部称为真实临床验证。

## 文件与复现

- 临床来源与数据：data/clinical/source_review.md、rule_cards.json、candidates.jsonl、protocol.json。
- 临床原始闭卷及三方法输出：results/obs1_clinical/。
- 临床论文图：figures/O1_clinical_faithfulness_misuse、O1_clinical_screening、O1_clinical_subset_sensitivity，均为300dpi PNG、PDF、SVG。
- 临床详细报告：results/obs1_clinical/clinical_report.md。
- 初始O2/O3完整报告：results/observations_summary_initial_synthetic.md。
- 复现入口：REPRODUCE_CLINICAL.sh；所有模型执行均在服务器指定环境进行。
"""
    (R/'observations_summary.md').write_text(final_report)
    root_manifest=R/'manifest.json'
    initial_manifest=R/'manifest_initial_synthetic.json'
    if not initial_manifest.exists(): initial_manifest.write_text(root_manifest.read_text())
    rm=json.loads(initial_manifest.read_text())
    rm['initial_pilot_synthetic_only']=True
    rm['initial_pilot_clinical_conflict_eligible']=0
    rm['clinical_conflict_eligible']=count['eligible']
    rm['synthetic_only']=False
    rm['patient_data_are_simulated']=True
    rm['legacy_fields_scope']='groups/cases/seconds and original artifact hashes describe initial pilot; clinical follow-up is separate below'
    rm['clinical_followup']={'manifest':'results/obs1_clinical/manifest.json','summary':'results/obs1_clinical/summary.json',
        'candidate_cases':384,'candidate_pairs':192,'rule_families':12,'eligible_cases':count['eligible'],
        'eligible_app':count['eligible_app'],'eligible_non':count['eligible_non'],'both_eligible_pairs':count['complete_eligible_pairs'],
        'source_grounded_rules':True,'patient_vignettes_simulated':True,'seconds':cm['seconds'],
        'peak_memory_allocated_GiB':cm['peak_memory_allocated_GiB']}
    rm['clinical_artifact_hashes']={}
    for folder in [D,C,ROOT/'scripts',ROOT/'tests']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and 'logs' not in p.parts and p.suffix!='.log' and p.name!='manifest.json':
                rm['clinical_artifact_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/'figures').glob('O1_clinical*'):
        rm['clinical_artifact_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    rm['updated_at_beijing']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    dump(root_manifest,rm)
    dump(R/'analysis_summary_updated.json',{'clinical_obs1':s,'initial_obs2':old['obs2'],'initial_obs3':old['obs3']})
    dump(R/'blockers.json',{'clinical_O1':'resolved: source-grounded, screened, real three-method model evaluation completed',
        'eligible_cases':count['eligible'],'remaining_blockers':[],'limitations':'See clinical report; no independent clinician review or real patient cohort.'})
    (R/'STATUS.md').write_text(f"Clinical O1 follow-up completed: {count['eligible']} eligible cases, {count['complete_eligible_pairs']} complete conflict pairs. O2/O3 retain their original synthetic-only scope.\n")
    readme=ROOT/'README.md'
    body=readme.read_text(); marker='## 临床观察一更新'
    if marker in body: body=body[:body.index(marker)]
    # Replace stale opening scope sentence, keeping provenance of initial files.
    body=body.replace('**科学范围：synthetic-only。没有经过临床来源与规则配对核验的先验冲突样本，临床版观察一不可检验。**',
        '**科学范围：临床O1为真实规则与模拟患者；O2/O3仍为虚构规则机制实验。临床O1已补齐，见下方更新。**')
    body=body.replace('和7组论文图（每组PNG、PDF、SVG）','和10组论文图（每组PNG、PDF、SVG，含3组临床补充图）')
    body+=f"\n{marker}\n\n合格{count['eligible']}例（适用{count['eligible_app']}，不适用{count['eligible_non']}），完整双侧冲突配对{count['complete_eligible_pairs']}对。详见results/observations_summary.md和results/obs1_clinical/clinical_report.md。新增3组PNG/PDF/SVG图。原始先导结果保留，不能将O2/O3改称临床验证。复现：REPRODUCE_CLINICAL.sh。\n"
    readme.write_text(body)
    caps=ROOT/'figures/CAPTIONS.md'; text=caps.read_text(); marker='\n# Clinical O1 follow-up'
    if marker in text: text=text[:text.index(marker)]
    text=text.replace('All figures show real Qwen3-4B-Base measurements on fictional conditional rules.',
        'Original figures 1–7 show real Qwen3-4B-Base measurements on fictional conditional rules; the additional clinical figures are described below.')
    text+=f"""{marker}

These three additional figures use authentic clinical reference rules with simulated patients, not altered counterfactual medical guidelines. Screening requires source verification, stable choice under two closed-book formulations and disagreement with rule gold. No confidence threshold is imposed. All three methods are evaluated out of their clinical-rule family's training/development fold.

- **O1_clinical_faithfulness_misuse:** {count['eligible']} eligible cases ({count['eligible_app']} applicable, {count['eligible_non']} nonapplicable) from {s['subsets']['eligible_cases']['n_families']} rule families. Left: applicable/nonapplicable accuracy with family-cluster 95% bootstrap intervals. Right: paired changes in accuracy and selection of the highlighted, inapplicable action. Intervals use 10,000 family resamples and fixed fitted projections.
- **O1_clinical_screening:** Screening funnel and candidate/eligible counts for every clinical family, including families with zero eligible cases. Only screening counts are plotted; these are not independent efficacy estimates.
- **O1_clinical_subset_sensitivity:** Harmonic accuracy with family-bootstrap intervals for all 384 candidate cases, eligible cases, and both-eligible complete pairs ({count['complete_eligible_pairs']} pairs, {2*count['complete_eligible_pairs']} cases). Denominators and family counts are shown separately; these subsets are not pooled.

Source verification does not imply independent clinician adjudication. Stable closed-book error is an operational conflict definition and may include answer-option preference. Confidence intervals do not capture model-refitting uncertainty or justify clinical safety claims.
"""
    caps.write_text(text)
    # Keep archived initial hashes in manifest_initial_synthetic.json; the
    # current manifest reflects current files, including the updated captions.
    for name in list(rm.get('artifact_hashes',{})):
        p=ROOT/name
        if p.is_file(): rm['artifact_hashes'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(root_manifest,rm)
    print(f"Integrated clinical results: eligible={count['eligible']}, both-eligible pairs={count['complete_eligible_pairs']}",flush=True)
if __name__=='__main__': main()

