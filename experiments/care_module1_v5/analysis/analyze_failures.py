"""Post-run CPU-only output diagnostics; never changes predictions or metrics."""
from __future__ import annotations
import collections,hashlib,json,pathlib,re,sys
ROOT=pathlib.Path('/home/gcx/TRIAGE-v5-pr4')
sys.path.insert(0,str(ROOT))
from src.module1.care_v5 import TrialMatchAIAdapter
EXP=ROOT/'experiments/care_module1_v5'
RUN=EXP/'runs/v5_20261018'
OUT=EXP/'analysis'
OUT.mkdir(exist_ok=True)
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
raw=read(RUN/'private/trialmatch_precomputed.jsonl')
private=read(RUN/'private/predictions_private.jsonl')
public=read(RUN/'predictions_public.jsonl')
labels={row['sample_id']:row for row in read(EXP/'data/labels_public.jsonl')}
source_hashes={name:sha(RUN/name) for name in ('run_status.json','metrics.json','errors_public.jsonl','predictions_public.jsonl','private/trialmatch_precomputed.jsonl','private/predictions_private.jsonl')}
parse=collections.Counter(); accepted=collections.Counter(); shapes=collections.Counter(); raw_single_labels=collections.defaultdict(collections.Counter)
invalid_wrapper=collections.Counter(); diagnostic_categories=collections.Counter(); all_condition_diagnostics=[]; schema_by_key={}
for row in raw:
    output=row['trialmatch'];error=output.get('parse_error') or 'accepted';parse[error]+=1
    if error=='accepted':accepted[output.get('raw_label')]+=1
    clean=TrialMatchAIAdapter._strip_thinking(output['raw_output'])
    fence=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',clean,re.I)
    if fence:clean=fence.group(1).strip()
    objects=[]
    for match in re.finditer(r'\{',clean):
        try:
            obj,end=json.JSONDecoder().raw_decode(clean[match.start():])
            if isinstance(obj,dict) and 'Inclusion_Criteria_Evaluation' in obj:objects.append(obj)
        except Exception:pass
    schema=objects[-1] if objects else None
    target=[];other=[];diagnostic_label=None;target_exact=None;placeholder=False
    if schema:
        target=schema.get('Exclusion_Criteria_Evaluation' if row['criterion_kind']=='exclusion' else 'Inclusion_Criteria_Evaluation',[])
        other=schema.get('Inclusion_Criteria_Evaluation' if row['criterion_kind']=='exclusion' else 'Exclusion_Criteria_Evaluation',[])
        shapes[(error,len(target),len(other))]+=1
        if len(target)==1 and isinstance(target[0],dict):
            diagnostic_label=target[0].get('Classification');raw_single_labels[error][str(diagnostic_label)]+=1
        if error=='expected_exactly_one_target_criterion':
            if len(target)!=1:category='target_array_not_single'
            elif other and all(isinstance(item,dict) and item.get('Classification')=='Irrelevant' and
                    re.fullmatch(r'no (?:specific )?(?:exclusion|inclusion) criteria(?: (?:were )?(?:provided|specified|listed|mentioned))?\.?',str(item.get('Criterion','')).strip(),re.I)
                    for item in other):category='non_target_no_criteria_irrelevant_placeholder'
            else:category='other_non_target_entries_or_multiple_items'
            diagnostic_categories[category]+=1
        if error=='invalid_json_response':
            before=clean[:clean.find('{')].strip()
            invalid_wrapper['json_fence_plus_trailing_text' if before=='```json' else 'other_wrapper']+=1
    if output.get('truncated'):diagnostic_categories['2048_tokens_without_eos']+=1
    item={key:row[key] for key in ('sample_id','rule_id','condition_id','criterion_kind')}
    item.update(parse_error=output.get('parse_error'),truncated=output.get('truncated',False),raw_label=output.get('raw_label'),
        raw_state=output['state'],output_tokens=output['output_tokens'],raw_output_sha256=hashlib.sha256(output['raw_output'].encode()).hexdigest(),
        diagnostic_schema_object_found=schema is not None,diagnostic_target_count=len(target) if schema else None,
        diagnostic_other_count=len(other) if schema else None,diagnostic_single_target_label=diagnostic_label)
    all_condition_diagnostics.append(item);schema_by_key[(row['sample_id'],row['condition_id'])]=item

by_sample={row['sample_id']:row for row in private}
known=[];sample_stats=collections.Counter();resolutions=collections.Counter();accepted_resolution=collections.Counter()
for row in private:
    pred=row['module1_output']['rules'][0];conditions=pred['conditions']+pred['exceptions'];errors=sum(bool(c['trialmatch'].get('parse_error')) for c in conditions)
    sample_stats['samples_with_any_parse_error' if errors else 'samples_all_conditions_parse_accepted']+=1
    if pred['trialmatch_model_only_applicability']!='U':
        known.append({'sample_id':row['sample_id'],'rule_id':row['rule_id'],'gold':labels[row['sample_id']]['label'],
            'trialmatch_model_only':pred['trialmatch_model_only_applicability'],'trialmatch_numeric':pred['trialmatch_applicability'],
            'combined':pred['applicability'],'condition_count':len(conditions),'conditions_with_parse_error':errors})
    for condition in conditions:
        resolutions[condition['resolution']]+=1
        if not condition['trialmatch'].get('parse_error'):accepted_resolution[condition['resolution']]+=1

selected=['C01_multiple_values','C05_negation','C08_original','C09_original','C11_counterfactual','C20_original','C22_original','C27_multiple_values']
explanations={
'C01_multiple_values':'两个条件的模型数组均失败；首条件MedNLI反驳，第二条件出现数值跨阈值/多值未知，规则最终U。',
'C05_negation':'围栏、截断、数组失败并存；三个分期条件均未得到程序可用的数值观察，最终U。',
'C08_original':'术前分期数值观察缺失；未接受的用药条件由MedNLI判F，AND逻辑使最终错误拒绝F。',
'C09_original':'三条模型输出均因JSON围栏失效；MedNLI分别支持，但数值观察程序均未抽取到可用数值，最终U。',
'C11_counterfactual':'两个目标条件均成功输出Met；数值观察缺失且证据关系未知，程序保守输出U；纯模型原本正确。',
'C20_original':'单条件生成达到2048无EOS而失败；MedNLI给contradiction并解未知为F，造成错误拒绝。',
'C22_original':'两个Met条件中一条与证据冲突；另一未解析条件由MedNLI判F，最终AND错误拒绝F。',
'C27_multiple_values':'五条件有四条解析失败；仅有的Met也遇支持/反驳冲突，OR结构最终U。'}
cases=[]
for sid in selected:
    row=by_sample[sid];pred=row['module1_output']['rules'][0];gold=labels[sid]['label']
    if pred['applicability']==gold:raise ValueError('selected case is not a final failure')
    conditions=[]
    for condition in pred['conditions']+pred['exceptions']:
        diagnostic=schema_by_key[(sid,condition['condition_id'])]
        conditions.append({'condition_id':condition['condition_id'],'condition_text':condition['condition'],
            'accepted_raw_label':condition['trialmatch'].get('raw_label'),'raw_state':condition['trialmatch']['state'],
            'parse_error':condition['trialmatch'].get('parse_error'),'truncated':condition['trialmatch'].get('truncated',False),
            'diagnostic_schema_single_target_label':diagnostic['diagnostic_single_target_label'],
            'diagnostic_target_count':diagnostic['diagnostic_target_count'],'diagnostic_other_count':diagnostic['diagnostic_other_count'],
            'raw_output_sha256':diagnostic['raw_output_sha256'], 'evidence_state':condition['evidence_state'],
            'final_condition_state':condition['status'],'resolution':condition['resolution'],'conflict':condition['conflict']})
    cases.append({'sample_id':sid,'rule_id':row['rule_id'],'gold':gold,'trialmatch_model_only':pred['trialmatch_model_only_applicability'],
        'trialmatch_numeric':pred['trialmatch_applicability'],'combined':pred['applicability'],'conditions':conditions,'interpretation':explanations[sid]})
summary={'analysis_scope':'post_run_diagnostics_only_no_rescoring_no_inference_changes','run_status':json.loads((RUN/'run_status.json').read_text())['status'],
 'source_sha256':source_hashes,'condition_count':len(raw),'parse_status_counts':dict(parse),'accepted_label_counts':dict(accepted),
 'strict_interface_success_rate':parse['accepted']/len(raw),'truncated_conditions':sum(bool(row['trialmatch'].get('truncated')) for row in raw),
 'diagnostic_schema_shapes':[{'formal_parse_status':key[0],'target_count':key[1],'other_count':key[2],'count':count} for key,count in sorted(shapes.items())],
 'diagnostic_categories':dict(diagnostic_categories),'invalid_json_wrapper_counts':dict(invalid_wrapper),
 'diagnostic_single_target_labels_by_formal_parse_status':{key:dict(value) for key,value in raw_single_labels.items()},
 'sample_parse_status_counts':dict(sample_stats),'pure_model_known_sample_count':len(known),'pure_model_known_with_any_parse_error':sum(row['conditions_with_parse_error']>0 for row in known),
 'pure_model_known_correct_count':sum(row['gold']==row['trialmatch_model_only'] for row in known),
 'pure_model_known_samples':known,'condition_resolution_counts':dict(resolutions),'accepted_conditions_resolution_counts':dict(accepted_resolution),
 'failure_examples':cases,'conditions':all_condition_diagnostics,'privacy':'No patient notes, justifications, or raw reasoning. Condition text is the public structured guideline atom.'}
(OUT/'diagnostics_public.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
# Always replace report examples/counts from actual frozen records.
report=OUT/'ERROR_ANALYSIS.zh.md'
text=report.read_text()
start=text.index('## 失败样本摘要')
end=text.index('## 与整体结果的关系',start)
lines=['## 失败样本摘要','','以下8条均为最终判定失败；只公开规则条件、标签、状态和简短原因。诊断性标签不进入主指标。','',
 '|样本|条件|gold|纯TrialMatchAI|加数值|最终|主要原因|','|---|---|---:|---:|---:|---:|---|']
for case in cases:
 lines.append('|'+ '|'.join([case['sample_id'],','.join(c['condition_id'] for c in case['conditions']),case['gold'],case['trialmatch_model_only'],case['trialmatch_numeric'],case['combined'],case['interpretation']])+'|')
lines+=['','逐条件证据（accepted raw label为正式parser接受标签；诊断标签只从完整目标JSON读取）：','']
for case in cases:
 lines.append('- '+case['sample_id']+'：'+ '; '.join(c['condition_id']+' ['+c['condition_text']+'] accepted='+str(c['accepted_raw_label'])+', diagnostic='+str(c['diagnostic_schema_single_target_label'])+', raw='+c['raw_state']+', final='+c['final_condition_state']+', reason='+str(c['parse_error'] or c['resolution']) for c in case['conditions']))
text=text[:start]+'\n'.join(lines)+'\n\n'+text[end:]
# Correct counts directly from loaded data; preserve primary metrics unchanged.
text=text.replace('13 条中 12 条至少有一个条件级解析失败',f'13 条中 {summary["pure_model_known_with_any_parse_error"]} 条至少有一个条件级解析失败')
report.write_text(text)
print(json.dumps({key:summary[key] for key in ('condition_count','parse_status_counts','diagnostic_categories','sample_parse_status_counts','pure_model_known_sample_count','pure_model_known_with_any_parse_error','pure_model_known_correct_count')},ensure_ascii=False))
