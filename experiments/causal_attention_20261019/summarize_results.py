"""Aggregate saved real-model outputs without selecting cases or rerunning inference."""
from pathlib import Path
import csv,json,sys
from collections import defaultdict
from analysis import read_jsonl, score_cases, avg
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'care_reliability_20261019/data/mechanism_cases.jsonl'

def main():
    output=ROOT/'analysis_outputs';output.mkdir(exist_ok=True)
    cases=read_jsonl(DATA)
    datasets={s:[c for c in cases if c['split']==s] for s in ('dev','confirmation')}
    files={}
    for path in sorted((ROOT/'results').glob('**/*.jsonl')):
        rows=read_jsonl(path)
        if rows and all('prediction' in r or r.get('status')=='technical_failure' for r in rows) and rows[0].get('split') in datasets:
            files[str(path.relative_to(ROOT))]=rows
    baselines={}
    preferred={'dev':'results/group1_gpu0/baseline.jsonl','confirmation':'results/group1_confirmation_main/baseline.jsonl'}
    for split in datasets:
        candidates=[(p,r) for p,r in files.items() if r[0]['split']==split and r[0].get('config',{}).get('baseline')]
        if candidates:
            baselines[split]=next((r for p,r in candidates if p==preferred[split]),candidates[0][1])
    summaries={};errors=[];table=[]
    for path,rows in files.items():
        split=rows[0]['split'];cfg=rows[0].get('config',{})
        score,error=score_cases(datasets[split],rows,baselines.get(split,[]))
        diagnostics=[d for r in rows for d in r.get('attention_diagnostics',[])]
        attention={}
        for sign in ('positive','negative'):
            ds=[d[sign] for d in diagnostics if sign in d]
            attention[sign]={'layer_case_records':len(ds),
                'score_delta_mean':avg([d['score_delta_mean'] for d in ds]),
                'score_delta_min':min([d['score_delta_min'] for d in ds],default=None),
                'score_delta_max':max([d['score_delta_max'] for d in ds],default=None),
                'attention_mass_before_mean':avg([d['attention_mass_before'] for d in ds]),
                'attention_mass_after_mean':avg([d['attention_mass_after'] for d in ds])}
        attention.update(mean_absolute_attention_change=avg([d['attention_absolute_change_mean'] for d in diagnostics]),
            changed_cases=sum(any(d['attention_absolute_change_mean']>1e-9 for d in r.get('attention_diagnostics',[])) for r in rows),
            noop_cases=sum(r.get('noop',False) for r in rows),
            mean_target_tokens=avg([r.get('target_token_count') for r in rows]))
        summaries[path]={'split':split,'arm':rows[0].get('arm'),'source':rows[0].get('source'),'config':cfg,**score,'attention':attention}
        errors += [dict(arm_path=path,**e) for e in error]
        table.append({'path':path,'split':split,'source':rows[0].get('source'),'arm':rows[0].get('arm'),
            **{k:score[k] for k in ('cases','actual_outputs','correct','accuracy','pair_correct','pairs','corrected','damaged','net_corrected','prediction_changed','mean_gold_margin','mean_margin_delta_vs_baseline')},
            'primary_pair_correct':score['primary_canonical']['pair_correct'], 'primary_pairs':score['primary_canonical']['pairs'],
            'attention_changed_cases':attention['changed_cases'],'noop_cases':attention['noop_cases']})
    (output/'all_arm_metrics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    with (output/'summary.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (output/'errors_and_changes.jsonl').open('w') as stream:
        for row in errors:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'arms':len(summaries),'model_output_rows':sum(len(v) for v in files.values()),'outputs':str(output)}))
if __name__=='__main__':main()
