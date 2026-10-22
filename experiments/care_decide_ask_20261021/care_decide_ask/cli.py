"""CPU-only benchmark prototype; no model loading and no hidden network use."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .core import decide_or_ask,derive_truth_states
from .a1_adapter import frozen_a1_hard_sets,require_a1_probabilities
from .calibration import calibrate_joint,make_prediction_sets
from .metrics import evaluate


def rows(path):
    for n,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            data=json.loads(line)
            if not isinstance(data,dict):raise ValueError(f'expected object {path}:{n}')
            yield data


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        if isinstance(value,list):
            for item in value:f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
        else:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--data',required=True);c.add_argument('--alpha',type=float,required=True);c.add_argument('--unit',choices=['case','family'],default='case');c.add_argument('--out',required=True)
    e=sub.add_parser('run');e.add_argument('--cases',required=True);e.add_argument('--mode',required=True,choices=['hard','conformal','oracle']);e.add_argument('--calibration');e.add_argument('--out',required=True)
    v=sub.add_parser('evaluate');v.add_argument('--predictions',required=True);v.add_argument('--gold',required=True);v.add_argument('--out',required=True)
    a=ap.parse_args(argv)
    if a.command=='calibrate':
        report=calibrate_joint(rows(a.data),a.alpha,unit=a.unit)
        report['calibration_source_sha256']=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
        save(a.out,report)
        print(json.dumps({'status':'calibration_written','n_cases':report['n_cases'],'out':a.out}))
        return
    if a.command=='evaluate':
        labels={}
        for item in rows(a.gold):
            cid=item.get('case_id')
            if cid in labels:raise ValueError('duplicate independent gold case_id')
            labels[cid]=item
        report=evaluate(rows(a.predictions),labels,bootstrap_repetitions=10000)
        save(a.out,report)
        print(json.dumps({'status':'evaluated_independent_gold','cases':report['n_cases'],'out':a.out}))
        return
    if a.mode=='conformal' and not a.calibration:
        raise ValueError('conformal mode requires an independent calibration file')
    if a.mode!='conformal' and a.calibration:
        raise ValueError('calibration may be used only in conformal mode')
    calibration=json.loads(Path(a.calibration).read_text(encoding='utf-8')) if a.calibration else None
    output=[];cases=list(rows(a.cases))
    for case in cases:
        policy=case['policy'];ids=[r['rule_id'] for r in policy['rules']]
        observations=case['observations']
        if a.mode=='oracle':
            states=derive_truth_states(policy,observations)
            sets={rid:[v] if v!='U' else ['T','F','U'] for rid,v in states.items()}
            assurance='oracle_observation_label_only'
        elif a.mode=='hard':
            sets=frozen_a1_hard_sets(case['frozen_A1'],ids)
            assurance='uncalibrated_frozen_A1_hard_labels'
        else:
            dist=require_a1_probabilities(case,ids)
            sets=make_prediction_sets(dist,calibration)
            assurance='conditional_on_case_exchangeability_and_sound_policy'
        result=decide_or_ask(policy,observations,sets,assurance=assurance)
        output.append({'case_id':case['case_id'],'mode':a.mode,'state_sets':sets,**result})
    save(a.out,output)
    print(json.dumps({'status':'completed_CPU_no_neural_forward','cases':len(output),'out':a.out}))

if __name__=='__main__':main()
