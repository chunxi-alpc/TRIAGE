"""Reference-path selective coverage/risk, without off-path semantic labels."""
import argparse,collections,json
from pathlib import Path
from .audit import read,write,load_units
from src.aster.manuscript_core import plan_gains
from experiments.aster_manuscript_clinical_20261030.engine import decisions_from

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((run/'frozen.json').read_text());_,classes=load_units(run,'confirm');cases=read(run/'cases/confirm.jsonl')
    sums=collections.Counter();by_depth=collections.defaultdict(collections.Counter)
    for case in cases:
        cls=classes.get(case['case_id']);sums['registered_cases']+=1
        if not cls or 'raw_decisions' not in cls:sums['case_classification_unavailable']+=1;continue
        plan=plan_gains(case['guideline'],decisions_from(cls['raw_decisions'],freeze),g_plus=freeze['g_plus'],g_minus=freeze['g_minus'])
        for depth,rid in enumerate(case['gold_rules']):
            state=plan['groups']['G'+rid[1:-1]];c=by_depth[depth];c['reference_groups']+=1;sums['reference_groups']+=1
            for counter in [c,sums]:
                counter['reachable']+=state['reachable'];counter['enabled']+=state['enabled']
                counter['accepted_correct']+=state['enabled'] and state['prediction']==rid
                counter['accepted_incorrect']+=state['enabled'] and state['prediction']!=rid
                counter['NONE']+=state['reachable'] and state['reason']=='none'
                counter['abstain']+=state['reachable'] and state['reason']=='abstain'
                counter['interface_failure']+=state['reachable'] and state['reason']=='classification_failure'
                counter['ancestor_blocked']+=not state['reachable']
    def metrics(c):
        d=dict(c);n=c['reference_groups'];d['coverage']=c['enabled']/n if n else None
        d['selective_risk']=c['accepted_incorrect']/c['enabled'] if c['enabled'] else None
        return d
    report={'scope':'existing benchmark reference path; off-path groups have no semantic gold label',
            'frozen_probability':freeze['thresholds']['probability'],'frozen_margin':freeze['thresholds']['margin'],
            'overall':metrics(sums),'depths':{str(d):metrics(c) for d,c in sorted(by_depth.items())},
            'independence':'Counts are descriptive; path groups are not independent samples; use source clusters for inference.'}
    write(out/'risk.json',report);print(json.dumps(report['overall']))
if __name__=='__main__':main()
