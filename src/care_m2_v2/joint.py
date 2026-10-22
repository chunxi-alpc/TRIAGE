from pathlib import Path
import time,torch,shutil
from care_m2_inc.io import read_json,write_json,write_jsonl
from .engine import Engine
from .study import load,spec,summarize,correct
from .data import span,digest

def run_joint(args,root,out):
    from .cli import save_manifest
    started=time.time();e=Engine(args.model,out/'raw')
    axes={t:torch.load(root/t/'axes.pt',weights_only=True) for t in ('N1','N2','N3')}
    sel={t:read_json(root/t/'dev_selection.json') for t in axes}
    n1go=read_json(root/'N1/gate.json')['go']
    methods={'baseline':[], 'source_only':['C'],'dq_only':['D'],'rule_only':['A'],
             'source_dq':['C','D'],'dq_source':['D','C'],'source_rule':['C','A'],'rule_source':['A','C'],
             'dq_rule':['D','A'],'rule_dq':['A','D'],'route':['D','C','A'],'text':[]}
    write_json(out/'frozen_components.json',{'N1_success':n1go,'selections':sel,'methods':methods,'no_refit_no_reselection':True})
    predictions=[]
    for r in load(args.data,'N4','test'):
        c=r['controls']
        for method,sequence in methods.items():
            edits=[];biases=[];prompt=r['prompt']
            for component in sequence:
                task={'C':'N2','D':'N3','A':'N1'}[component]
                signal={'source':c['source']} if component=='C' else {'DQ':c['DQ']} if component=='D' else {'R':1,'A':c['A']}
                rr={**r,'task':task}
                if component=='A':rr['spans']=[span(prompt,'use the selected source route')]
                arm='direct_bias' if component=='A' and not n1go else 'route'
                _,ee,bb=spec(e,rr,signal,arm,sel[task]['selected'][arm],axes[task]);edits.extend(ee);biases.extend(bb)
            if method=='text':
                from .study import append_policy
                prompt=append_policy(prompt,f"First {'decide the unique action' if c['DQ'] else 'ask the decisive missing observation'}. Then use {'Context' if c['source']=='C' else 'Prior'} record. The processing rule is {'applicable' if c['A'] else 'inapplicable'}. Follow rule priority.")
            result=e.score(prompt,r['options'],edits,biases)
            predictions.append({**result,'method':method,'case_id':r['case_id'],'family_id':r['family_id'],'pair_id':r['pair_id'],'split':'test',
              'control':c,'correct':correct(r,c,result['prediction']),'expected':r['gold'],'option_labels':r['options'],'control_origin':'oracle_diagnostic'})
    write_jsonl(out/'test_predictions.jsonl',predictions);summary=summarize('N4',predictions,out)
    write_json(out/'gate.json',{'status':'gpu_evaluated','N2_go':True,'N3_go':True})
    save_manifest(e,out,'N4',started,'gpu_evaluated');del e;torch.cuda.empty_cache()
    return summary
