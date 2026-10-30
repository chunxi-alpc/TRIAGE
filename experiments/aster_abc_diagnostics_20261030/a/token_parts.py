"""CPU decomposition of saved GPU keys into rule-ID and action token spans."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from src.aster.manuscript_runtime import PromptBuilder
from .common import BASE,write_json,source_mean_ci,cosine
from .gpu_controls import variant

def id_token_count(builder,row):
    prompt=builder.prepare(row['guideline'],row['states']['positive'],row['question'],online=False)
    guide,spans=builder.guideline(row['guideline']);start=prompt.text.rfind(guide)
    if start<0:raise ValueError('constructed guide missing')
    left,right=spans[row['target_rule_id']][0];left+=start;right+=start
    offsets=builder.tokenizer(prompt.text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    ids=[i for i,(a,b) in enumerate(offsets) if b>left and a<right and b>a]
    if list(prompt.positions[row['target_rule_id']][:len(ids)])!=ids:raise ValueError('rule-ID span is not the first editable span')
    return len(ids)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--base',default=str(BASE));p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);base=Path(args.base);expert=json.loads((base/'analysis/expert.json').read_text());tok=AutoTokenizer.from_pretrained(base/'models/Qwen3-8B-2602814',local_files_only=True);builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
    raw=[]
    for path in sorted(Path(args.gpu).glob('*.pt')):
        r=torch.load(path,map_location='cpu',weights_only=False);orig=r['variants']['base'];n0=id_token_count(builder,r['row'])
        for control in ['base','action_swap','paraphrase','position_control']:
            changed=r['variants'].get(control)
            if changed is None:continue
            nc=id_token_count(builder,variant(r['row'],control))
            for name in expert['selected_heads']:
                h=tuple(map(int,name.split(':')));v=torch.tensor(expert['candidates'][name]['direction'],dtype=torch.float64)
                for part in ['rule_id','action']:
                    sl0=slice(None,n0) if part=='rule_id' else slice(n0,None);slc=slice(None,nc) if part=='rule_id' else slice(nc,None)
                    d0=(orig['offline']['positive'][h][sl0]-orig['offline']['negative'][h][sl0]).mean(0);dc=(changed['offline']['positive'][h][slc]-changed['offline']['negative'][h][slc]).mean(0)
                    a,b=float(d0@v),float(dc@v)
                    raw.append({'pair_id':r['row']['pair_id'],'source_id':r['row']['source_id'],'head':name,'variant':control,'part':part,
                        'base_projection':a,'control_projection':b,'relative_projection_change':None if abs(a)<1e-12 else (b-a)/abs(a),'direction_cosine':cosine(d0,dc),
                        'base_tokens':len(orig['offline']['positive'][h][sl0]),'control_tokens':len(changed['offline']['positive'][h][slc])})
    groups=defaultdict(list)
    for r in raw:groups[(r['head'],r['variant'],r['part'])].append(r)
    summary={'status':'complete','kind':'post hoc CPU token decomposition of saved GPU keys; no new GPU run','sources':len({r['source_id'] for r in raw}),
        'metrics':{'/'.join(k):{m:source_mean_ci(g,m) for m in ['relative_projection_change','direction_cosine']} for k,g in groups.items()},
        'limitation':'ID and action contexts also change; token decomposition localizes effects but does not remove contextual confounding; two sources remain descriptive'}
    write_json(out/'summary.json',summary)
    with (out/'records.jsonl').open('x') as f:
        for r in raw:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','sources':summary['sources'],'observations':len(raw)}))
if __name__=='__main__':main()
