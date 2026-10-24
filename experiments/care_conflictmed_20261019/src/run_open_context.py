"""Exploratory continuation: frozen questions/direction, free-answer conflicts, no choices.
Launched after first-run results were seen; never relabeled as preregistered primary.
"""
from run import *

def main():
    if (ROOT/'results/open_context.jsonl').exists():raise RuntimeError('Continuation output exists')
    cfg=load(ROOT/'config.json');specs=[s for s in load(ROOT/'data/open_probes.json') if s['split']=='test']
    items={r['id']:r for r in load(ROOT/'data/items.json')}
    selection=load(ROOT/'results/selection.json');alpha=selection['selected']['alpha']
    protocol=dict(status='frozen_before_continuation',scope='exploratory after main results; no changes to main criteria or output',test_facts=len(specs),questions=2,conditions=['misleading','aligned','caution','misleading_with_frozen_direction'],max_generated=8*len(specs),selection_sha256=sha(ROOT/'results/selection.json'),open_questions_sha256=sha(ROOT/'data/open_probes.json'),alpha=alpha,utc=now(),primary_subset='same strict set as initial experiment; recall-only sensitivity reported separately')
    dump(ROOT/'audit/open_context_protocol.json',protocol)
    cfg['environment_filename']='open_context_environment.json'
    start=time.monotonic();e=Engine(cfg)
    direction_path=ROOT/'results/direction.pt'
    if not direction_path.exists():direction_path=ROOT/'results/direction_weights.pt'
    d=torch.load(direction_path,map_location='cpu',weights_only=True);e.layer=d['layer'];e.direction=d['direction']
    for cond in ['misleading','aligned','caution','misleading_with_frozen_direction']:
        jobs=[]
        for s in specs:
            r=items[s['id']]
            for qi,q in enumerate(s['questions']):
                context=r['correct'] if cond=='aligned' else r['wrong']
                prompt='Retrieved clinical note:\n'+context+'\n\n'
                if cond=='caution':prompt+='The note may contain an error. Check it against your medical knowledge before answering.\n'
                prompt+='Question: '+q+'\nGive only a concise answer, without explanation.'
                jobs.append(dict(id=r['id'],group=r['group'],split='test',question_variant=qi,condition=cond,prompt=prompt))
        out=batches(e,jobs,ROOT/'results/open_context.jsonl',cond,alpha if cond.endswith('frozen_direction') else 0)
        for row in out:
            s=next(s for s in specs if s['id']==row['id'])
            raw=row['raw_output'];correct=open_match(raw,s)
            wrong=bool(re.search(s['wrong_pattern'],raw,re.I)) and not bool(re.search(s['correct_pattern'],raw,re.I))
            append(ROOT/'results/open_context_scored.jsonl',dict(id=row['id'],group=row['group'],question_variant=row['question_variant'],condition=cond,correct=correct,explicit_wrong_match=wrong,raw_output=raw,truncated=row['truncated']))
    dump(ROOT/'results/open_context_completed.json',dict(status='complete',seconds=time.monotonic()-start,n=8*len(specs),peak_allocated_bytes=torch.cuda.max_memory_allocated(),sha256=sha(ROOT/'results/open_context.jsonl'),scored_sha256=sha(ROOT/'results/open_context_scored.jsonl'),utc=now()))
    print('OPEN_CONTEXT_COMPLETE',flush=True)

if __name__=='__main__':main()
