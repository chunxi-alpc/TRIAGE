"""Train two M3 variants, select on autonomous dev paths, evaluate five methods."""
import argparse, gc, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .engine import Engine
from .m3 import LearnableM3
from .data import (read_rows,save_json,append,prepare_data,canonical_guide,
                   target_prompt,reference_prompt)

ROOT=Path('/home/gcx/TRIAGE/results/aster_learnable_m3_20261106')
BASIS=Path('/home/gcx/aster_existing_data_20261106/operators/aster.pt')
SEED=20261105
METHODS=('full','aster','learned_reference','learned_self','direct_reference')

def load_editor(engine, frozen):
    return LearnableM3(frozen,engine.n_heads).to(engine.device)

def reference_heads(engine,case,gid,mode,base=None):
    if mode=='self':return base['heads'].flatten(0,1).to(engine.device)
    context=engine.prepare(reference_prompt(case,gid))
    result=engine.run(context)
    del context
    return result['heads'].flatten(0,1).to(engine.device)

def score_node(engine,case,gid,method,frozen,editors):
    diag={}
    if method=='direct_reference':
        context=engine.prepare(reference_prompt(case,gid));result=engine.run(context,capture=False)
    elif method=='learned_reference':
        ref=reference_heads(engine,case,gid,'reference')
        context=engine.prepare(target_prompt(case,gid))
        result=engine.run(context,m3=editors['reference'],reference=ref,capture=False)
    else:
        context=engine.prepare(target_prompt(case,gid))
        if method=='full':result=engine.run(context,capture=False)
        elif method=='aster':result=engine.run(context,operators=frozen,capture=False)
        else:
            base=engine.run(context)
            ref=reference_heads(engine,case,gid,'self',base)
            result=engine.run(context,m3=editors['self'],reference=ref,capture=False)
    for key in ('magnitude','signal_change','gate'):
        if key in result:diag[key]=float(result[key].detach())
    return result['ell'],diag

def predict_case(engine,case,method,frozen,editors):
    # Only model inputs enter traversal. Gold fields are read after it terminates.
    tree=canonical_guide(case['guideline']);nodes={n['id']:n for n in tree['nodes']}
    gid='G0';path=[];diags=[]
    while gid not in tree['leaves']:
        if any(n['gid']==gid for n in path):raise ValueError('Cycle in input tree')
        ell,diag=score_node(engine,case,gid,method,frozen,editors)
        if not math.isfinite(ell):raise ValueError('Nonfinite model score')
        label=int(ell>0);path.append(dict(gid=gid,label=label,ell=ell));diags.append(diag)
        gid=nodes[gid]['yes' if label else 'no']
    predicted=[dict(gid=n['gid'],label=n['label']) for n in path]
    return dict(case_id=case['case_id'],source_id=case['source_id'],method=method,
        predicted_path=predicted,predicted_leaf=gid,predictions=path,
        correct=predicted==case['gold_path'] and gid==case['gold_leaf'],
        gold_path=case['gold_path'],gold_leaf=case['gold_leaf'],status='complete',
        magnitude=sum(d.get('magnitude',0) for d in diags)/max(1,len(diags)),
        signal_change=sum(d.get('signal_change',0) for d in diags)/max(1,len(diags)))

def verify(engine,frozen,root):
    case=read_rows(root/'data/train.jsonl')[0];gid=case['gold_path'][0]['gid']
    editor=load_editor(engine,frozen)
    ref=reference_heads(engine,case,gid,'reference')
    context=engine.prepare(target_prompt(case,gid));base=engine.run(context)
    zero=engine.run(context,m3=editor,reference=ref,force_zero=True)
    result=engine.run(context,m3=editor,reference=ref,training=True)
    result['logits_tensor'].sum().backward()
    checks=dict(zero_edit_difference=abs(zero['ell']-base['ell']),
        only_m3_gradients=all(p.grad is None for p in engine.model.parameters()),
        all_mappings_and_gates_receive_gradient=all(p.grad is not None for p in editor.parameters()),
        trainable_parameters=sum(p.numel() for p in editor.parameters()),
        heads=sorted(int(h) for h in frozen),edited_heads=result['edited_heads'],
        position='final input token, o_proj input',
        gold_independence='predict_case traversal uses only patient/guideline and model scores; gold compared after traversal')
    assert checks['zero_edit_difference']==0
    assert checks['only_m3_gradients'] and checks['all_mappings_and_gates_receive_gradient']
    assert checks['edited_heads']==len(frozen)==51
    # Change all labels; traversal and reference text must remain identical.
    changed=dict(case,gold_path=[],gold_leaf='unused')
    assert reference_prompt(case,gid)==reference_prompt(changed,gid)
    a=predict_case(engine,case,'learned_reference',frozen,{'reference':editor})
    b=predict_case(engine,changed,'learned_reference',frozen,{'reference':editor})
    assert a['predicted_path']==b['predicted_path'] and a['predicted_leaf']==b['predicted_leaf']
    checks['prediction_label_independence']=True
    save_json(root/'checks.json',checks)

def train(engine,frozen,root,mode):
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    editor=load_editor(engine,frozen);opt=torch.optim.AdamW(editor.parameters(),lr=.001,weight_decay=.01)
    trainset=read_rows(root/'data/train.jsonl');dev=read_rows(root/'data/dev.jsonl')
    dest=root/mode;dest.mkdir(exist_ok=True);best=None
    start=time.perf_counter();total_cases=0
    for epoch in range(1,4):
        order=list(trainset);random.Random(SEED+epoch).shuffle(order)
        opt.zero_grad(set_to_none=True)
        for i,case in enumerate(order):
            path=case['gold_path'];selected=sorted(set((0,len(path)//2,len(path)-1)))
            group_size=min(8,len(order)-(i//8)*8)
            for index in selected:
                node=path[index];gid=node['gid']
                if mode=='reference':ref=reference_heads(engine,case,gid,mode)
                context=engine.prepare(target_prompt(case,gid));base=engine.run(context,capture=mode=='self')
                if mode=='self':ref=reference_heads(engine,case,gid,mode,base)
                edited=engine.run(context,m3=editor,reference=ref,training=True,capture=False)
                logits=edited['logits_tensor'];target=torch.tensor(0 if node['label'] else 1,device=engine.device)
                ce=F.cross_entropy(logits[None],target[None])
                base_logits=torch.tensor([base['ell'],0.],device=engine.device)
                keep=ce*0
                if base['prediction']==node['label']:
                    pb=base_logits.softmax(-1)
                    keep=(pb*(pb.clamp_min(1e-12).log()-logits.log_softmax(-1))).sum()
                loss=ce+.1*keep+.001*edited['magnitude']
                (loss/len(selected)/group_size).backward()
                append(dest/'training_nodes.jsonl',dict(epoch=epoch,case_id=case['case_id'],gid=gid,
                    loss=float(loss.detach()),ce=float(ce.detach()),preserve=float(keep.detach()),
                    magnitude=float(edited['magnitude'].detach()),signal_change=float(edited['signal_change'].detach())))
                del context,edited,logits,loss,ce,keep,ref
            if (i+1)%8==0 or i+1==len(order):
                opt.step();opt.zero_grad(set_to_none=True)
            total_cases+=1
            save_json(dest/'status.json',dict(stage='train',epoch=epoch,cases=i+1,total_cases=total_cases,
                elapsed_seconds=time.perf_counter()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/2**30))
            if total_cases==20:
                save_json(dest/'first20.json',dict(seconds=time.perf_counter()-start,
                    seconds_per_case=(time.perf_counter()-start)/20,
                    train_three_epoch_hours=(time.perf_counter()-start)/20*600/3600,
                    peak_memory_gb=torch.cuda.max_memory_allocated()/2**30))
        torch.save(dict(state_dict=editor.state_dict(),epoch=epoch,mode=mode),dest/f'epoch_{epoch}.pt')
        results=[]
        for case in dev:
            r=predict_case(engine,case,'learned_'+mode,frozen,{mode:editor})
            results.append(r);append(dest/f'dev_epoch_{epoch}.jsonl',r)
        accuracy=sum(r['correct'] for r in results)/len(results)
        magnitude=sum(r['magnitude'] for r in results)/len(results)
        key=(accuracy,-magnitude,-epoch)
        append(dest/'selection.jsonl',dict(epoch=epoch,path_accuracy=accuracy,magnitude=magnitude))
        if best is None or key>best:
            best=key;torch.save(dict(state_dict=editor.state_dict(),epoch=epoch,mode=mode),dest/'selected.pt')
            save_json(dest/'selected.json',dict(epoch=epoch,path_accuracy=accuracy,magnitude=magnitude))
    save_json(dest/'status.json',dict(stage='complete',elapsed_seconds=time.perf_counter()-start))

def evaluate(engine,frozen,root,shard,shards,worklist=None,dynamic_queue=None):
    editors={}
    for mode in ('reference','self'):
        editor=load_editor(engine,frozen)
        editor.load_state_dict(torch.load(root/mode/'selected.pt',map_location=engine.device,weights_only=False)['state_dict'])
        editor.eval();editors[mode]=editor
    out=root/'test';out.mkdir(exist_ok=True)
    output=out/(f'predictions_shared_{shard}.jsonl' if dynamic_queue else f'predictions_rebalanced_{shard}.jsonl' if worklist else f'predictions_{shard}.jsonl')
    done={(r['case_id'],r['method']) for path in out.glob('predictions_*.jsonl') for r in read_rows(path)}
    cases=read_rows(root/'data/test.jsonl')
    if dynamic_queue:
        import json
        assigned=set(json.loads(Path(dynamic_queue).read_text())['case_ids'])
        cases=[c for c in cases if c['case_id'] in assigned]
    elif worklist:
        import json
        assigned=set(json.loads(Path(worklist).read_text())[str(shard)])
        cases=[c for c in cases if c['case_id'] in assigned]
    else:cases=cases[shard::shards]
    start=time.perf_counter()
    already_complete=sum(all((c['case_id'],m) in done for m in METHODS) for c in cases)
    remaining_cases=len(cases)-already_complete
    processed=0
    def scheduled_cases():
        if not dynamic_queue:
            yield from enumerate(cases)
            return
        import fcntl,json
        by_id={c['case_id']:c for c in cases}
        queue_path=Path(dynamic_queue)
        while True:
            with queue_path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                queue=json.loads(queue_path.read_text())
                index=queue['next']
                if index>=len(queue['case_ids']):return
                cid=queue['case_ids'][index]
                queue['next']=index+1
                queue.setdefault('claimed',{})[cid]=shard
                save_json(queue_path,queue)
            yield index,by_id[cid]
    for i,case in scheduled_cases():
        if all((case['case_id'],m) in done for m in METHODS):continue
        for method in METHODS:
            if (case['case_id'],method) in done:continue
            result=predict_case(engine,case,method,frozen,editors);append(output,result)
        processed+=1
        save_json(out/f'status_{shard}.json',dict(stage='test',cases=processed if dynamic_queue else i+1,total=len(cases),dynamic_queue=dynamic_queue,
            elapsed_seconds=time.perf_counter()-start,processed_since_resume=processed,
            complete_before_resume=already_complete,peak_memory_gb=torch.cuda.max_memory_allocated()/2**30))
        if processed==20:
            save_json(out/f'first20_{shard}.json',dict(seconds=time.perf_counter()-start,
                estimated_shard_hours=(time.perf_counter()-start)/20*remaining_cases/3600,
                peak_memory_gb=torch.cuda.max_memory_allocated()/2**30))
            save_json(root/f'test_resource_update_{shard}.json',dict(
                observed_cases=20,methods=list(METHODS),shard=shard,
                estimated_shard_hours=(time.perf_counter()-start)/20*remaining_cases/3600,
                peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
    save_json(out/f'status_{shard}.json',dict(stage='complete',cases=processed if dynamic_queue else len(cases),dynamic_queue=dynamic_queue,elapsed_seconds=time.perf_counter()-start))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','verify','train','evaluate'],required=True)
    parser.add_argument('--mode',choices=['reference','self']);parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--worklist');parser.add_argument('--dynamic-queue')
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare_data(root);return
    engine=Engine()
    basis=root/'frozen_m1_m2.pt'
    frozen=torch.load(basis if basis.exists() else BASIS,map_location='cpu',weights_only=False)
    if args.stage=='verify':verify(engine,frozen,root)
    elif args.stage=='train':train(engine,frozen,root,args.mode)
    else:evaluate(engine,frozen,root,args.shard,args.shards,args.worklist,args.dynamic_queue)
if __name__=='__main__':main()
