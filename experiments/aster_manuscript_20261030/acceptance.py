"""Real local-weight interface acceptance; raw prompts stay on the server."""
import argparse, json, subprocess, time, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.aster.manuscript_runtime import PromptBuilder, ManuscriptBackbone, aligned_triplet
from src.aster.manuscript_core import PairMoments, FitConfig, rotate, edit_kv_layers, select_expert, plan_gains
from src.aster import ActionBlindBranchClassifier, HFLabelScorer

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dtype',required=True); p.add_argument('--output',required=True); p.add_argument('--model',default='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'); p.add_argument('--model-revision',default='local-modelscope-snapshot-20261014-upstream-revision-unavailable'); p.add_argument('--triplets'); p.add_argument('--pair-index',type=int,default=0)
    a=p.parse_args(); out=Path(a.output); out.mkdir(exist_ok=False,parents=True)
    start=time.time(); checks={}; report={'status':'running','dtype':a.dtype,'checks':checks}; save(out/'status.json',report)
    try:
        torch.set_num_threads(4); torch.manual_seed(20261030)
        model_path=a.model
        tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
        guide={'groups':[{'group_id':'g','mutually_exclusive':True,'ancestor_conditions':[], 'rules':[
            {'rule_id':'R1','condition':'x=1','action':'Alpha'}, {'rule_id':'R2','condition':'x=0','action':'Beta'}]}]}
        row={'pair_id':'interface-logic','source_id':'synthetic-interface-only','split':'fit','guideline':guide,
             'target_rule_id':'R1','question':'选择哪个规则？','states':{'positive':'x=1\n编码：2表示未记录。','negative':'x=0\n编码：2表示未记录。','unknown':'x=2\n编码：2表示未记录。'},
             'verification':{'verifier':'explicit-x-equality-v1','reference':'synthetic logical patient; no clinical gold',
             'positive_rule_id':'R1','negative_rule_id':'R2','unknown_label':'NONE','ancestors_reachable':True,'independent_of_classifier':True}}
        if a.triplets:
            row=json.loads(Path(a.triplets).read_text().splitlines()[a.pair_index]); guide=row['guideline']
        target,offline=aligned_triplet(builder,row)
        online={s:builder.prepare(guide,t,row['question'],online=True) for s,t in row['states'].items()}
        save(out/'tokens.json',{'row':row,'template_id':builder.template_id,'prompts':{
            mode:{s:{'text':q.text,'token_ids':q.input_ids[0].tolist(),'positions':q.positions,'prefix_length':q.prefix_length} for s,q in prompts.items()}
            for mode,prompts in [('offline',offline),('online',online)]}})
        checks['triplet_token_alignment']=True
        for q in online.values(): assert q.text.count('患者：\n')==1
        checks['patient_once']=True
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=getattr(torch,a.dtype),attn_implementation='eager').to('cuda:0').eval()
        backend=ManuscriptBackbone(model,model_id='Qwen/Qwen3-8B',model_revision=a.model_revision)
        report['identity']=backend.identity
        snaps={s:backend.extract(q,target,keys=True,queries=False) for s,q in offline.items()}
        queries={s:backend.extract(online[s],target,keys=False,queries=True) for s in ['positive','negative']}
        h=(18,0); moment=PairMoments(); moment.update(snaps['positive'].keys[h],snaps['negative'].keys[h]); cand=moment.estimate(FitConfig(1,1))
        direction=torch.tensor(cand['direction'],dtype=torch.float64)
        got={}; attn=model.model.layers[18].self_attn
        def kh(m,args,value): got['k']=value[0,list(offline['positive'].positions[target]),0,:].detach().double().cpu()
        handle=attn.k_norm.register_forward_hook(kh)
        with backend.session(): backend._forward(offline['positive'].input_ids)
        handle.remove(); assert torch.equal(got['k'],snaps['positive'].keys[h]); checks['post_k_norm_pre_rope']=True
        context={}
        def ph(m,args,kw): context['cos'],context['sin']=kw['position_embeddings']
        def qh(m,args,value):
            got['q']=rotate(value[0,-1,:4,:],context['cos'][0,-1],context['sin'][0,-1]).detach().double().cpu()
        handles=[attn.register_forward_pre_hook(ph,with_kwargs=True),attn.q_norm.register_forward_hook(qh)]
        with backend.session(): backend._forward(online['positive'].input_ids)
        for handle in handles: handle.remove()
        assert torch.equal(got['q'],queries['positive'].queries[h]); checks['rotated_all_four_gqa_queries']=True
        prompt=online['positive']; prefix=backend.prefill(prompt)
        original=prefix.cache.to_legacy_cache(); before=[(k.clone(),v.clone()) for k,v in original]
        gains={'R1':1.0,'R2':-1.0}
        layers,audit=edit_kv_layers(original,{h:direction},prompt.positions,gains,prefix.cos,prefix.sin)
        mask=torch.zeros(original[18][0].shape,dtype=torch.bool,device='cuda:0')
        for rid,indices in prompt.positions.items(): mask[0,0,list(indices)]=True
        assert torch.equal(original[18][0][~mask],layers[18][0][~mask])
        for i,((k,v),(bk,bv),(ak,av)) in enumerate(zip(original,before,layers)):
            assert torch.equal(k,bk) and torch.equal(v,bv) and torch.equal(v,av)
            if i!=18: assert torch.equal(k,ak)
        checks['original_cache_values_and_nontarget_unchanged']=True; report['edit']=audit
        q=queries['positive'].queries[h]; actual=[]; expected=[]; bounds=[]
        for rid,indices in prompt.positions.items():
            for j in indices:
                d=(layers[18][0][0,0,j].double()-original[18][0][0,0,j].double()).cpu()
                ideal=gains[rid]*rotate(direction,prefix.cos[j],prefix.sin[j])
                observed=(q@d)/backend.dim**.5; theory=(q@ideal)/backend.dim**.5
                bound=q.norm(dim=-1)*(d-ideal).norm()/backend.dim**.5
                assert torch.all((observed-theory).abs()<=bound+1e-10)
                actual.append(observed.tolist()); expected.append(theory.tolist()); bounds.append(bound.tolist())
        report['score_equation']={'actual':actual,'ideal':expected,'rounding_bound':bounds,
             'max_abs_error':float((torch.tensor(actual)-torch.tensor(expected)).abs().max())}
        checks['fixed_query_score_formula_with_rounding_bound']=True
        report['tiny_gain_edit']=edit_kv_layers(original,{h:direction},prompt.positions,{'R1':1e-8,'R2':-1e-8},prefix.cos,prefix.sin)[1]
        empty=select_expert({}, {},FitConfig(1,0),identity=backend.identity,provenance={'template_id':builder.template_id})
        outputs=[]; logits=[]
        with backend.session():
            for expert,gg in [(None,{'R1':0.,'R2':0.}),(empty,{'R1':0.,'R2':0.}),(empty,gains)]:
                outputs.append(backend.generate(prompt,expert,gg,max_new_tokens=4,prefix=prefix))
                cloned,_=edit_kv_layers(original,{},prompt.positions,gg,prefix.cos,prefix.sin)
                cache=type(prefix.cache).from_legacy_cache(tuple(cloned))
                logits.append(backend._forward(prompt.input_ids[:,prompt.prefix_length:],cache=cache,start=prompt.prefix_length,use_cache=True).logits.detach().float())
            full=backend._forward(prompt.input_ids).logits.detach().float()
        assert outputs[0]['token_ids']==outputs[1]['token_ids']==outputs[2]['token_ids']
        assert torch.equal(logits[0],logits[1]) and torch.equal(logits[0],logits[2])
        checks['segmented_zero_empty_logits_equal']=True
        report['full_vs_segmented']={'max_abs_logit_error':float((full-logits[0]).abs().max()),
          'mean_abs_logit_error':float((full-logits[0]).abs().mean()),'argmax_equal':int(full.argmax())==int(logits[0].argmax())}
        backend.generate(online['negative'],None,{'R1':0.,'R2':0.},max_new_tokens=2,prefix=prefix)
        for (k,v),(bk,bv) in zip(original,before): assert torch.equal(k,bk) and torch.equal(v,bv)
        checks['two_patients_share_original_prefix']=True
        classifier=ActionBlindBranchClassifier(HFLabelScorer(model,tok,max_input_tokens=4096),temperature=1.,probability_threshold=0.,margin_threshold=0.)
        report['classification']=classifier.run(guide,row['states']['positive'],include_logits=True)
        assert classifier.scorer.model is backend.model; checks['classifier_generation_same_model']=True
        failure=plan_gains(guide,{'g':None},g_plus=1,g_minus=1)
        assert all(g==0 for g in failure['gains'].values()); report['classification_failure']=failure
        child={'group_id':'child','parent_group_id':'g','parent_rule_id':'R1','mutually_exclusive':True,'ancestor_conditions':[guide['groups'][0]['rules'][0]['condition']],
               'rules':[{'rule_id':'C1','condition':'y=1','action':'Gamma'},{'rule_id':'C2','condition':'y=0','action':'Delta'}]}
        closed=plan_gains({'groups':guide['groups']+[child]},{'g':{'prediction':'R1','abstain':True},'child':{'prediction':'C1','abstain':False}},g_plus=1,g_minus=1)
        assert closed['gains']['C1']==closed['gains']['C2']==0; checks['ancestor_abstention_closes_child']=True
        report['ancestor_abstention']=closed
        torch.save({'offline':{s:snap.keys for s,snap in snaps.items()},'online_queries':{s:snap.queries for s,snap in queries.items()},'segmented_logits':logits[0].cpu(),'full_logits':full.cpu()},out/'numerical.pt')
        report.update(status='passed',elapsed_seconds=time.time()-start,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),upstream_model_revision_verified=a.model_revision!='local-modelscope-snapshot-20261014-upstream-revision-unavailable')
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.time()-start)
        traceback.print_exc()
    save(out/'status.json',report)
    return 0 if report['status']=='passed' else 1
if __name__=='__main__': raise SystemExit(main())

