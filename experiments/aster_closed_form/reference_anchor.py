"""Fixed same-fact reference experiment. No fitting, selection, or new annotation."""
import argparse, copy, json, os, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import torch
from .common import rows, write_rows, write_json, append, status, SEED
from . import engine as engine_module
from .existing import historical_template
from .soft_angle_report import save_table
from .data import canonical_guide
ROOT=Path("/home/gcx/aster_reference_anchor_20261106")
SOURCE=Path("/home/gcx/aster_existing_data_20261106")
SOFT=Path("/home/gcx/aster_soft_angle_20261106")
ORIGINAL=Path("/home/gcx/aster_closed_form_20261105")
HISTORY=Path("/home/gcx/semantic_masking_followup_20261105")
MAIN_SYSTEM=engine_module.SYSTEM

def remove_suffix(patient, suffix):
    marked="\n"+suffix
    assert suffix and patient.endswith(marked), "Saved background is not a terminal suffix"
    return patient[:-len(marked)]

def main_reference(v, suffix, construction):
    original=v["prompt"]
    ref=v["prompts"]["boundary_guideline"] if v["boundary_annotation_available"] else original
    operations=[]
    if v["boundary_annotation_available"]:
        assert construction.get("status")=="complete"
        protected=set(construction["value"].get("protected_action_ids",[]))
        # Every condition, transition, definition and patient line stays unchanged.
        a, tail_a=original.split("\n\n\u6307\u5357\u539f\u6587\u5b9a\u4e49\u4e0e\u8bf4\u660e\uff1a\n",1)
        b, tail_b=ref.split("\n\n\u6307\u5357\u539f\u6587\u5b9a\u4e49\u4e0e\u8bf4\u660e\uff1a\n",1)
        assert tail_a==tail_b
        al,bl=a.splitlines(),b.splitlines()
        assert len(al)==len(bl)
        for x,y in zip(al,bl):
            leaf=x.split(": ",1)[0]
            if leaf.startswith("L") and leaf[1:].isdigit() and leaf not in protected:
                assert y==leaf+": [\u5904\u7f6e\u5df2\u906e\u6321]"
            else:
                assert x==y
        if ref!=original:
            operations.append(dict(operation="saved_boundary_guideline",protected_action_ids=sorted(protected),
                annotation=str(ORIGINAL/"data/constructions.jsonl"),group_id=v["group_id"]))
    patient=v["patient"]
    if suffix is not None:
        short=remove_suffix(patient,suffix)
        marker="\n\n\u75c5\u5386\uff1a\n"+patient+"\n\n\u5f53\u524d\u8282\u70b9\uff1a"
        replacement="\n\n\u75c5\u5386\uff1a\n"+short+"\n\n\u5f53\u524d\u8282\u70b9\uff1a"
        assert ref.count(marker)==1
        ref=ref.replace(marker,replacement,1)
        patient=short
        operations.append(dict(operation="remove_paired_background_suffix",text=suffix,
            annotation=str(SOURCE/"data/pairs.jsonl")))
    assert "\n\u5f53\u524d\u6761\u4ef6\uff1a"+v["condition"]+"\n" in ref
    assert "\n\n\u75c5\u5386\uff1a\n"+patient+"\n\n\u5f53\u524d\u8282\u70b9\uff1a" in ref
    return ref,operations

def historical_reference(r, make_prompt):
    altered=copy.deepcopy(r);operations=[]
    if r["population"]=="constructed":
        suffix=r["construction"][r["background"]+"_background"]
        altered["patient"]=remove_suffix(r["patient"],suffix)
        operations.append(dict(operation="remove_saved_background_suffix",text=suffix,
            annotation=str(HISTORY/"data/inputs_full.jsonl")))
    cuts=defaultdict(list)
    for c in r["changes"].get("conservative_cuts",[]):
        cuts[c["block_id"]].append(c)
    context=r["context"]
    for block in r["changes"].get("deleted_blocks",[]):
        text=block["text"]; bc=cuts.get(block["block_id"],[])
        if not bc or text not in context: continue
        edited=text
        spans=sorted(bc,key=lambda c:c["start_in_block"],reverse=True)
        boundary=len(text)
        for c in spans:
            lo,hi=c["start_in_block"],c["end_in_block"]
            assert text[lo:hi]==c["quote"] and hi<=boundary
            boundary=lo
            edited=edited[:lo]+edited[hi:]
        context=context.replace(text,edited)
        operations.append(dict(operation="saved_conservative_cuts",block_id=block["block_id"],
            full_block_matched=True,cuts=bc,annotation=str(HISTORY/"data/inputs_full.jsonl")))
    altered["context"]=context
    assert altered["condition"]==r["condition"] and altered["ancestors"]==r["ancestors"]
    # Existing restoration text remains; only cuts from matching saved full blocks are deleted.
    for restore in r["changes"].get("necessary_restores",[]):
        quote=restore["quote"]
        if quote in r["context"]: assert quote in altered["context"]
    return make_prompt(altered),operations

def prepare(root):
    torch.set_num_threads(4)
    hard=torch.load(SOURCE/"operators/aster.pt",map_location="cpu",weights_only=False)
    operators={};checks=[]
    for h,op in hard.items():
        u=op["U_N"].float().clone()
        operators[h]=dict(N=u,B=u.T.contiguous(),mu=op["mu"].clone(),S=op["S"].clone(),amplification=1.)
        d=u@u.T
        norm=float(torch.linalg.matrix_norm(d.double(),ord=2))
        checks.append(dict(head=int(h),directions=u.shape[1],projection_norm=norm,
            basis_equals_saved=torch.equal(u,op["U_N"]),transpose_exact=torch.equal(operators[h]["B"],u.T),
            signal_residual=float(torch.linalg.matrix_norm(d@op["S"])),
            idempotence_error=float((d@d-d).abs().max())))
        assert norm<=1+1e-5
    assert len(operators)==51 and sum(x["N"].shape[1] for x in operators.values())==244
    torch.save(operators,root/"operators/unscreened.pt")
    save_table(root,"operator_checks",checks)
    views=rows(SOURCE/"data/views.jsonl");pairs=rows(SOURCE/"data/pairs.jsonl")
    constructions={r["group_id"]:r for r in rows(ORIGINAL/"data/constructions.jsonl")}
    suffixes={}
    for p in pairs:
        if p["kind"]=="P":
            for side,suffix in zip(("left","right"),p["edit_locations"]):
                assert p[side] not in suffixes or suffixes[p[side]]==suffix
                suffixes[p[side]]=suffix
    system,make_prompt=historical_template()
    references=[];lookup={};mapping=[]
    def register(dataset,identifier,prompt,reference,system,metadata,operations):
        key=(system,reference)
        if key not in lookup:
            ref_id=f"ref_{len(references):05d}"
            lookup[key]=ref_id
            references.append(dict(reference_id=ref_id,system=system,prompt=reference,template=dataset))
        mapping.append(dict(dataset=dataset,input_id=identifier,target_prompt=prompt,
            reference_id=lookup[key],reference_changed=prompt!=reference,operations=operations,**metadata))
    for v in views:
        ref,ops=main_reference(v,suffixes.get(v["view_id"]),constructions.get(v["group_id"],{}))
        register("main",v["view_id"],v["prompt"],ref,MAIN_SYSTEM,
            {**{k:v[k] for k in ("source_id","group_id","split","variant","condition","original")},
             "gold":v["label"]},ops)
    inputs=rows(SOURCE/"diagnostics/fact_preference_inputs.jsonl")+rows(SOURCE/"diagnostics/deletion_restore_inputs.jsonl")
    from .existing_report import threshold_value
    for r in inputs:
        ref,ops=historical_reference(r,make_prompt)
        register("historical",r["score_id"],make_prompt(r),ref,system,
            {**{k:r.get(k) for k in ("source_id","node_id","example_id","split","view","population",
                    "condition","metric_level","metric_value","background")},
             "gold":r["gold"],"exact_numeric_threshold":threshold_value(r) if r["population"]=="constructed" else False},ops)
    byid={r["input_id"]:r for r in mapping}
    coverage=[]
    for dataset in ("main","historical"):
        current=[r for r in mapping if r["dataset"]==dataset]
        for split in ("all",)+(("fit","dev","test") if dataset=="main" else ("test",)):
            rs=[r for r in current if split=="all" or r["split"]==split]
            coverage.append(dict(dataset=dataset,split=split,inputs=len(rs),
                reference_changed=sum(r["reference_changed"] for r in rs),
                reference_unchanged=sum(not r["reference_changed"] for r in rs),
                unique_references=len({r["reference_id"] for r in rs})))
    pair_coverage=[]
    for p in pairs:
        a,b=byid[p["left"]],byid[p["right"]]
        pair_coverage.append(dict(**p,common_reference=a["reference_id"]==b["reference_id"],
            left_reference=a["reference_id"],right_reference=b["reference_id"],
            left_changed=a["reference_changed"],right_changed=b["reference_changed"]))
    main_cov=coverage[0]; hist_cov=next(x for x in coverage if x["dataset"]=="historical" and x["split"]=="all")
    expected=dict(main_inputs=1290,main_changed=954,main_unique_references=592,historical_inputs=569,
        historical_changed=515,A_common=232,A_total=396,P_common=231,P_total=231)
    observed=dict(main_inputs=main_cov["inputs"],main_changed=main_cov["reference_changed"],
        main_unique_references=main_cov["unique_references"],historical_inputs=hist_cov["inputs"],
        historical_changed=hist_cov["reference_changed"],
        **{kind+"_"+suffix:sum(p["kind"]==kind and (p["common_reference"] if suffix=="common" else True) for p in pair_coverage)
           for kind in ("A","P") for suffix in ("common","total")})
    assert observed==expected,(observed,expected)
    # Assign each distinct reference to one worker, so the natural activation is computed once.
    groups=defaultdict(list)
    for r in mapping:groups[r["reference_id"]].append(r)
    loads=[0,0]
    for ref in references:
        cost=len(ref["prompt"])+sum(len(r["target_prompt"]) for r in groups[ref["reference_id"]])
        shard=min(range(2),key=lambda j:loads[j]);loads[shard]+=cost
        ref["shard"]=shard
        for r in groups[ref["reference_id"]]:r["shard"]=shard
    write_rows(root/"inputs/reference_mapping.jsonl",mapping)
    write_rows(root/"references/prompts.jsonl",references)
    save_table(root,"reference_coverage",coverage)
    save_table(root,"pair_reference_coverage",pair_coverage)
    write_json(root/"results/preparation_verification.json",dict(expected=expected,observed=observed,
        conditions_definitions_facts_preserved=True,no_labels_used_to_choose_references=True,
        projection_shared_by_both_methods=True,projection_norm_max=max(x["projection_norm"] for x in checks),
        reference_key="exact system and user strings, without hashes",
        no_signal_annihilation_constraint=True,reference_counts=Counter(r["template"] for r in references),
        worker_inputs=Counter(r["shard"] for r in mapping)))
    config=dict(model=engine_module.DEFAULT_MODEL,seed=SEED,gpus=[0,1],precision="bfloat16",
        heads=sorted(operators),head_count=51,direction_count=244,rank_cap_per_type=4,energy_target=.95,
        edit_strength=1.,operator="D=U_N U_N^T; N=U_N; B=U_N^T",basis_source=str(SOURCE/"operators/aster.pt"),
        methods={"full":"saved unedited score","global_mean":"z-D(z-mu_saved)",
            "same_fact":"z-D(z-z_ref_unedited_frozen)","direct_reference":"unedited reference prompt score"},
        signal_constraint=None,parameter_search=False,refitting=False,new_annotations=0,
        sources=[str(SOURCE),str(SOFT),str(HISTORY)],source_splits="unchanged",
        scale=dict(main_inputs=1290,main_pairs=1053,historical_inputs=569,unique_references=len(references)),
        reference_activation="Computed once per exact template/prompt and frozen; targets edited online layer by layer",
        comparisons_reused=["hard-screened ASTER","soft-angle","main calibrated_full"],
        mechanisms="Reuse nuisance_orthogonal bidirectional joint repairs; independent of reference vector",
        scoring="Original per-dataset system and user templates; ell=logit(\u662f)-logit(\u5426); prediction=ell>0",
        bootstrap=dict(unit="source",resamples=2000,seed=SEED),
        path_experiments=False,unmodified_reference_policy="Separate identity inputs; never protection evidence")
    write_json(root/"configuration.json",config)
    write_json(root/"resource_plan.json",dict(gpus=[0,1],copies=2,expected_gib_per_gpu=[18,25],
        expected_minutes=[15,25],scale=config["scale"],monitoring="first 20 inputs then completion only"))
    print(json.dumps(observed,ensure_ascii=False))

def anchored(operators,heads):
    return {h:dict(op,mu=heads[h]) for h,op in operators.items()}

def verify_hooks(engine,context,operators,root,shard,natural):
    zero={h:dict(op,N=op["N"][:,:0],B=op["B"][:0]) for h,op in operators.items()}
    result=engine.run(context,operators=zero,capture=False)
    assert result["ell"]==natural
    captured={};handles=[]
    for layer in sorted({h//engine.n_heads for h in operators}):
        def capture(mod,args,out,layer=layer):
            captured[layer]=args[0].detach().float().cpu()
        handles.append(engine.layers[layer].self_attn.o_proj.register_forward_hook(capture))
    try:edited=engine.run(context,operators=operators,capture=True)
    finally:
        for h in handles:h.remove()
    errors=[];untargeted=[]
    for layer,value in captured.items():
        assert value.shape[1]==1
        actual=value.reshape(engine.n_heads,engine.head_dim);raw=edited["heads"][layer]
        for j in range(engine.n_heads):
            h=layer*engine.n_heads+j
            if h in operators:
                op=operators[h]
                expected=(raw[j]-(raw[j]-op["mu"])@op["B"].T@op["N"].T).to(engine.model.dtype).float()
                errors.append(float((expected-actual[j]).abs().max()))
            else:untargeted.append(torch.equal(raw[j],actual[j]))
    assert max(errors)==0 and all(untargeted)
    write_json(root/"results"/f"engine_verification_{shard}.json",dict(zero_score_equal=True,
        all_selected_layers_checked=sorted(captured),selected_heads=sorted(operators),
        final_position_only=True,non_target_heads_unchanged=True,max_factor_error=max(errors)))

def worker(root,shard):
    torch.set_num_threads(4);torch.manual_seed(SEED)
    status(root,f"worker_{shard}",state="loading",gpu=shard)
    started=time.time();engine=engine_module.Engine()
    operators=torch.load(root/"operators/unscreened.pt",map_location="cpu",weights_only=False)
    refs=[r for r in rows(root/"references/prompts.jsonl") if r["shard"]==shard]
    inputs=[r for r in rows(root/"inputs/reference_mapping.jsonl") if r["shard"]==shard]
    groups=defaultdict(list)
    for r in inputs:groups[r["reference_id"]].append(r)
    output=root/"scores"/f"scores_{shard}.jsonl"
    complete={r["input_id"] for r in rows(output)}
    from .pipeline import base_scores
    full_main=base_scores(SOURCE)["full"]
    full_history={r["score_id"]:r["ell"] for f in sorted((SOURCE/"diagnostics").glob("scores_*.jsonl"))
                  for r in rows(f) if r["method"]=="natural"}
    checked_templates=set();done=len(complete);timing_start=time.time();first20=False
    for ref in refs:
        group=groups[ref["reference_id"]]
        if all(r["input_id"] in complete for r in group):continue
        engine_module.SYSTEM=ref["system"]
        path=root/"cache"/(ref["reference_id"]+".pt")
        if path.exists():cached=torch.load(path,map_location="cpu",weights_only=False)
        else:
            context=engine.prepare(ref["prompt"])
            result=engine.run(context,capture=True)
            heads={h:result["heads"].reshape(-1,engine.head_dim)[h].clone() for h in operators}
            ref_ops=anchored(operators,heads)
            # Check identity on one reference per distinct historical route and main template.
            templates={(r["dataset"],r.get("view","main")) for r in group}
            unseen=templates-checked_templates
            identity=None
            if unseen:
                identity=engine.run(context,operators=ref_ops,capture=False)["ell"]
                assert identity==result["ell"]
            cached=dict(reference_id=ref["reference_id"],ell=result["ell"],prediction=result["prediction"],
                tokens=result["tokens"],heads=heads,identity_check=identity,template=ref["template"])
            torch.save(cached,path)
            append(root/"references"/f"scores_{shard}.jsonl",{k:v for k,v in cached.items() if k!="heads"})
            del context,result
        ref_ops=anchored(operators,cached["heads"])
        for r in group:
            if r["input_id"] in complete:continue
            context=engine.prepare(r["target_prompt"])
            template=(r["dataset"],r.get("view","main"))
            if template not in checked_templates:
                natural=engine.run(context,capture=False)["ell"]
                expected=(full_main if r["dataset"]=="main" else full_history)[r["input_id"]]
                assert natural==expected,(r["input_id"],natural,expected)
                if not checked_templates:verify_hooks(engine,context,operators,root,shard,natural)
                append(root/"results"/f"template_verification_{shard}.jsonl",dict(dataset=r["dataset"],
                    view=r.get("view","main"),input_id=r["input_id"],natural=natural,saved=expected,exact=True))
                checked_templates.add(template)
            global_score=engine.run(context,operators=operators,capture=False)["ell"]
            same_score=engine.run(context,operators=ref_ops,capture=False)["ell"]
            if not r["reference_changed"]:
                baseline=(full_main if r["dataset"]=="main" else full_history)[r["input_id"]]
                assert same_score==cached["ell"]==baseline,(r["input_id"],same_score,cached["ell"],baseline)
            append(output,dict(input_id=r["input_id"],dataset=r["dataset"],reference_id=r["reference_id"],
                status="complete",global_mean=global_score,same_fact=same_score,direct_reference=cached["ell"],
                tokens=context.length,reference_changed=r["reference_changed"]))
            del context
            complete.add(r["input_id"]);done+=1
            if done>=20 and not first20:
                elapsed=time.time()-timing_start
                write_json(root/"timing"/f"first20_{shard}.json",dict(inputs=done,seconds=elapsed,
                    seconds_per_input=elapsed/done,projected_minutes=len(inputs)*elapsed/done/60,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gpu=shard,total_inputs=len(inputs)))
                first20=True
        status(root,f"worker_{shard}",state="running",done=done,total=len(inputs),
            elapsed_seconds=time.time()-started)
    write_json(root/"timing"/f"worker_{shard}.json",dict(inputs=done,references=len(refs),
        total_seconds=time.time()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        counts=engine.counts))
    status(root,f"worker_{shard}",state="complete",done=done,total=len(inputs))
    assert done==len(inputs)

def launch(root):
    started=time.time();procs=[]
    for shard in (0,1):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
        with (root/"logs"/f"worker_{shard}.log").open("a") as log:
            procs.append(subprocess.Popen([sys.executable,"-m","experiments.aster_closed_form.reference_anchor",
                "--root",str(root),"--stage","worker","--shard",str(shard)],
                cwd=root/"code",env=env,stdout=log,stderr=log))
    status(root,"all",state="running",worker_pids=[p.pid for p in procs],gpus=[0,1])
    codes=[p.wait() for p in procs]
    write_json(root/"runtime.json",dict(seconds=time.time()-started,gpus=[0,1],exit_codes=codes))
    if any(codes):raise RuntimeError("Worker failed; see server log")
    status(root,"all",state="scoring_complete")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--stage",choices=("prepare","all","worker"),default="all")
    p.add_argument("--shard",type=int,default=0)
    a=p.parse_args()
    try:
        if a.stage=="prepare":prepare(a.root)
        elif a.stage=="all":launch(a.root)
        else:worker(a.root,a.shard)
    except Exception as exc:
        status(a.root,f"worker_{a.shard}" if a.stage=="worker" else "all",state="failed",error=str(exc))
        traceback.print_exc();raise
if __name__=="__main__":main()

