"""Resumable pilot stages. Patient text and per-case outputs stay on the server."""
import gc, json, random, shutil, time, traceback, os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .core import (SEED, write_json, rows, append, journal, nodes, swap_actions, deployment,
                   screen_cases, group_indices, separation, rank_heads, make_module,
                   parameter_count, prediction_summary, selection_key, paired, objective)
from .engine import Engine, fork_cache

SOURCE = Path("/home/gcx/aster_representation_20261101")

def prepare(root):
    root = Path(root); data = root / "data"; data.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train", "age_train", "development", "confirmation"):
        target = data / (name + ".jsonl")
        if not target.exists(): shutil.copyfile(SOURCE / target.name, target)
        counts[name] = len(rows(target))
    write_json(root / "data_manifest.json", dict(source="nchennnn/MEGA-CDP train_val",
               original_directory=str(SOURCE), counts=counts, seed=SEED,
               age_note="16 derived training states; no independent age test",
               screen_cases=[c["case_id"] for c in screen_cases(rows(data/"development.jsonl"))]))
    return counts

def architecture(engine):
    c = engine.config
    return dict(layers=c.num_hidden_layers, query_heads=c.num_attention_heads,
                kv_heads=c.num_key_value_heads, head_dim=c.head_dim, hidden_size=c.hidden_size,
                layer_types=list(c.layer_types), sliding_window=c.sliding_window,
                device_map={k: str(v) for k,v in engine.model.hf_device_map.items()})

def cleanup(engine):
    engine.clear_reference_memory(); gc.collect()
    for device in (0, 1):
        with torch.cuda.device(device): torch.cuda.empty_cache()

def smoke(engine, root):
    started = time.time(); case = rows(Path(root)/"data/development.jsonl")[0]
    reference = engine.reference(case, "G0")
    context = engine.context(deployment(case))
    query = engine.decision_query(context, [], "G0")
    with torch.no_grad(): baseline, _ = engine.scores(query)
    layer = engine.config.num_hidden_layers // 2
    result = dict(started=started, architecture=architecture(engine))
    from .core import ConditionalIntervention
    whole = make_module(dict(kind="whole", layer=layer), engine.config, engine.layer_device(layer))
    head = make_module(dict(kind="heads", layer=layer, groups=[0,1,2,3]), engine.config, engine.layer_device(layer))
    for name, module in (("whole", whole), ("heads", head)):
        with torch.no_grad():
            zero, _ = engine.scores(query, layer=layer, module=module, reference=reference, force_zero=True)
        if not torch.equal(baseline, zero): raise AssertionError("Zero edit changes native decision: " + name)
        engine.model.zero_grad(set_to_none=True); module.zero_grad(set_to_none=True)
        edited, diag = engine.scores(query, layer=layer, module=module, reference=reference)
        loss = -edited.log_softmax(-1)[0] + .001 * diag["magnitude"].to(edited.device); loss.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters()):
            raise AssertionError("No finite intervention gradient")
        if any(p.grad is not None for p in engine.model.parameters()): raise AssertionError("Backbone gradient")
        result[name] = dict(zero_identical=True, gradient=True, parameters=parameter_count(module),
                            magnitude=float(diag["magnitude"].detach()))
    # Compare native full and chunked prefill beyond a sliding window.
    ids = engine.ids(context.rendered)[:engine.config.sliding_window + 17]
    with torch.no_grad():
        native = engine.forward(ids).logits.detach().float().cpu()
        cache = None; last = None
        for start in range(0, len(ids), 256):
            last = engine.forward(ids[start:start+256], cache, start); cache=last.past_key_values
        chunked = last.logits.detach().float().cpu()
    result["chunked_native_max_logit_difference"] = float((native - chunked).abs().max())
    result["chunked_native_argmax_equal"] = bool(torch.equal(native.argmax(-1), chunked.argmax(-1)))
    result["chunked_native_allclose"] = bool(torch.allclose(native, chunked, atol=.05, rtol=.01))
    result["prefill_policy"] = "native whole prefill; chunk comparison is diagnostic only"
    if not torch.isfinite(native).all() or not torch.isfinite(chunked).all():
        raise AssertionError("Nonfinite native prefill logits")
    # Fork must preserve both sliding-layer cumulative positions and global keys.
    old_lengths = [l.get_seq_length() for l in query.cache.layers]
    with torch.no_grad(): engine.scores(query)
    if old_lengths != [l.get_seq_length() for l in query.cache.layers]: raise AssertionError("Persistent cache mutated")
    result.update(seconds=time.time()-started, peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)],
                  peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in (0,1)], complete=True)
    write_json(Path(root)/"smoke.json", result)
    cleanup(engine)
    return result

def collect(engine, root):
    root = Path(root); target=root/"training_states"; target.mkdir(exist_ok=True)
    cases=rows(root/"data/train.jsonl")+rows(root/"data/age_train.jsonl")
    c=engine.config; dims=c.num_attention_heads//c.num_key_value_heads*c.head_dim
    sums=torch.zeros((2,c.num_hidden_layers,c.num_key_value_heads,dims),dtype=torch.float64)
    squares=torch.zeros_like(sums); counts=torch.zeros(2,dtype=torch.float64)
    started=time.time(); total=0
    for index,case in enumerate(cases):
        path=target/(case["case_id"]+".pt")
        if path.exists(): records=torch.load(path,map_location="cpu",weights_only=True)
        else:
            records=[]; context=engine.context(deployment(case))
            for node in nodes(case):
                query=engine.decision_query(context,node["history"],node["gid"])
                captured={}
                with torch.no_grad(): scores,_=engine.scores(query,capture=captured)
                reference=engine.reference(case,node["gid"])
                records.append(dict(node=node,full_pre=torch.stack([captured["pre"][l] for l in range(c.num_hidden_layers)]),
                    full_post=torch.stack([captured["post"][l] for l in range(c.num_hidden_layers)]),
                    blind_pre=torch.stack([reference["pre"][l] for l in range(c.num_hidden_layers)]),
                    blind_post=torch.stack([reference["post"][l] for l in range(c.num_hidden_layers)]),
                    baseline_scores=scores.detach().cpu()))
            temporary=path.with_suffix(".partial");torch.save(records,temporary);temporary.replace(path)
            del context,query;cleanup(engine)
        for record in records:
            label=0 if record["node"]["target"].endswith("Y") else 1
            x=record["full_pre"].reshape(c.num_hidden_layers,c.num_key_value_heads,dims).double()
            sums[label]+=x;squares[label]+=x.square();counts[label]+=1;total+=1
        write_json(root/"status.json",dict(stage="collection",cases=index+1,total_cases=len(cases),nodes=total))
    stats=dict(separation=separation(sums,squares,counts),class_counts=counts.tolist(),nodes=total,
               seconds=time.time()-started,definition="L2 distance between Y/N means divided by pooled within-class L2 std")
    write_json(root/"representation_statistics.json",stats);return stats

def scan(engine, root):
    root=Path(root);stats=json.loads((root/"representation_statistics.json").read_text())
    chosen=screen_cases(rows(root/"data/development.jsonl"))
    output=root/"head_scan.jsonl";saved=journal(output,lambda r:(r["case_id"],r["layer"],r["group"]))
    c=engine.config;started=time.time()
    for case_index,case in enumerate(chosen):
        needed=[(l,g) for l in range(c.num_hidden_layers) for g in range(c.num_key_value_heads)
                if (case["case_id"],l,g) not in saved]
        if not needed:continue
        context=engine.context(deployment(case));query=engine.decision_query(context,[],"G0")
        reference=engine.reference(case,"G0")
        with torch.no_grad():baseline,_=engine.scores(query)
        target=0 if case["gold_rules"][0].endswith("Y") else 1
        base_margin=float((baseline[target]-baseline[1-target]).detach())
        for layer,group in needed:
            start=time.time();entry=dict(case_id=case["case_id"],source_id=case["source_id"],layer=layer,group=group,base_margin=base_margin)
            try:
                with torch.no_grad():
                    plus,_=engine.scores(query,layer=layer,reference=reference,indices=group_indices([group],c),strength=.25)
                    minus,_=engine.scores(query,layer=layer,reference=reference,indices=group_indices([group],c),strength=-.25)
                entry.update(status="complete",plus_margin=float((plus[target]-plus[1-target]).detach()),
                             minus_margin=float((minus[target]-minus[1-target]).detach()))
            except Exception as exc:
                entry.update(status="failed",error=type(exc).__name__+": "+str(exc)[:500])
                torch.cuda.empty_cache()
            entry["seconds"]=time.time()-start;append(output,entry);saved[(case["case_id"],layer,group)]=entry
        write_json(root/"status.json",dict(stage="scan",cases=case_index+1,total_cases=len(chosen),cells=len(saved)))
        del context,query,reference;cleanup(engine)
    selected=rank_heads(list(saved.values()),stats["separation"],list(c.layer_types),c.num_key_value_heads)
    selected.update(screen_cases=len(chosen),groups=c.num_hidden_layers*c.num_key_value_heads,
                    cells=len(saved),failed_cells=sum(r["status"]!="complete" for r in saved.values()),seconds=time.time()-started)
    write_json(root/"head_selection.json",selected);return selected

def predict(engine, cases, path, spec=None, module=None, mode="zero"):
    saved=journal(path,lambda r:(r["case_id"],r["view"]))
    if module is not None:module.eval()
    for case in cases:
        for view in ("original","swapped"):
            key=(case["case_id"],view)
            if key in saved:continue
            row=dict(case_id=case["case_id"],source_id=case["source_id"],view=view)
            try:
                actual=swap_actions(case) if view=="swapped" else case
                with torch.no_grad():result=engine.generate(deployment(actual),spec,module,mode)
                row.update(result);row["status"]="failed" if result["output_limit"] else "complete"
            except Exception as exc:
                row.update(status="failed",path=[],leaf=None,magnitudes=[],error=type(exc).__name__+": "+str(exc)[:500])
            append(path,row);saved[key]=row
            cleanup(engine)
    return prediction_summary(cases,list(saved.values()))

def save_progress(path, module, optimizer, epoch, next_index):
    tmp=path.with_suffix(".partial")
    torch.save(dict(module=module.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,next_index=next_index,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(path)

def train_job(engine,root,spec):
    root=Path(root);job=root/"jobs"/spec["name"];job.mkdir(parents=True,exist_ok=True)
    spec_path=job/"spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:raise ValueError("Existing job has a different specification")
    write_json(spec_path,spec)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    module=make_module(spec,engine.config,engine.layer_device(spec["layer"]))
    optimizer=torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.01)
    cases=rows(root/"data/train.jsonl")+rows(root/"data/age_train.jsonl")
    development=rows(root/"data/development.jsonl")
    progress=job/"progress.pt";start_epoch,next_index=1,0
    if progress.exists():
        old=torch.load(progress,map_location="cpu",weights_only=True)
        module.load_state_dict(old["module"]);optimizer.load_state_dict(old["optimizer"])
        start_epoch,next_index=old["epoch"],old["next_index"]
        torch.set_rng_state(old["cpu_rng"]);torch.cuda.set_rng_state_all(old["cuda_rng"])
    started=time.time()
    for epoch in (1,2,3):
        weights=job/f"epoch{epoch}.safetensors"
        if not weights.exists():
            if epoch<start_epoch:raise ValueError("Missing completed checkpoint")
            module.train();order=list(cases);random.Random(SEED+epoch).shuffle(order)
            begin=next_index if epoch==start_epoch else 0
            if progress.exists() and epoch==start_epoch:
                module.load_state_dict(old["module"]);optimizer.load_state_dict(old["optimizer"])
                torch.set_rng_state(old["cpu_rng"]);torch.cuda.set_rng_state_all(old["cuda_rng"])
            optimizer.zero_grad(set_to_none=True)
            for index in range(begin,len(order)):
                case=order[index];sample_nodes=nodes(case)
                contexts=[engine.context(deployment(v)) for v in (case,swap_actions(case))]
                for node in sample_nodes:
                    reference=engine.reference(case,node["gid"]);logits=[];bases=[];mags=[]
                    for context in contexts:
                        query=engine.decision_query(context,node["history"],node["gid"])
                        with torch.no_grad():base,_=engine.scores(query)
                        edited,diag=engine.scores(query,layer=spec["layer"],module=module,reference=reference)
                        logits.append(edited);bases.append(base);mags.append(diag["magnitude"].to(edited.device))
                    target=torch.tensor(0 if node["target"].endswith("Y") else 1,device=logits[0].device)
                    loss,terms=objective(logits,bases,target,mags,use_js=False)
                    (loss/len(sample_nodes)/8).backward()
                    del logits,bases,mags,loss,edited,diag
                del contexts,reference,query;engine.clear_reference_memory()
                if (index+1)%8==0 or index+1==len(order):
                    remainder=(index+1)%8
                    if index+1==len(order) and remainder:
                        for p in module.parameters():
                            if p.grad is not None:p.grad.mul_(8/remainder)
                    torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                    save_progress(progress,module,optimizer,epoch,index+1)
                    append(job/"training_log.jsonl",dict(epoch=epoch,completed_cases=index+1,terms=terms,elapsed=time.time()-started))
                    write_json(root/"status.json",dict(stage="training",job=spec["name"],epoch=epoch,cases=index+1,total_cases=len(order)))
            save_file({k:v.detach().cpu().contiguous() for k,v in module.state_dict().items()},str(weights))
            save_progress(progress,module,optimizer,epoch+1,0)
        module.load_state_dict(load_file(str(weights)))
        summary=predict(engine,development,job/f"development_epoch{epoch}.jsonl",spec,module,mode="edited")
        summary.update(layer=spec["layer"],epoch=epoch,kind=spec["kind"],groups=spec.get("groups"),
                       checkpoint=str(weights),parameters=parameter_count(module),name=spec["name"])
        write_json(job/f"development_epoch{epoch}.json",summary)
    write_json(job/"training_complete.json",dict(seconds=time.time()-started,parameters=parameter_count(module)))
    del module,optimizer;cleanup(engine)
    return [json.loads((job/f"development_epoch{e}.json").read_text()) for e in (1,2,3)]

def choose(results):
    return max(results,key=lambda r:selection_key(r,r["layer"],r["epoch"]))

def train_candidates(engine,root):
    root=Path(root);selection=json.loads((root/"head_selection.json").read_text());whole=[];heads=[]
    for entry in selection["candidates"]:
        layer=entry["layer"]
        for kind,destination in (("whole",whole),("heads",heads)):
            spec=dict(name=f"{kind}_layer{layer}",kind=kind,layer=layer)
            if kind=="heads":spec["groups"]=entry["groups"]
            destination.extend(train_job(engine,root,spec))
    best_whole,best_heads=choose(whole),choose(heads)
    control_groups=[i for i in range(engine.config.num_key_value_heads) if i not in best_heads["groups"]][:4]
    control_spec=dict(name=f"control_layer{best_heads['layer']}",kind="heads",layer=best_heads["layer"],groups=control_groups)
    best_control=choose(train_job(engine,root,control_spec))
    frozen=dict(whole=best_whole,heads=best_heads,control=best_control,seed=SEED,rank=4,
                selection="development original PathAcc, both views correct, lower post-projection edit, lower layer, earlier epoch",
                frozen_before_confirmation=time.time(),all_whole=whole,all_heads=heads)
    write_json(root/"frozen_selection.json",frozen);return frozen

def confirm(engine,root):
    root=Path(root);frozen=json.loads((root/"frozen_selection.json").read_text())
    cases=rows(root/"data/confirmation.jsonl");output=root/"confirmation";output.mkdir(exist_ok=True)
    summaries={}
    for name,mode in (("zero","zero"),("blind","blind")):
        summaries[name]=predict(engine,cases,output/(name+".jsonl"),mode=mode)
        write_json(root/"status.json",dict(stage="confirmation",completed=name))
    for name in ("whole","heads","control"):
        chosen=frozen[name];spec=dict(name=chosen["name"],kind=chosen["kind"],layer=chosen["layer"])
        if chosen.get("groups") is not None:spec["groups"]=chosen["groups"]
        module=make_module(spec,engine.config,engine.layer_device(spec["layer"]))
        module.load_state_dict(load_file(chosen["checkpoint"]))
        summaries[name]=predict(engine,cases,output/(name+".jsonl"),spec,module,mode="edited")
        del module;cleanup(engine)
        write_json(root/"status.json",dict(stage="confirmation",completed=name))
    return report(root)

def report(root):
    root=Path(root);cases=rows(root/"data/confirmation.jsonl");methods={}
    for name in ("zero","blind","whole","heads","control"):
        path=root/"confirmation"/(name+".jsonl")
        methods[name]=list(journal(path,lambda r:(r["case_id"],r["view"])).values())
    summaries={name:prediction_summary(cases,values) for name,values in methods.items()}
    pair_tables={base:paired(cases,methods,base) for base in ("zero","whole","control")}
    frozen=json.loads((root/"frozen_selection.json").read_text())
    final=dict(complete=all(len(v)==len(cases)*2 for v in methods.values()),summaries=summaries,
               paired=pair_tables,selection={k:frozen[k] for k in ("whole","heads","control")},
               age_scope="training diagnostic only",official_test_run=False)
    write_json(root/"results.json",final)
    lines=["# Gemma 3 27B 选层选头先导结果","",
           "数据：200例训练、16个派生年龄状态、100例开发、100例确认。年龄状态属于训练，不是独立确认。",
           "","| 方法 | 原指南完整路径正确/100 | 两视图均正确/100 | 纠正 | 退化 | 净增加 | 失败单元/200 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name,s in summaries.items():
        p=pair_tables["zero"][name]
        lines.append(f"| {name} | {s['original_correct']} | {s['both_views_correct']} | {p['corrections']} | {p['regressions']} | {p['net']} | {s['failures']} |")
    lines+=["","## 冻结配置"]
    for name in ("whole","heads","control"):
        s=frozen[name];lines.append(f"- {name}: 层{s['layer']}，组{s.get('groups')}，第{s['epoch']}轮，参数{s['parameters']}。")
    lines+=["","选头相对整层和未选头对照的配对计数见 results.json。失败保留在全部100例的分母内。",
            "本轮未运行官方7,458例。局部/全局窗口均为模型原生设置；当前结果只支持本轮先导结论。"]
    (root/"REPORT.zh.md").write_text("\n".join(lines)+"\n")
    write_json(root/"status.json",dict(stage="complete" if final["complete"] else "incomplete",results=str(root/"results.json")))
    return final

def run(root,model_path,stage="all",chunk=0):
    root=Path(root);prepare(root)
    if stage=="report":return report(root)
    started=time.time()
    engine=Engine.load(model_path,root,chunk)
    write_json(root/"architecture.json",architecture(engine))
    for device in (0,1):torch.cuda.reset_peak_memory_stats(device)
    if stage=="smoke":return smoke(engine,root)
    if stage in ("all","collect"):
        if not (root/"representation_statistics.json").exists():collect(engine,root)
        if stage=="collect":return
    if stage in ("all","scan"):
        if not (root/"head_selection.json").exists():scan(engine,root)
        if stage=="scan":return
    if stage in ("all","train"):
        if not (root/"frozen_selection.json").exists():train_candidates(engine,root)
        if stage=="train":return
    if stage in ("all","confirm"):confirm(engine,root)
    write_json(root/"costs.json",dict(seconds=time.time()-started,peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)],
                                     peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in (0,1)]))
