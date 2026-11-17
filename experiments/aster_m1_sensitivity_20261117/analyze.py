"""Source-cluster inference and complete re-selection on fixed training subsets."""
from __future__ import annotations
import argparse,gzip,json,sqlite3,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from .prepare import CODE,OUT,RUN,PRIVATE
from .audit import read,lines,table
from .statistics import KINDS,scenarios,candidates,source_statistics,select,sets,jaccard,bh
from experiments.aster_cr_retrain_paired_v2_20261111.common import write

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def aggregate():
    path=OUT/"matched_source_effects.npz"
    if path.exists():return dict(np.load(path,allow_pickle=False))
    plan=read(OUT/"plan.json")
    for w in range(4):
        p=OUT/"workers"/str(w)/"completion.json"
        if not p.exists() or not read(p).get("complete"):
            raise RuntimeError("Missing completed matched-control worker "+str(w))
    conn=sqlite3.connect("file:"+str(RUN/"existing_scores.sqlite")+"?mode=ro",uri=True)
    for w in range(4):
        conn.execute("ATTACH DATABASE ? AS w"+str(w),("file:"+str(RUN/("new_scores_"+str(w)+".sqlite"))+"?mode=ro",))
    query=" UNION ALL ".join("SELECT value FROM "+db+".scores WHERE target=? AND donor=? AND head=?"
                             for db in ["main","w0","w1","w2","w3"])+" LIMIT 1"
    @lru_cache(maxsize=100000)
    def score(a,b,h):
        r=conn.execute(query,(a,b,int(h))*5).fetchone()
        if r is None:raise ValueError("Missing intervention; no zero imputation")
        return r[0]
    cache={r["uid"]:r for r in lines(PRIVATE/"cache/scores_fit.jsonl")}
    sources=plan["confirmation_sources"];lookup={s:i for i,s in enumerate(sources)}
    shape=(4,len(sources),1152)
    sums=np.zeros(shape);true=np.zeros(shape);null=np.zeros(shape);counts=np.zeros(shape,int)
    tasks=read(OUT/"matched_donor_plan.json")
    with gzip.open(OUT/"matched_pair_effects.jsonl.gz","wt",encoding="utf-8") as f:
        for task in tasks:
            if not task["eligible"]:continue
            p=task["target"];k=KINDS.index(p["kind"]);s=lookup[p["source_id"]]
            a,b=p["left"],p["right"];direction=p["right_label"]-p["left_label"]
            for h in task["heads"]:
                sl,sr=score(a,a,h),score(b,b,h)
                bl,br=cache[a]["ell"],cache[b]["ell"]
                floor=.5*(abs(sl-bl)+abs(sr-br)+2e-4+1e-5*(abs(bl)+abs(br)))
                c=.5*((score(a,b,h)-sl)-(score(b,a,h)-sr))
                actual=max(abs(c)-floor,0.) if k<2 else direction*c
                controls=[]
                for d in task["donors"]:
                    z=.5*((score(a,d["right"],h)-sl)-(score(b,d["left"],h)-sr))
                    controls.append(max(abs(z)-floor,0.) if k<2 else direction*z)
                control=float(np.mean(controls));contrast=actual-control
                sums[k,s,h]+=contrast;true[k,s,h]+=actual;null[k,s,h]+=control;counts[k,s,h]+=1
                f.write(json.dumps(dict(kind=p["kind"],source_id=p["source_id"],pair_id=p["pair_id"],
                    head=h,true_effect=actual,null_effect=control,contrast=contrast,
                    control_effects=controls,donors=len(controls)),allow_nan=False)+"\n")
    result=dict(contrast=np.divide(sums,counts,out=np.full(shape,np.nan),where=counts>0),
                true=np.divide(true,counts,out=np.full(shape,np.nan),where=counts>0),
                null=np.divide(null,counts,out=np.full(shape,np.nan),where=counts>0),
                pair_counts=counts,sources=np.array(sources))
    np.savez_compressed(path,**result)
    table(OUT/"matched_source_effects.csv.gz",
          (dict(kind=k,source_id=s,head=h,pairs=int(counts[i,j,h]),
                true_effect=float(result["true"][i,j,h]),null_effect=float(result["null"][i,j,h]),
                contrast=float(result["contrast"][i,j,h]))
           for i,k in enumerate(KINDS) for j,s in enumerate(sources)
           for h in np.flatnonzero(counts[i,j])),True)
    conn.close()
    return result

def stats_for(effect,cfg,seed,source_ix):
    x=effect["contrast"][:,source_ix,:]
    shape=(4,1152)
    matrix=x.transpose(1,0,2).reshape(len(source_ix),-1)
    available=np.isfinite(matrix).any(0)
    fields=["mean","p","ci_low","ci_high","direction_consistency","valid_sources",
            "positive_sources","negative_sources","source_direction_consistency","sign_test_p"]
    result={k:np.full(4608,np.nan) for k in fields}
    result["p"].fill(1);result["sign_test_p"].fill(1);result["valid_sources"].fill(0)
    ss=source_statistics(matrix[:,available],np.repeat([False,False,True,True],1152)[available],cfg,seed)
    for key,v in ss.items():result[key][available]=v
    return {k:v.reshape(shape) for k,v in result.items()}

def by_selected(mask,stats,settings):
    chosen,q=select(mask,stats,settings)
    harmonic=sum(1/i for i in range(1,int(mask.sum())+1))
    qb=np.minimum(q*harmonic,1)
    return chosen&(qb<=settings["fdr"]),qb

def priority_selection(cfg,plan,stats,attribution,folds):
    """Use the full confirmation sample; no source-subset results are required."""
    full={};summaries=[]
    for sc in scenarios(cfg):
        mask,frequency=candidates(attribution,folds,np.arange(40),sc)
        selected,q=select(mask,stats,sc)
        by,qby=by_selected(mask,stats,sc)
        ss=sets(selected)
        summary=dict(name=sc["name"],settings=sc,counts={k:len(v) for k,v in ss.items()},
                     candidates=int(mask.sum()),by_union_count=int(by.any(0).sum()),
                     selected_sets={k:sorted(v) for k,v in ss.items()},
                     valid=True,unavailable_reason=None)
        full[sc["name"]]=dict(mask=mask,frequency=frequency,selected=selected,q=q,
                             by=by,qby=qby,sets=ss,summary=summary)
        summaries.append(summary)
    output=OUT/"priority_figures";output.mkdir(exist_ok=True)
    marker=output/"completion.json"
    if not marker.exists():
        from .report import render_selection
        import subprocess
        render_selection(cfg,full,output)
        write(output/"configurations_results.json",clean(summaries))
        table(output/"configuration_counts.csv",(dict(configuration=x["name"],**x["settings"],
            **x["counts"],candidate_hypotheses=x["candidates"],BY_union=x["by_union_count"])
            for x in summaries))
        version=subprocess.check_output(["git","rev-parse","HEAD"],cwd=CODE.parent.parent,text=True).strip()
        (output/"README.zh.md").write_text(
            "# M1 阈值敏感性：优先图表\n\n"
            "图1和图2使用完整确认来源及全部已完成的匹配对照。每个配置分别重算候选家族的 BH 校正。"
            "所有阈值沿用预先固定配置，展示最终筛选集合。来源子集重选随后执行，稳健性 A/B/C 建议见完成后的总报告。\n\n"
            "模型：Qwen3-8B；候选来源40个、确认来源40个；有效来源数："+json.dumps(plan["eligible_sources_by_kind"],ensure_ascii=False)+"。\n\n"
            "实际提交："+version+"。\n\n"
            "![阈值与头数量](figures/figure1_threshold_counts.png)\n\n"
            "![集合稳定性热图](figures/figure2_jaccard.png)\n\n"
            "图1为单因素扫描，虚线标记默认值；A/P、C/F 与并集分别显示。图2为全部12个预定配置的并集 Jaccard，"
            "两个集合均为空时显示 NA。数据文件保存 A/P、C/F、并集相似度、各配置数量和完整选头集合。"
            "图中的相似度反映阈值变化，跨来源重复性由后续子集统计评价。\n",
            encoding="utf-8")
        write(marker,dict(complete=True,model="Qwen3-8B",commit=version,
                          stage="complete_source_threshold_sensitivity",configurations=len(full),
                          configuration=cfg,source_counts=plan["eligible_sources_by_kind"],
                          matched_pairs=plan["matched_pairs_by_kind"],
                          source_subset_analysis_pending_at_publication=True,
                          parameters_changed_using_outcomes=False))
    return full

def run():
    started=time.monotonic()
    cfg=read(CODE/"configuration.json");plan=read(OUT/"plan.json")
    if cfg!=plan["configuration"]:raise ValueError("Settings changed after candidate plan")
    effect=aggregate()
    from .worker import publish
    publish("matched controls and source effects completed")
    source=np.load(OUT/"screening_source_attributions.npz",allow_pickle=False)
    attribution,folds=source["attribution"],source["folds"]
    configs=scenarios(cfg)
    full_path=OUT/"statistics_full.npz"
    if full_path.exists():stats=dict(np.load(full_path,allow_pickle=False))
    else:
        stats=stats_for(effect,cfg,cfg["seed"]+1000,np.arange(40))
        np.savez_compressed(full_path,**stats)
    preselected=priority_selection(cfg,plan,stats,attribution,folds)
    write(OUT/"analysis_progress.json",dict(stage="priority_figures_complete",
                source_subset_analysis_pending=True,seconds=time.monotonic()-started))
    publish("threshold curves and set stability heatmap")
    full={}
    frequencies={s["name"]:np.zeros((4,1152),int) for s in configs}
    union_frequency={s["name"]:np.zeros(1152,int) for s in configs}
    ap_frequency={s["name"]:np.zeros(1152,int) for s in configs}
    cf_frequency={s["name"]:np.zeros(1152,int) for s in configs}
    subset_rows=[]
    subsets=read(OUT/"source_subsets.json")
    stage=OUT/"subsets";stage.mkdir(exist_ok=True)
    for sub in subsets:
        target=stage/(str(sub["id"])+".npz")
        if target.exists():st=dict(np.load(target,allow_pickle=False))
        else:
            st=stats_for(effect,cfg,cfg["seed"]+2000+sub["id"],sub["confirm"])
            np.savez_compressed(target,**st)
        saved={}
        for sc in configs:
            mask,_=candidates(attribution,folds,sub["screen"],sc)
            selected,q=select(mask,st,sc)
            frequencies[sc["name"]]+=selected
            union_frequency[sc["name"]]+=selected.any(0)
            ap_frequency[sc["name"]]+=selected[:2].any(0)
            cf_frequency[sc["name"]]+=selected[2:].any(0)
            ss=sets(selected);saved[sc["name"]]={k:sorted(v) for k,v in ss.items()}
            subset_rows.append(dict(subset=sub["id"],configuration=sc["name"],
                screening_sources=len(sub["screen"]),confirmation_sources=len(sub["confirm"]),
                candidates=int(mask.sum()),**{k:len(v) for k,v in ss.items()}))
        write(stage/(str(sub["id"])+".json"),saved)
        write(OUT/"analysis_progress.json",dict(stage="source_subsets",completed=sub["id"]+1,
                    total=len(subsets),seconds=time.monotonic()-started))
    table(OUT/"source_subset_counts.csv",subset_rows)
    summaries=[];heads=[]
    for sc in configs:
        name=sc["name"];early=preselected[name]
        mask,frequency,selected,q=(early[k] for k in ["mask","frequency","selected","q"])
        by,qby=early["by"],early["qby"]
        ss=early["sets"];uf=union_frequency[name]/len(subsets)
        apf=ap_frequency[name]/len(subsets);cff=cf_frequency[name]/len(subsets)
        core={h for h in ss["union"] if uf[h]>=cfg["stability_criteria"]["core_subset_frequency_min"]}
        summary=dict(name=name,settings=sc,counts={k:len(v) for k,v in ss.items()},
                     candidates=int(mask.sum()),by_union_count=int(by.any(0).sum()),
                     selected_sets={k:sorted(v) for k,v in ss.items()},core_heads=sorted(core),
                     core_coverage=len(core)/len(ss["union"]) if ss["union"] else None,
                     valid=True,unavailable_reason=None)
        summaries.append(summary)
        full[name]=dict(mask=mask,selected=selected,sets=ss,union_frequency=uf,AP_frequency=apf,
                        CF_frequency=cff,category_frequency=frequencies[name]/len(subsets),summary=summary)
        for k,kind in enumerate(KINDS):
            for h in range(1152):
                vals=effect["contrast"][k,:,h]
                valid=np.isfinite(vals)
                total=np.abs(vals[valid]).sum()
                largest=float(np.abs(vals[valid]).max()/total) if total>0 else None
                row=dict(configuration=name,kind=kind,head=h,layer=h//32,head_in_layer=h%32,
                         candidate=bool(mask[k,h]),five_fold_frequency=float(frequency[k,h]),
                         selected=bool(selected[k,h]),q_BH=q[k,h],q_BY=qby[k,h],
                         selected_BY=bool(by[k,h]),subset_selection_frequency=frequencies[name][k,h]/len(subsets),
                         max_source_absolute_share=largest,
                         true_mean=float(np.mean(effect["true"][k,valid,h])) if valid.any() else None,
                         null_mean=float(np.mean(effect["null"][k,valid,h])) if valid.any() else None,
                         **{z:a[k,h] for z,a in stats.items()})
                row["effect_measured"]=bool(valid.any())
                if not valid.any():
                    for field in stats:
                        if field!="valid_sources":row[field]=None
                heads.append(clean(row))
    table(OUT/"all_configuration_heads.csv.gz",heads,True)
    write(OUT/"configurations_results.json",clean(summaries))
    table(OUT/"configuration_counts.csv",(dict(configuration=x["name"],**x["settings"],
                   **x["counts"],candidate_hypotheses=x["candidates"],BY_union=x["by_union_count"],
                   core_count=len(x["core_heads"]),core_coverage=x["core_coverage"]) for x in summaries))
    matrix=[]
    for a,x in full.items():
        for b,y in full.items():
            matrix.append(dict(configuration_a=a,configuration_b=b,
                **{k+"_jaccard":jaccard(x["sets"][k],y["sets"][k]) for k in ["union","AP","CF"]}))
    table(OUT/"jaccard.csv",matrix)
    default=full["default"]
    criteria=cfg["stability_criteria"]
    diagnostics=[]
    for name,x in full.items():
        jj={k:jaccard(default["sets"][k],x["sets"][k]) for k in ["union","AP","CF"]}
        rel=abs(len(x["sets"]["union"])-len(default["sets"]["union"]))/max(1,len(default["sets"]["union"]))
        groups_ok=all(jj[k] is None or jj[k]>=criteria["group_jaccard_min"] for k in ["AP","CF"])
        passing=(jj["union"] is not None and jj["union"]>=criteria["union_jaccard_min"] and groups_ok
                 and rel<=criteria["relative_count_change_max"] and x["summary"]["core_coverage"] is not None
                 and x["summary"]["core_coverage"]>=criteria["default_core_coverage_min"])
        diagnostics.append(dict(configuration=name,union_jaccard=jj["union"],AP_jaccard=jj["AP"],
            CF_jaccard=jj["CF"],relative_count_change=rel,core_coverage=x["summary"]["core_coverage"],
            passes_criteria=bool(passing)))
    table(OUT/"stability_criteria.csv",diagnostics)
    passed={r["configuration"]:r["passes_criteria"] for r in diagnostics}
    neighbour_names=[]
    for parameter,values in cfg["grid"].items():
        ordered=sorted(values);center=ordered.index(cfg["defaults"][parameter])
        for j in [center-1,center+1]:
            if 0<=j<len(ordered):
                neighbour_names.append(next(sc["name"] for sc in configs
                    if sc["name"].startswith(parameter+"_") and sc[parameter]==ordered[j]))
    corecoverage=default["summary"]["core_coverage"]
    if default["sets"]["union"] and corecoverage is not None and corecoverage>=criteria["default_core_coverage_min"]:
        recommendation="A" if all(passed[n] for n in neighbour_names) else "B"
    else:recommendation="C"
    intervals={}
    for param,values in cfg["grid"].items():
        values=sorted(values);flags=[]
        for v in values:
            name="default" if v==cfg["defaults"][param] else next(s["name"] for s in configs
                    if s["name"].startswith(param+"_") and s[param]==v)
            flags.append(passed[name])
        center=values.index(cfg["defaults"][param]);lo=hi=center
        if flags[center]:
            while lo>0 and flags[lo-1]:lo-=1
            while hi+1<len(values) and flags[hi+1]:hi+=1
            intervals[param]=values[lo:hi+1]
        else:intervals[param]=[]
    core_records=[]
    for h in sorted(default["sets"]["union"]):
        a,e=default["AP_frequency"][h],default["CF_frequency"][h]
        core_records.append(dict(head=h,layer=h//32,head_in_layer=h%32,
            default_AP=h in default["sets"]["AP"],default_CF=h in default["sets"]["CF"],
            union_subset_frequency=default["union_frequency"][h],AP_subset_frequency=a,CF_subset_frequency=e,
            stable_core=h in default["summary"]["core_heads"],
            stable_AP_only=bool(a>=.8 and e<=.2),stable_CF_only=bool(e>=.8 and a<=.2),
            stable_shared=bool(a>=.8 and e>=.8)))
    table(OUT/"core_heads.csv",core_records,fieldnames=["head","layer","head_in_layer","default_AP","default_CF","union_subset_frequency","AP_subset_frequency","CF_subset_frequency","stable_core","stable_AP_only","stable_CF_only","stable_shared"])
    result=dict(complete=True,model="Qwen3-8B",recommendation=recommendation,
                default=default["summary"],stable_scan_values=intervals,stability=diagnostics,
                source_counts=plan["eligible_sources_by_kind"],matched_pairs=plan["matched_pairs_by_kind"],
                independent_replications=0,source_subsets=len(subsets),
                parameters_changed_using_outcomes=False,M2_fits=0,M3_fits=0,
                no_task_accuracy_used=True,seconds=time.monotonic()-started)
    write(OUT/"analysis_results.json",clean(result))
    from .report import render
    render(cfg,plan,effect,stats,full,result,core_records)
    write(OUT/"completion.json",clean(result))
    return result

if __name__=="__main__":
    from .worker import publish
    try:
        r=run()
        publish("completed analysis figures and report")
        print(json.dumps(dict(complete=True,recommendation=r["recommendation"])))
    except Exception as exc:
        import traceback
        write(OUT/"analysis_failure.json",dict(error=str(exc),traceback=traceback.format_exc()))
        try:publish("analysis failure retained")
        except Exception:pass
        raise
