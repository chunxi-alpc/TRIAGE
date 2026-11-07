"""Programmatic A/P/C/F variants around original cases and recorded fact edits."""
import copy,itertools,math
from collections import Counter,defaultdict
from .common import *
from experiments.aster_closed_form.common import evaluate,fields
from experiments.aster_closed_form.data import prompt,canonical_guide,set_condition,protected_swap
GRID=[-2,-1,-.5,0,.5,1,2]
def leaves(e):
 if e["op"] in ("and","or"):
  for x in e["args"]:yield from leaves(x)
 elif e["op"]=="not":yield from leaves(e["arg"])
 else:yield e
def expr_text(e,names):
 op=e["op"]
 if op in ("and","or"):return "("+(" AND " if op=="and" else " OR ").join(expr_text(x,names) for x in e["args"])+")"
 if op=="not":return "NOT ("+expr_text(e["arg"],names)+")"
 return names[e["field"]]+" "+op+" "+json.dumps(e["value"],ensure_ascii=False)
def ancestors_ok(g,a,facts):
 ex={x["gid"]:x["predicate"] for x in a["ancestor_predicates"]}
 return all(int(evaluate(ex[x["gid"]],facts))==x["label"] for x in g["ancestors"])
def edited_patient(g,ed,value):
 patient=g["patient"]
 options=ed.get("legal_values")
 if options:
  item=next(x for x in options if x["value"]==value);reps=item["replacements"]
  if len(reps)!=len(ed["spans"]):raise ValueError("wrong discrete replacements")
 else:
  number=format(value,".12g");reps=[x["template"].replace("{value}",number) for x in ed["spans"]]
 for s,repl in zip(ed["spans"],reps):patient=patient.replace(s["old"],repl)
 return patient
def replace_target(guide,gid,text):
 out=copy.deepcopy(guide)
 node=next(n for n in out["groups"] if n["group_id"]==gid)
 for r in node["rules"]:
  suffix="\uff08\u5224\u65ad\u4e3a"+("\u662f" if r["rule_id"].endswith("Y") else "\u5426")+"\uff09"
  r["condition"]=text+suffix
 return out
def build():
 groups=rows(ROOT/"data/nodes.jsonl")
 ann={r["group_id"]:r for i in range(3) for r in rows(ROOT/"data"/f"annotations_{i}.jsonl")}
 views=[];pairs=[];rules=[];e4=[];missing=[]
 for g in groups:
  gid=g["group_id"];a=ann.get(gid,{}).get("value",{});valid=ann.get(gid,{}).get("validation",{})
  base=dict(group_id=gid,source_id=g["source_id"],split=g["split"],case_id=g["case_id"],cohort="main")
  def view(name,patient=None,guide=None,label=None,kind="original",**kw):
   v=dict(base,view_id=gid+":"+name,variant=name,kind=kind,original=name=="full",
    label=g["original_label"] if label is None else int(label),
    prompt=prompt(guide or g["guideline"],g["definitions"],patient if patient is not None else g["patient"],g["gid"]),**kw)
   views.append(v);return v
  def pair(kind,l,r,subtype="",**kw):
   pairs.append(dict(base,pair_id=gid+":"+kind+":"+str(len(pairs)),kind=kind,subtype=subtype,
    left=l["view_id"],right=r["view_id"],left_label=l["label"],right_label=r["label"],**kw))
  full=view("full")
  # Output forms are all decoded back to the same semantic labels.
  for name,tail,labels in [
   ("order_no_yes",'请判断当前条件。只输出“否”或“是”。\n答案：',["是","否"]),
   ("map_ab",'请判断当前条件。A 表示条件成立，B 表示条件不成立。只输出 A 或 B。\n答案：',["A","B"]),
   ("map_ba",'请判断当前条件。B 表示条件成立，A 表示条件不成立。只输出 A 或 B。\n答案：',["B","A"]),
   ("equivalent",'根据给定事实，当前节点的完整条件是否满足？只输出“是”或“否”。\n答案：',["是","否"])]:
   v=view(name,kind="format");v["prompt"]=full["prompt"].rsplit('请判断当前条件。',1)[0]+tail;v["labels"]=labels
  if valid.get("state")!="complete":
   missing.append(dict(base,state=valid.get("state","missing"),reason=valid.get("reason")))
   if valid.get("state")=="E4_candidate":e4.append(dict(base,**valid))
   continue
  rule=dict(base,predicate=a["predicate"],facts=a["facts"],descriptions=a["descriptions"],
     original_label=g["original_label"],program_label=int(evaluate(a["predicate"],a["facts"])))
  rules.append(rule)
  protected=set(a.get("protected_action_ids",[]))
  def action_guide(mode):
   guide=copy.deepcopy(g["guideline"])
   for node in guide["groups"]:
    for r in node["rules"]:
     import re
     m=re.match(r"(L\d+)([：:])(.*)",r["action"],re.S)
     if not m or m[1] in protected:continue
     old=m[3]
     if mode=="delete":new="[处置已删除]"
     else:
      unit="记录内容不影响当前条件判断。" if g["language"]=="zh" else "Administrative information. "
      new=(unit*(1+len(old)//len(unit)))[:len(old)]
     r["action"]=m[1]+m[2]+new
   return guide
  deletion=action_guide("delete");clean=view("remove_action",guide=deletion,kind="A")
  if clean["prompt"]!=full["prompt"]:pair("A",full,clean,"presence")
  neutral=view("neutral_action",guide=action_guide("neutral"),kind="A")
  if neutral["prompt"]!=full["prompt"]:pair("A",full,neutral,"neutral")
  sw,changed=protected_swap(g,protected)
  if changed:pair("A",full,view("swap_action",guide=sw,kind="A"),"swap")
  decisive="\n".join(dict.fromkeys(a["decisive_sentences"]))
  if decisive:
   view("neutral_background",patient=decisive,kind="cleaning")
   view("both_clean",patient=decisive,guide=deletion,kind="cleaning")
  cp=view("condition_ancestors",kind="cleaning")
  context="\n".join(x["condition"] for x in g["ancestors"])
  target=next(x["condition"] for x in canonical_guide(g["guideline"])["nodes"] if x["id"]==g["gid"])
  cp["prompt"]=f"祖先条件：\n{context}\n当前条件：{target}\n病历：\n{g['patient']}\n请判断当前条件。只输出“是”或“否”。\n答案："
  bg=a.get("background")
  if bg:
   f0=dict(a["facts"],**{bg["field"]:bg["before"]});f1=dict(a["facts"],**{bg["field"]:bg["after"]})
   if evaluate(a["predicate"],f0)==evaluate(a["predicate"],f1) and ancestors_ok(g,a,f1):
    v=view("P_background",patient=g["patient"].replace(bg["old"],bg["new"]),kind="P")
    if v["prompt"]!=full["prompt"]:pair("P",full,v,"background",changed_field=bg["field"],decisive_facts=a["facts"])
  grid_views=[]
  for j,ed in enumerate(a.get("edits",[])):
   field=ed["field"];matched=[x for x in leaves(a["predicate"]) if x["field"]==field]
   if ed.get("legal_values"):values=[(x["value"],None,None) for x in ed["legal_values"]]
   else:
    thresholds=sorted({float(x["value"]) for x in matched if type(x["value"]) in (int,float)})
    values=[(t+d*ed["step"],t,d) for t in thresholds for d in GRID]
   seen=set()
   for k,(val,t,distance) in enumerate(values):
    if val in seen:continue
    integer_only=ed.get("integer_only") or str(ed.get("unit","")).lower() in {"count","number","lesions","events","episodes"}
    if integer_only and type(val) in (int,float) and abs(val-round(val))>1e-8:continue
    seen.add(val)
    if type(val) in (int,float) and (val<ed.get("minimum",-math.inf) or val>ed.get("maximum",math.inf)):continue
    facts=dict(a["facts"],**{field:val})
    if not ancestors_ok(g,a,facts):continue
    patient=edited_patient(g,ed,val);label=int(evaluate(a["predicate"],facts))
    v=view(f"F_grid_{j}_{k}",patient=patient,label=label,kind="grid",field=field,value=val,threshold=t,step=ed.get("step"),distance=distance,facts=facts)
    grid_views.append((v,patient,facts))
    if patient!=g["patient"]:pair("F" if label!=full["label"] else "F0",full,v,"cross" if label!=full["label"] else "same_side")
  # C changes the full executable condition, with unchanged facts and a program-computed label.
  # Numeric C uses the fixed grid above; no unconstrained text generation.
  records=[(full,g["patient"],a["facts"])]+grid_views
  c_bases={}
  def c_base(old,pt,fa):
   if old["view_id"] not in c_bases:
    original=replace_target(g["guideline"],g["gid"],expr_text(a["predicate"],a["descriptions"]))
    c_bases[old["view_id"]]=view("C_base_"+str(len(c_bases)),guide=original,patient=pt,label=old["label"],kind="C",
      predicate=a["predicate"],facts=fa,original_patient_view=old["view_id"])
   return c_bases[old["view_id"]]
  for j,leaf in enumerate(leaves(a["predicate"])):
   variants=[];op=leaf["op"];value=leaf["value"]
   if op in ("<","<=",">",">=") and type(value) in (int,float):
    step=next((e["step"] for e in a.get("edits",[]) if e["field"]==leaf["field"] and e.get("step")),None)
    if step:
     variants += [(dict(leaf,value=value+d*step),"threshold") for d in GRID if d]
     variants += [(dict(leaf,op={"<":"<=","<=":"<",">":">=",">=":">"}[op]),"endpoint")]
   elif op in ("in","not_in") and isinstance(value,list) and len(value)>1:
    variants += [(dict(leaf,value=value[:k]+value[k+1:]),"list") for k in range(len(value))]
   elif op=="==" and type(value)==bool:variants.append((dict(leaf,value=not value),"mapping"))
   for k,(newleaf,subtype) in enumerate(variants):
    expr=copy.deepcopy(a["predicate"]);list(leaves(expr))[j].update(newleaf)
    guide=replace_target(g["guideline"],g["gid"],expr_text(expr,a["descriptions"]))
    # One example per changed rule suffices; multiple pairs never count as independent nodes.
    candidate=next(((v,pt,fa) for v,pt,fa in records if int(evaluate(expr,fa))!=v["label"]),None)
    if candidate:
     old,pt,fa=candidate
     v=view(f"C_{j}_{k}",guide=guide,patient=pt,label=evaluate(expr,fa),kind="C",predicate=expr,facts=fa)
     pair("C",c_base(old,pt,fa),v,subtype)
  if a["predicate"]["op"] in ("and","or"):
   expr=copy.deepcopy(a["predicate"]);expr["op"]="or" if expr["op"]=="and" else "and"
   candidate=next(((v,pt,fa) for v,pt,fa in records if int(evaluate(expr,fa))!=v["label"]),None)
   if candidate:
    old,pt,fa=candidate;guide=replace_target(g["guideline"],g["gid"],expr_text(expr,a["descriptions"]))
    pair("C",c_base(old,pt,fa),view("C_logic",guide=guide,patient=pt,label=evaluate(expr,fa),kind="C",predicate=expr,facts=fa),"logic")
 lines(ROOT/"data/views.jsonl",views);lines(ROOT/"data/pairs.jsonl",pairs)
 lines(ROOT/"data/rules.jsonl",rules);lines(ROOT/"data/E4_candidates.jsonl",e4);lines(ROOT/"data/unavailable.jsonl",missing)
 summary={}
 for split in ("fit","dev","test"):
  ps=[p for p in pairs if p["split"]==split]
  summary[split]={k:dict(pairs=sum(p["kind"]==k for p in ps),nodes=len({p["group_id"] for p in ps if p["kind"]==k}),sources=len({p["source_id"] for p in ps if p["kind"]==k})) for k in ("A","P","C","F","F0")}
  summary[split]["views"]=sum(v["split"]==split for v in views)
 summary["C_scope"]="exploratory" if summary["dev"]["C"]["pairs"]<50 or summary["test"]["C"]["pairs"]<100 else "target_met"
 summary["annotation_unavailable"]=len(missing);summary["E4_candidates"]=len(e4)
 write(ROOT/"data/variants.json",summary);status("build",state="complete",counts=summary)
 return summary
if __name__=="__main__":print(json.dumps(build(),ensure_ascii=False))
