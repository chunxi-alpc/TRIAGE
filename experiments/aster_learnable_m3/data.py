import json, re
from pathlib import Path

def canonical_guide(guide):
    """Full conditions, transitions and terminal payloads; no gold path."""
    groups=[];leaves={}
    for g in guide["groups"]:
        q=re.sub(r"（判断为[是否]）$","",g["rules"][0]["condition"])
        node=dict(id=g["group_id"],condition=q)
        for r in g["rules"]:
            m=re.match(r"(L\d+)[：:](.*)",r["action"],re.S)
            if m: target=m[1];leaves[target]=m[2]
            else:
                m=re.search(r"G\d+",r["action"])
                if not m: raise ValueError("Unrecognized transition: "+r["action"])
                target=m[0]
            node["yes" if r["rule_id"].endswith("Y") else "no"]=target
        groups.append(node)
    return dict(nodes=groups,leaves=leaves)
def render_guide(guide,definitions,mode="full",semantic=None,protect=()):
    g=canonical_guide(guide);semantic=semantic or {}
    lines=[f"{n['id']}: {n['condition']}\n  是 -> {n['yes']}；否 -> {n['no']}" for n in g["nodes"]]
    for key,text in g["leaves"].items():
        keep=mode=="full" or mode=="boundary_guideline" and key in protect or mode=="semantic" and key in semantic.get("keep_action_ids",[])
        lines.append(key+": "+(text if keep else "[处置已遮挡]"))
    defs=[]
    for i,d in enumerate(definitions):
        keep=mode in ("full","boundary_guideline","definition_restore") or mode=="semantic" and i in semantic.get("keep_definition_indices",[])
        if keep:defs.append(d["quote"])
    return "\n".join(lines)+"\n\n指南原文定义与说明：\n"+"\n".join(defs)
def prompt(guide,definitions,patient,gid,mode="full",semantic=None,protect=()):
    context=render_guide(guide,definitions,mode,semantic,protect)
    q=next(n["condition"] for n in canonical_guide(guide)["nodes"] if n["id"]==gid)
    return f"完整决策树：\n{context}\n\n病历：\n{patient}\n\n当前节点：{gid}\n当前条件：{q}\n请判断当前条件。只输出“是”或“否”。\n答案："


def read_rows(path):
    import gzip
    path=Path(path)
    if not path.exists():
        path=Path(str(path)+'.gz')
    if not path.exists():return []
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:
            return [json.loads(s) for s in f if s.strip()]
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def append(path, value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')

def reference_prompt(case, gid):
    group=next(g for g in case['guideline']['groups'] if g['group_id']==gid)
    q=re.sub(r'（判断为[是否]）$', '',group['rules'][0]['condition'])
    ancestors=[re.sub(r'（判断为[是否]）$', '',s) for s in group['ancestor_conditions']]
    return ('原祖先条件：\n'+'\n'.join(ancestors)+'\n\n病历：\n'+case['patient']+
            '\n\n当前节点：'+gid+'\n当前条件：'+q+'\n请判断当前条件。只输出“是”或“否”。\n答案：')

def target_prompt(case,gid):
    return prompt(case['guideline'],[],case['patient'],gid)

def prepare_data(outdir):
    from .tree import tree_guideline
    root=Path('/home/gcx/TRIAGE/data/MEGA-CDP')
    splits={}
    for split in ('train_val','test'):
        trees={r['filename']:r for r in read_rows(root/split/'decision_trees.jsonl')}
        paths={r['filename']:r for r in read_rows(root/split/'case_paths.jsonl')}
        converted={}
        def convert(r):
            filename=r['filename']
            if filename not in converted:
                converted[filename]=tree_guideline(json.loads(trees[filename]['text'])['decision_tree'])
            guide,mapping=converted[filename]
            path=paths[filename]['paths'][int(r['path_id'])-1]
            gold=[dict(gid='G'+str(mapping[str(n['index'])]['node']),label=int(n['decision_outcome']=='yes'))
                  for n in path if 'decision_outcome' in n]
            return dict(case_id=f"mega:{split}:{r['guideline_id']}:{r['path_id']}",
                source_id=f"{split}:{filename}",filename=filename,patient=r['text'],guideline=guide,
                gold_path=gold,gold_leaf='L'+str(mapping[str(path[-1]['index'])]['node']),
                language=r['language'],path_id=r['path_id'])
        allrows=read_rows(root/split/'vignette_cases.jsonl')
        if split=='test':splits['test']=[convert(r) for r in allrows]
        else:
            bykey={(r['filename'],int(r['path_id'])):r for r in allrows}
            for name,old in [('train','train'),('dev','development')]:
                selected=read_rows('/home/gcx/aster_representation_20261101/'+old+'.jsonl')
                splits[name]=[convert(bykey[(r['filename'],int(r['path_id']))]) for r in selected]
    for name,rs in splits.items():
        dest=Path(outdir)/'data'/f'{name}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w') as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save_json(Path(outdir)/'data/manifest.json',dict(counts={k:len(v) for k,v in splits.items()},
        original_cases=True,selection='existing 200 train / 100 development identities; original raw cases reloaded by filename/path_id',
        reference='unchanged patient, current condition, structural ancestor conditions without branch outcomes',
        source='/home/gcx/TRIAGE/data/MEGA-CDP'))
    return splits
