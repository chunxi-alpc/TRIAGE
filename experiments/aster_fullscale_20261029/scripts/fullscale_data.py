"""Real snapshot adapters. Private records stay on the experiment server."""
import json,re,random
from pathlib import Path
from collections import Counter
from prepare_group_splits import allocate

def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def norm(s):
    return re.sub(r'\s+',' ',str(s)).strip().casefold()
def flatten(node):
    result=[node]
    for k in ('yes','no'):
        if isinstance(node.get(k),dict):result+=flatten(node[k])
    return result

def mega(root,split):
    root=Path(root)/split
    ts=read(root/'decision_trees.jsonl'); ps=read(root/'case_paths.jsonl'); cs=read(root/'vignette_cases.jsonl')
    trees={str(r['guideline_id']):r for r in ts};paths={str(r['guideline_id']):r['paths'] for r in ps}
    cases=[]; reasons=Counter(); cache={}
    for i,c in enumerate(cs):
        gid=str(c['guideline_id']); t=trees.get(gid);pp=paths.get(gid,[])
        row={'case_id':f'mega:{split}:{gid}:{c["path_id"]}', 'dataset':'mega','source_split':split,'group':c['filename'],'patient':c['text'],'invalid_source':None}
        try:
            tree,nodes=cache.setdefault(gid,(None,None))
            if tree is None:
                tree=json.loads(t['text'])['decision_tree'];nodes=flatten(tree);cache[gid]=(tree,nodes)
            path=pp[int(c['path_id'])-1]
            row.update(rules=[{'id':str(n['index']),'condition':n.get('description',''),'text':json.dumps({k:v for k,v in n.items() if k not in ('yes','no')} | {k:n[k]['index'] for k in ('yes','no') if isinstance(n.get(k),dict)},ensure_ascii=False)} for n in nodes if not n.get('leaf')],tree=tree,gold=[n['index'] for n in path],gold_route={str(n['index']):1 if n.get('decision_outcome')=='yes' else -1 for n in path if 'decision_outcome' in n})
            file=root/'guideline_md'/c['filename']/(c['filename']+'.md')
            row['guideline']=file.read_text() if file.exists() else ''
        except Exception as e:
            row['invalid_source']=type(e).__name__;reasons[type(e).__name__]+=1
        cases.append(row)
    return cases,{'cases':len(cs),'trees':len(ts),'case_path_trees':len(ps),'case_fields':list(cs[0]),'tree_fields':list(ts[0]),'invalid_source':dict(reasons)}

def med(parquet,pathsfile):
    import pyarrow.parquet as pq
    rows=pq.read_table(parquet).to_pylist(); paths=json.loads(Path(pathsfile).read_text());out=[]
    for i,c in enumerate(rows):
        disease=c['disease']; source=paths.get(disease,[])
        rules=[{'id':str(j),'condition':' > '.join(p[:-1]),'text':f'R{j}: If the clinical pathway follows {" > ".join(p[:-1])}, next action: {p[-1]}.'} for j,p in enumerate(source) if len(p)>=2]
        matched=[r['id'] for r in rules if norm(r['condition'])==norm(' > '.join(c['path'][:-1]))]
        out.append({'case_id':f'med:{i}','dataset':'med','source_split':'train','group':re.sub(r'_\d+$','',disease),'source_tree':disease,'patient':c['profile'],'options':c['options'],'gold':c['answer'],'rules':rules,'gold_route':{r['id']:1 if r['id'] in matched else -1 for r in rules},'invalid_source':None if rules else 'missing_source_tree'})
    return out,{'cases':len(rows),'source_trees':len(set(r['disease'] for r in rows)),'source_families':len(set(r['group'] for r in out)),'fields':list(rows[0])}

def prepare(args):
    root=Path(args.run);root.mkdir(parents=True,exist_ok=False)
    train,a=mega(args.mega,'train_val');test,b=mega(args.mega,'test');md,c=med(args.med,args.paths)
    # Filename and exact normalized guideline text form a source family across train/test.
    testgroups={r['group'] for r in test}; text_to_groups={}
    for r in train+test:
        text=norm(r.get('guideline',''))
        if text:text_to_groups.setdefault(text,set()).add(r['group'])
    parent={r['group']:r['group'] for r in train+test}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    for gs in text_to_groups.values():
        gs=sorted(gs)
        for g in gs[1:]:parent[find(g)]=find(gs[0])
    for r in train+test:r['group']=find(r['group'])
    testgroups={r['group'] for r in test}
    mapping=allocate(sorted({r['group'] for r in train}-testgroups),{'fit':.7,'select':.15,'confirm':.15},20261029)
    for r in train:r['split']=mapping.get(r['group'],'overlap_official_test')
    for r in test:r['split']='official_test'
    mm=allocate(sorted({r['group'] for r in md}),{'fit':.6,'select':.2,'heldout':.2},20261029)
    for r in md:r['split']=mm[r['group']]
    for name,rows in [('mega',train+test),('med',md)]:
        with (root/f'{name}.jsonl').open('x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    audit={'mega_train':a,'mega_test':b,'med':c,'split_counts':dict(Counter(r['dataset']+':'+r['split'] for r in train+test+md)),'mega_source_overlap_excluded_from_development':sum(r['split']=='overlap_official_test' for r in train),'cross_dataset_overlap':'MEGA filename/text families and MedGUIDE disease families grouped; cross-corpus clinical source identity is not established; MedGUIDE is source-heldout transfer, independence not claimed','hashing':False}
    write(root/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False))
