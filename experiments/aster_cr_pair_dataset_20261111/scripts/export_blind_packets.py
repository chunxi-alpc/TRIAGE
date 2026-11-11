"""Export lossless review input without generation reasoning or program-derived rules."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,json,pathlib
from restore_historical import ROOT,rows,write_json
def export(stage, file=None, folder=None):
    file=pathlib.Path(file) if file is not None else ROOT/('historical_pairs.jsonl' if stage=='historical' else 'generated_candidates.jsonl')
    data=rows(file);groups=collections.defaultdict(list)
    for r in data:groups[r['source_id']].append(r)
    folder=pathlib.Path(folder) if folder is not None else ROOT/'blind_packets'/stage;folder.mkdir(parents=True,exist_ok=True)
    index=[]
    for sid,rs in sorted(groups.items()):
        pack=dict(source_id=sid,stage=stage,full_source_documents={},original_guidelines={},cases={},nodes={},contrasts={},pairs=[])
        seen={}
        for r in rs:
            a,b=r['original'],r['modified'];gid=str(r['guideline_id'])
            doc=r['source_document']
            if doc not in pack['full_source_documents']:pack['full_source_documents'][doc]=source_document_path(doc).read_text(encoding='utf-8')
            pack['original_guidelines'].setdefault(gid,dict(document=doc,guide=a['guide'],definitions=a['definitions']))
            if r['case_id'] in pack['cases'] and pack['cases'][r['case_id']]!=a['patient']:raise ValueError('original case mismatch')
            pack['cases'][r['case_id']]=a['patient']
            pack['nodes'].setdefault(r['group_id'],dict(case_id=r['case_id'],guideline_id=r['guideline_id'],node_id=r['node_id'],raw_node_index=r['raw_node_index'],condition=a['condition'],original_label=a['label']))
            contrast={}
            if a['patient']!=b['patient']:contrast['modified_patient']=b['patient']
            if a['condition']!=b['condition']:contrast['modified_condition']=b['condition']
            if a['guide']!=b['guide']:
                contrast['guide_literal_edits']=[{k:e[k] for k in ['old','new','occurrence','all','expected_count','start','end','operation'] if k in e} for e in r['edits'] if e['component']=='guide']
            if a['definitions']!=b['definitions']:contrast['modified_definitions']=b['definitions']
            key=json.dumps(contrast,ensure_ascii=False,sort_keys=True)
            if key not in seen:
                cid='edit'+str(len(seen));seen[key]=cid;pack['contrasts'][cid]=contrast
            pack['pairs'].append(dict(pair_id=r['pair_id'],group_id=r['group_id'],expected_kind=r['kind'],contrast=seen[key],declared_left_label=a['label'],is_artificial_counterfactual=r['kind'] in ['C','F']))
        target=folder/(sid.replace(':','_')+'.json');write_json(target,pack)
        index.append(dict(source_id=sid,path=str(target),pairs=len(rs),cases=len(pack['cases']),nodes=len(pack['nodes']),guidelines=len(pack['original_guidelines']),contrasts=len(pack['contrasts'])))
    write_json(folder/'index.json',index)
    print(json.dumps(dict(stage=stage,sources=len(index),pairs=sum(r['pairs'] for r in index)),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['historical','generated'],required=True);a=p.parse_args();export(a.stage)
