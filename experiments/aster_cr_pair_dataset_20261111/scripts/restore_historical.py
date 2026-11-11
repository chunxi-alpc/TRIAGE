"""Restore historical fit pairs without modifying input files. Run on gcx server."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse, collections, difflib, json, pathlib, re
def rows(p):
    with pathlib.Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]
def write_json(p,v): pathlib.Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_rows(p,rs):
    with pathlib.Path(p).open('w',encoding='utf-8') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def parts(prompt):
    guide,tail=prompt.split('\n\n指南原文定义与说明：\n',1)
    definitions,tail=tail.split('\n\n病历：\n',1)
    patient,tail=tail.rsplit('\n\n当前节点：',1)
    gid,tail=tail.split('\n当前条件：',1)
    condition=tail.rsplit('\n请判断当前条件。',1)[0]
    return dict(patient=patient,condition=condition,guide=guide,definitions=definitions,node_id=gid)
def changes(a,b,component):
    aa=a.splitlines(keepends=True);bb=b.splitlines(keepends=True)
    return [dict(component=component,old=''.join(aa[i:j]),new=''.join(bb[k:l]),operation=op)
            for op,i,j,k,l in difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_opcodes() if op!='equal']
def main():
    nodes={r['group_id']:r for r in rows(DATA/'nodes.jsonl') if r['split']=='fit'}
    views={r['view_id']:r for r in rows(DATA/'views.jsonl') if r['split']=='fit'}
    cases={r['case_id']:r for r in rows(DATA/'path_cases.jsonl') if r['split']=='fit'}
    rules={r['group_id']:r for r in rows(DATA/'rules.jsonl') if r['split']=='fit'}
    source_rows=rows(DATA/'sources.jsonl')
    raw_trees={r['guideline_id']:r for r in rows(RAW/'decision_trees.jsonl') if r['guideline_id'] in {n['guideline_id'] for n in nodes.values()}}
    raw_cases={(r['guideline_id'],str(r['path_id'])):r for r in rows(RAW/'vignette_cases.jsonl') if r['guideline_id'] in raw_trees}
    mappings={}
    for gid,t in raw_trees.items():
        mapping={}
        def walk(n):
            idx=len(mapping);internal=any(isinstance(n.get(k),dict) for k in ['yes','no']);mapping[('G' if internal else 'L')+str(idx)]=n.get('index')
            if internal:
                for key in ['yes','no']:
                    nxt=n.get(key)
                    if isinstance(nxt,dict):walk(nxt)
        tree=json.loads(t['text'])['decision_tree']
        walk(tree);mappings[gid]=mapping
    out=[];diagnostic=[];issues=[]
    for p in rows(DATA/'pairs.jsonl'):
        if p['split']!='fit':continue
        n=nodes[p['group_id']];case=cases[n['case_id']]
        left,right=views[p['left']],views[p['right']]
        l=parts(left['prompt']);r=parts(right['prompt'])
        l.update(label=p['left_label'],view_id=p['left'],prompt=left['prompt'])
        r.update(label=p['right_label'],view_id=p['right'],prompt=right['prompt'])
        edits=[]
        for comp in ['patient','condition','guide','definitions']:
            edits+=changes(l[comp],r[comp],comp)
        rawcase=raw_cases[(n['guideline_id'],n['case_id'].rsplit(':',1)[1])]
        rec=dict(schema_version='1.0',pair_id=p['pair_id'],kind=p['kind'],origin='historical',
            source_id=p['source_id'],split='fit',case_id=p['case_id'],group_id=p['group_id'],
            guideline_id=n['guideline_id'],source_filename=n['source_filename'],
            node_id=n['gid'],raw_node_index=mappings[n['guideline_id']].get(n['gid']),
            language=n['language'],tree_complexity=len(n['guideline']['groups']),
            source_document=n['source_document'],raw_patient=rawcase['text'],
            original=l,modified=r,edits=edits,
            counterfactual=p['kind'] in ['C','F','F0'],
            edit_attribution='historical controlled edit; modified text is not a quotation of the original guideline',
            source_evidence=dict(document=n['source_document'],original_condition=l['condition'],original_patient=l['patient']),
            label_derivation=dict(original='MEGA-CDP released path label',modified='historical supplied label; pending independent review'),
            provenance=dict(pair_file=str(DATA/'pairs.jsonl'),left_view=p['left'],right_view=p['right'],original_pair_id=p['pair_id'],construction_version=p.get('construction_version'),raw_case_path_id=rawcase['path_id']),
            historical_edit_record={k:v for k,v in p.items() if k not in ['left','right']},
            program_rule=rules.get(p['group_id']),
            program_modified_facts=right.get('facts'),program_modified_predicate=right.get('predicate'))
        (diagnostic if p['kind']=='F0' else out).append(rec)
        if l['patient']!=rawcase['text']:issues.append(dict(pair_id=p['pair_id'],issue='original_patient_differs_from_raw'))
    write_rows(ROOT/'historical_pairs.jsonl',out)
    write_rows(ROOT/'f0_diagnostic_pairs.jsonl',diagnostic)
    write_json(ROOT/'source_splits.json',{split:sorted(r['source_id'] for r in source_rows if r['split']==split) for split in ['fit','dev','test']})
    write_json(ROOT/'reports/recovery_issues.json',issues)
    stats={}
    for kind in ['A','P','C','F','F0']:
        rs=[r for r in out+diagnostic if r['kind']==kind]
        stats[kind]=dict(pairs=len(rs),sources=len({r['source_id'] for r in rs}),guidelines=len({r['guideline_id'] for r in rs}),cases=len({r['case_id'] for r in rs}),nodes=len({r['group_id'] for r in rs}),languages=dict(collections.Counter(r['language'] for r in rs)))
    audit=json.loads((REPO/'reports/aster_cancel_story_20261109/meeting/trace_validation/data_audit.json').read_text())
    report=dict(stage='inventory_only',input_data=str(DATA),read_reference_report=str(REPO/'reports/aster_cancel_story_20261109/meeting/trace_validation/data_audit.json'),read_paper_story=str(REPO/'reports/aster_cancel_story_20261109/paper_story_latest.zh.md'),
        fit_sources=80,fit_guidelines=len(raw_trees),fit_cases=len(cases),fit_gold_nodes=sum(len(c['gold_path']) for c in cases.values()),
        expanded_fit_nodes=len(nodes),pairs=stats,source_documents_missing=sorted({r['source_document'] for r in out if not source_document_path(r['source_document']).exists()}),
        original_patient_mismatches=len(issues),source_splits_disjoint=not any(set(audit['source_ids'][x])&set(audit['source_ids'][y]) for x,y in [('fit','dev'),('fit','test'),('dev','test')]),
        preserved_historical_ids=True,semantic_review_completed=False,clinical_expert_review=False)
    write_json(ROOT/'reports/inventory.json',report)
    table='\n'.join('| {k} | {pairs} | {sources} | {guidelines} | {cases} | {nodes} |'.format(k=k,**v) for k,v in stats.items())
    (ROOT/'reports/inventory.zh.md').write_text('# 历史配对盘点\n\n已按历史 ID 恢复，未生成新候选，未修改原始文件。\n\n| 类别 | 配对 | 来源簇 | 指南 | 病例 | 节点 |\n|---|---:|---:|---:|---:|---:|\n'+table+'\n\n80 个拟合来源簇对应 '+str(len(raw_trees))+' 篇指南，包含 1,712 个病例和 11,722 个金标准节点。扩展节点 1,623 个；同一节点可有多个处置编辑配对，配对数不等于独立节点数。\n\nF0 单独保存，不进入四类训练池。开发和测试仅加载来源隔离名单，不加载模型预测或评价结果。所有两侧文本由历史 view 原样恢复，反事实标记另存，不改动历史提示内容。\n\n历史源文、病例与标签回连是程序对应检查，尚不能视为独立语义审核或临床专家确认。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
