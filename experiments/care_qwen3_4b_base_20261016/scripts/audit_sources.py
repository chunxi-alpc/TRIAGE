import requests,json,pathlib
p=pathlib.Path('data/source_audit'); sess=requests.Session(); sess.proxies={'https':'http://127.0.0.1:28089'}
out=[]
for name in ['MedGUIDE/MedGUIDE-MCQA-8K','ncbi/MedCalc-Bench']:
    meta=json.loads((p/(name.replace('/','__')+'.json')).read_text())
    rev=meta['sha']
    readme=sess.get(f'https://huggingface.co/datasets/{name}/resolve/{rev}/README.md',timeout=40)
    (p/(name.replace('/','__')+'_README.md')).write_text(readme.text)
    split='train' if name.startswith('MedGUIDE') else 'test'
    response=sess.get('https://datasets-server.huggingface.co/rows',params={'dataset':name,'config':'default','split':split,'offset':0,'length':10},timeout=60)
    body=response.json(); (p/(name.replace('/','__')+'_rows.json')).write_text(json.dumps(body,indent=2))
    rows=body.get('rows',[])
    out.append({'dataset':name,'revision':rev,'license':meta.get('cardData',{}).get('license','unspecified in card'),
        'sample_fetch_status':response.status_code,'sample_rows':len(rows),
        'columns':[x['name'] for x in body.get('features',[])],
        'clinical_conflict_eligible':0,
        'reason':'NCCN-derived path examples are not verified complete executable mutually exclusive trees; license not explicit in card'
             if name.startswith('MedGUIDE') else 'Calculator QA; no source-verified applicability counterfactual pairs or matched intervention spans in pilot',
        'url':f'https://huggingface.co/datasets/{name}'})
pathlib.Path('results/source_audit.json').write_text(json.dumps(out,indent=2))
with pathlib.Path('data/curated_candidates.jsonl').open('w') as f:
    for x in out: f.write(json.dumps(x)+'\n')
print('Source audit saved',[(x['dataset'],x['sample_rows'],x['sample_fetch_status']) for x in out])
