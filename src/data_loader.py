"""Read original TRIAGE dataset files without annotating or generating answers.

Example: next(load_dataset('MedGUIDE', split='train'))
All records retain their unchanged original data in ``raw``.
"""
from pathlib import Path
import ast
import csv
import json

ROOT=Path(__file__).resolve().parents[1]

def _pick(raw,keys,default=None):
    for key in keys:
        if key in raw and raw[key] is not None and raw[key]!='':
            return raw[key]
    return default

def _decoded(value):
    if not isinstance(value,str):
        return value
    for parser in (json.loads,ast.literal_eval):
        try:
            return parser(value)
        except (ValueError,SyntaxError,TypeError):
            pass
    return value

def _json_records(file):
    with Path(file).open(encoding='utf-8') as handle:
        if str(file).endswith('.jsonl'):
            for line in handle:
                if line.strip():
                    yield json.loads(line)
            return
        data=json.load(handle)
        if isinstance(data,list):
            yield from data
        elif isinstance(data,dict):
            for key in ['questions','data','examples','records']:
                if isinstance(data.get(key),list):
                    yield from data[key]
                    return
            for key,value in data.items():
                if isinstance(value,dict):
                    yield {'_record_id':key,**value}

def _records(file):
    if file.suffix in {'.json','.jsonl'}:
        yield from _json_records(file)
    elif file.suffix=='.csv':
        with file.open(encoding='utf-8-sig',newline='') as handle:
            yield from csv.DictReader(handle)
    elif file.suffix in {'.parquet','.pqt'}:
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            yield from batch.to_pylist()

def _standard(name,raw):
    if name=='MedGUIDE':
        question=_pick(raw,['prompt','question','profile'])
        answer=_pick(raw,['answer_text','answer'])
    elif name=='MedCalc-Bench':
        question=raw['Question']
        answer=raw['Ground Truth Answer']
    elif name=='ConFiQA':
        question=raw['question']
        # Preserve both author-provided answers; evaluation should choose the matching context.
        answer={'original':raw.get('orig_answer'),'counterfactual':raw.get('cf_answer')}
    elif name=='MedChangeQA':
        question=raw['Question']
        answer=raw['Newest Label']
    elif name=='ClashEval':
        question=raw['question']
        answer={'original':raw.get('answer_original'),'modified':raw.get('answer_mod')}
    else:
        question=_pick(raw,['question','Question','Statement','statement','prompt','input','case','vignette','text'])
        answer=_pick(raw,['answer','Answer','answers','reference_answer','ground_truth','label','Label','gold_label'])
    context=_pick(raw,['patient_vignette','Patient Note','context','contexts','profile','premise'])
    if name=='ConFiQA':
        context={'original':raw.get('orig_context'),'counterfactual':raw.get('cf_context')}
    elif name=='ClashEval':
        context={'original':raw.get('context_original'),'modified':raw.get('context_mod')}
    return {'dataset':name,'id':_pick(raw,['instance_id','id','_record_id','Row Number','qid','Unnamed: 0']),
            'question':question,'reference_answer':answer,
            'context':context,
            'options':raw.get('options'),'raw':raw}

def _mega(path,split):
    folder=path/('train_val' if split in {'train','validation','train_val'} else 'test')
    refs={(r['filename'],str(r['path_id'])):r for r in _json_records(folder/'gt_paths.jsonl')}
    trees={r['filename']:r for r in _json_records(folder/'decision_trees.jsonl')}
    for row in _json_records(folder/'vignette_cases.jsonl'):
        ref=refs.get((row['filename'],str(row['path_id'])),{})
        yield {'dataset':'MEGA-CDP','id':f"{row['filename']}:{row['path_id']}",
               'question':_pick(row,['vignette_cases','cases','vignettes','case','vignette','text']),
               'reference_answer':_pick(ref,['output_text','gt_paths','paths','text']),
               'context':trees.get(row['filename'],{}).get('text'),
               'options':None,'raw':{'cases':row,'reference':ref,'decision_tree':trees.get(row['filename'])}}

def _nli(name,path,split):
    selected={'validation':'dev','development':'dev'}.get(split,split) or 'train'
    direct=path/f'{selected}.json'
    files=[direct] if direct.exists() else sorted(path.rglob(f'{selected}.json'))
    if not files:
        raise FileNotFoundError(f'{name}: no original {selected}.json')
    trials={f.stem:f for f in path.rglob('NCT*.json')}
    cache={}
    gold_file=path/f'gold_{selected}.json'
    gold=json.loads(gold_file.read_text()) if gold_file.exists() else {}
    for file in files:
        for record in _json_records(file):
            record_id=record.get('_record_id')
            raw={k:v for k,v in record.items() if k!='_record_id'}
            context={}
            for key in ['Primary_id','Secondary_id']:
                trial_id=raw.get(key)
                if trial_id:
                    if trial_id not in trials:
                        raise FileNotFoundError(f'{name}: missing original clinical trial {trial_id}')
                    if trial_id not in cache:
                        cache[trial_id]=json.loads(trials[trial_id].read_text())
                    context[trial_id]=cache[trial_id]
            reference=raw.get('Label')
            reference_source='original record.Label' if reference is not None else None
            official_gold=gold.get(record_id,{})
            if reference is None:
                reference=official_gold.get('Label') if isinstance(official_gold,dict) else official_gold
                if reference is not None:
                    reference_source=str(gold_file)+':'+str(record_id)
            yield {'dataset':name,'id':record_id,'question':raw['Statement'],
                   'reference_answer':reference,'reference_available':reference is not None,
                   'reference_source':reference_source,'official_gold':official_gold or None,
                   'context':context,'options':None,'raw':raw}

def load_dataset(name,split=None,config_path=None):
    """Yield a shared schema while keeping every original field in ``raw``.

    MEGA-CDP joins original cases and references using filename and path_id.
    Question and reference strings remain unchanged; no new question is fabricated.
    """
    config=json.loads(Path(config_path or ROOT/'configs'/'datasets.json').read_text())
    if name not in config:
        raise KeyError(f'Unknown dataset {name}; available: {list(config)}')
    entry=config[name]
    if entry.get('status')!='downloaded':
        raise RuntimeError(f'{name}: {entry.get("status")}: {entry.get("reason", "download incomplete")}')
    path=Path(entry['path'])
    split=split or entry.get('default_split')
    if name=='MEGA-CDP':
        yield from _mega(path,split)
        return
    if name in {'NLI4CT','NLI4CT-2024'}:
        yield from _nli(name,path,split)
        return
    if entry.get('split_files') and split:
        if split not in entry['split_files']:
            raise ValueError(f'{name}: unknown split {split}; available: {list(entry["split_files"])}')
        candidates=[path/f for f in entry['split_files'][split]]
    elif entry.get('data_files'):
        candidates=[path/f for f in entry['data_files']]
    elif name=='NLI4CT':
        candidates=list(path.rglob(f'{split or "train"}.json'))
    elif name=='ConFiQA':
        candidates=sorted((path/'ConFiQA').glob('ConFiQA-*.json'))
        if split:
            candidates=[f for f in candidates if f.stem.lower()==('ConFiQA-'+split).lower()]
    elif name=='MedCalc-Bench':
        candidates=list(path.glob(f'{split or "test"}_*.csv'))
    else:
        candidates=sorted(f for f in path.rglob('*') if f.suffix in {'.json','.jsonl','.csv','.parquet','.pqt'})
        if split:
            preferred=[f for f in candidates if split in str(f.relative_to(path))]
            if preferred:
                candidates=preferred
    if not candidates:
        raise FileNotFoundError(f'{name}: no original files found for split {split}')
    for file in candidates:
        for raw in _records(file):
            if isinstance(raw,dict):
                yield _standard(name,raw)
