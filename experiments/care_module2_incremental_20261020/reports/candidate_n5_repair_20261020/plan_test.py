"""CPU-only expansion of corrected test requests after dev parameters are frozen."""
import hashlib, json, os, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from care_m2_inc.io import read_json, read_jsonl, write_json, write_jsonl, sha256_file
from care_m2_v2.data import digest, oracle, choices
from care_m2_v2.candidates import candidate_contract, score_key, recover_legacy_top
from care_m2_v2.repair import CachedActivations, model_spec
from care_m2_v2.study import load, spec, controls
from care_m2_v2.knowledge import FACTS, match

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root = Path(sys.argv[1]); report = Path(__file__).parent
previous = Path('experiments/care_module2_incremental_20261020/runs/v2_20261020T134304Z')
data = Path('experiments/care_module2_incremental_20261020/data_v2_ready')
frozen = read_json(root/'dev_complete.json')
validation = read_json(report/'offline_verified/tokenizer_validation.json')
tok = AutoTokenizer.from_pretrained('/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base', local_files_only=True)
all_missing = []; budget = {}
for task in ('N1','N2','N3','N5'):
    if task == 'N1' and not read_json(root/task/'dev_gate.json')['go']:
        budget[task] = {'maximum_new_forwards': 0, 'status': 'not_run_gate_fail'}; continue
    assert sha256_file(root/task/'dev_selection.json') == frozen['frozen_selections'][task]
    selected = read_json(root/task/'dev_selection.json')['selected']
    axes = torch.load(root/task/'axes.pt', map_location='cpu', weights_only=True)
    rows = load(data, task); test = [r for r in rows if r['split']=='test']
    cpu = CachedActivations(tok, previous/task, [r for r in rows if r['split']=='train'])
    modelhash = digest(read_json(previous/task/'raw/model_fingerprint.json')['files'])
    legacy = {}; generations = {}
    for record in read_jsonl(previous/task/'raw/forward_journal.jsonl'):
        assert record['checksum'] == digest(record['payload'])
        if record['kind'] == 'score': legacy[record['key']] = record['payload']
        if record['kind'] == 'generate': generations[record['key']] = record['payload']
    missing = {}; recovered = set(); seen = set(); parts = {}
    def request(prompt, options, edits, biases, metadata):
        sp = model_spec(edits, biases); candidates = candidate_contract(tok, prompt, options)
        key = score_key(modelhash, prompt, options, sp, candidates)
        old = legacy.get(digest([modelhash, prompt, options, sp]))
        result = recover_legacy_top(old, candidates, validation['unique_candidate_decodings']) if old else None
        if result is not None: recovered.add(key)
        elif key not in seen: missing[key] = {'key': key, 'task': task, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **metadata}
        seen.add(key)
        return result
    def row_request(r, c, method, beta, phase):
        if method in ('oracle','any_unknown'): return
        if method.startswith('iia_'): assert beta == 0, 'nonzero IIA requires donor-specific budget'
        prompt, edits, biases = spec(cpu, r, c, method, beta, axes)
        return request(prompt, r['options'], edits, biases, {'case_id':r['case_id'],'control':c,'method':method,'beta':beta,'phase':phase})
    for r in test:
        for c in controls(r):
            for method,beta in selected.items(): row_request(r,c,method,beta,'main_test')
    parts['main_test'] = len(missing)
    if task == 'N2':
        start = len(missing); known = []; screening_complete = True
        for i, (domain, subject, q1, q2, answer, wrong) in enumerate(FACTS):
            good = []
            for q in (q1,q2):
                key = digest([modelhash,q+'\nGive only a short factual answer.\nAnswer:','generate',12])
                assert key in generations, 'missing original closed-book generation; no automatic rerun'
                good.append(match(generations[key]['text'],answer))
            for rot in (0,1):
                prompt,opts = choices(q1,[answer,wrong],rot)
                result = request(prompt,opts,[],[],{'case_id':f'knowledge_{i}','method':'closed_book_order','beta':0,'phase':'screening'})
                if result is None: screening_complete=False; good.append(True)
                else: good.append(opts[result['prediction']]==answer)
            if all(good): known.append(i)
        parts['screening'] = len(missing)-start; start = len(missing)
        oldrows = read_jsonl(previous/task/'knowledge_receivers.jsonl')
        eligible = len(known)>=8 and len({FACTS[i][0] for i in known})>=4
        for i,r in enumerate(oldrows):
            if r['split']!='test' or (screening_complete and (not eligible or i not in known)): continue
            for source in ('C','P'):
                for method in ('baseline','text','direct_bias','mean','route','random'):
                    row_request(r,{'source':source},method,selected[method],'knowledge_test')
        parts['knowledge_test'] = len(missing)-start
        parts['closed_book_generations_reused'] = 48
    if task == 'N3':
        start = len(missing)
        # Upper bound over every permitted follow-up question, independent of test
        # success. Actual evaluation only follows the chosen question.
        for r in test:
            if r['controls']['DQ']: continue
            for attribute in r['valid_attributes']:
                for value in (False,True):
                    values = {**r['values'],attribute:value}; possible,_ = oracle(r['ast'],values)
                    if len(possible)!=1: continue
                    follow = {**r,'prompt':r['prompt'].replace(attribute+'=unknown',attribute+'='+str(value).lower()), 'controls':{'DQ':1}}
                    for method,beta in selected.items(): row_request(follow,{'DQ':1},method,beta,'followup_upper_bound')
        parts['followup_upper_bound'] = len(missing)-start
    budget[task] = {'maximum_new_forwards':len(missing),'reused_unique_scores':len(recovered),'parts':parts,
                    'selected':selected,'test_cases':len(test),'new_training_forwards':0}
    all_missing.extend(missing.values())
write_jsonl(root/'planned_missing_test_requests.jsonl',all_missing)
write_json(root/'test_budget.json',{'by_task':budget,'maximum_new_forwards':sum(x['maximum_new_forwards'] for x in budget.values()),
    'selection_hashes':frozen['frozen_selections'],'GPU':0,'gpu_count':1,'reserve_GiB':10,'wall_time_budget_minutes':10,
    'new_A1_forwards':0,'new_free_generation_forwards':0,'scope':'corrected frozen test plus registered N3 follow-up and N2 closed-book reuse',
    'N4':'not_run_gate_fail' if not all(read_json(root/t/'dev_gate.json')['go'] for t in ('N2','N3')) else 'requires_separate_budget'})
print(json.dumps(read_json(root/'test_budget.json'),indent=2))
