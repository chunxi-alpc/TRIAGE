"""Semantic single-turn scoring of saved frozen predictions; never regenerates answers."""
import argparse
import bisect
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import official

OUT = official.OUT / 'semantic'
OLD_SCORES = Path('/home/gcx/aster_manuscript_20261030/clinical01/official_test_units')
METHODS = ['no_js', 'ordinary', 'context', 'blind_readout', 'full', 'zero', 'fusion']
MODULE = 'experiments.aster_representation_20261101.semantic'
PAPER_ORDER = ['zero', 'fusion', 'ordinary', 'context', 'blind_readout', 'full', 'no_js']
PAPER_LABELS = ['Qwen3-8B：零干预', '标签概率融合', '普通低秩干预', '概率上下文',
                '盲表示分类器', '处置盲内部干预＋JS', '**处置盲内部干预（本文）**']
MANUSCRIPT = Path(__file__).resolve().parents[2] / 'docs/ASTER/manuscript.zh.md'
CONFIG = {'judge': 'Qwen/Qwen3-8B', 'draws': 5, 'temperature': .7, 'top_p': .8,
          'top_k': 20, 'thinking': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
          'lnds_threshold': .9, 'lnds_comparison': '>=', 'ot_reg': .05, 'ot_tau': 1.,
          'ot_alpha': 1., 'ot_beta': .2,
          'ot_cost': '(semantic_cost + .2 * position_cost) / 1.2',
          'ot_parameters_are_our_fixed_implementation': True,
          'main_paper_GPT_judge': False}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    official.write(Path(path), value)


def case_key(case):
    return case['case_id'].replace(':', '_')


def prediction_rows(case):
    path = official.OUT / 'results' / f"{case['guideline_id']}.jsonl"
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A generator may still be appending the last line.
            if row['case_id'] == case['case_id']:
                result[row['kind']] = row
    return result


def canonical(case, text):
    from experiments.aster_manuscript_clinical_20261030.official_score import LocalOfficialJudge
    return LocalOfficialJudge.canonical(None, case, text)[0]


def lnds_length(sequence):
    tails = []
    for item in sequence:
        pos = bisect.bisect_right(tails, item)
        if pos == len(tails):
            tails.append(item)
        else:
            tails[pos] = item
    return len(tails)


def compute_metrics(score):
    import numpy as np
    from experiments.aster_manuscript_clinical_20261030.official_score import dynamic_time_warping
    from ot.unbalanced import sinkhorn_knopp_unbalanced
    path = score['path_draws']['results']
    outcome = score['outcome_draws']['results']
    if len(path) != 5 or len(outcome) != 5:
        raise ValueError('Exactly five draws are required for both semantic judgments')
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 3 or min(values.shape[1:]) == 0:
        raise ValueError('Expected draw x reference x predicted matrix')
    if not np.isfinite(values).all() or np.any((values < 1) | (values > 5)):
        raise ValueError('Judge scores outside 1..5')
    c = (values.mean(axis=0) - 1.) / 4.
    differences = 1. - c
    _, dtw, _ = dynamic_time_warping(differences.tolist())
    best = np.argmax(c, axis=1)
    sequence = [int(j) for i, j in enumerate(best) if c[i, j] >= CONFIG['lnds_threshold'] - 1e-12]
    lnds = lnds_length(sequence) / c.shape[0]
    m, n = c.shape
    pos = np.abs((np.arange(m)[:, None] + .5) / m - (np.arange(n)[None, :] + .5) / n)
    cost = (differences + CONFIG['ot_beta'] * pos) / (1. + CONFIG['ot_beta'])
    plan = sinkhorn_knopp_unbalanced(np.full(m, 1./m), np.full(n, 1./n), cost,
        CONFIG['ot_reg'], CONFIG['ot_tau'], numItermax=3000, stopThr=1e-9)
    mass = float(plan.sum())
    if not np.isfinite(plan).all() or mass <= 0:
        raise ValueError('Invalid unbalanced transport solution')
    outcomes = np.asarray(outcome, dtype=float)
    if np.any((outcomes < 1) | (outcomes > 5)) or not np.isfinite(outcomes).all():
        raise ValueError('Outcome judgment outside 1..5')
    return {'DTW_PCC': float(dtw), 'Acc': float(((outcomes - 1.) / 4.).mean()),
            'OT_PCC': float((plan * cost).sum() / mass), 'LNDS_PCS': float(lnds),
            'reference_steps': m, 'predicted_steps': n}


def reusable(score, text):
    return bool(score and score.get('status') == 'complete'
        and score.get('canonical_text_private') == text
        and score.get('judge_model') == 'Qwen/Qwen3-8B'
        and score.get('expected_draws') == 5
        and len(score.get('path_draws', {}).get('results', [])) == 5
        and len(score.get('outcome_draws', {}).get('results', [])) == 5)


def find_reusable(case, text):
    folder = OUT / 'representations' / case_key(case)
    for path in sorted(folder.glob('*.json')):
        score = read_json(path)
        if reusable(score, text):
            return path, score, False
    for path in sorted((OLD_SCORES / case_key(case) / 'official_scores').glob('*.json')):
        score = read_json(path)
        if reusable(score, text):
            return path, score, True
    return None, None, False


def new_representation_path(case, text):
    folder = OUT / 'representations' / case_key(case)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        path = folder / f'{i}.json'
        existing = read_json(path)
        draw_meta = read_json(OUT / 'draws' / case_key(case) / str(i) / 'identity.json')
        if existing and existing.get('canonical_text_private') == text:
            return path
        if draw_meta and draw_meta.get('canonical_text_private') == text:
            return path
        if existing is None and draw_meta is None:
            write(OUT / 'draws' / case_key(case) / str(i) / 'identity.json',
                  {'canonical_text_private': text})
            return path
    raise RuntimeError('Unexpected number of representations for one case')


def make_judge(memory_fraction=1.):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.aster_manuscript_clinical_20261030.official_score import LocalOfficialJudge
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    tok = AutoTokenizer.from_pretrained(official.MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(official.MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    class BatchJudge(LocalOfficialJudge):
        def score(self, case, text, seed, draw_directory=None):
            from experiments.aster_manuscript_clinical_20261030 import official_score as oscore
            canonical_text, _ = self.canonical(case, text)
            record = {'id': case['case_id'], 'case_path_id': case['guideline_id'],
                'path_index': case['path_id'], 'filename': case['filename'],
                'language': case['language'], 'llm_output': canonical_text}
            parsed = oscore.parse_record(record)
            reference = oscore.pcc.resolve_reference(parsed, self.refs[case['source_split']])
            ref_steps = oscore.pcc.load_reference_steps(reference)
            cand_steps = oscore.pcc.get_candidate_steps(parsed)
            path_prompt = oscore.pcc.build_prompt(case['language'], ref_steps, cand_steps)
            outcome_prompt = oscore.acc.render_template(
                oscore.acc.TEMPLATE_ZH if case['language'] == 'zh' else oscore.acc.TEMPLATE_EN,
                {'Reference_Conclusion': oscore.acc.load_reference_conclusion(reference),
                 'Candidate_Conclusion': oscore.acc.candidate_conclusion(parsed)})
            self.validators = {
                path_prompt: ('path', lambda x: oscore.pcc.parse_pairwise_score_matrix(x, ref_steps, cand_steps)),
                outcome_prompt: ('outcome', oscore.acc.parse_accuracy_score)}
            # Large matrices need more than the legacy 8192-token cap.
            self.token_limits = {path_prompt: max(8192, 14 * len(ref_steps) * len(cand_steps) + 256),
                                 outcome_prompt: 128}
            self.draw_directory = Path(draw_directory)
            self.offsets = {}
            self.sampling_seed = seed
            result = super().score(case, text, seed)
            result.update(transport='Batched local Transformers SDPA, 5 samples per prompt',
                semantic_protocol=CONFIG, draw_directory=str(self.draw_directory),
                maximum_generated_token_limits={'path': self.token_limits[path_prompt], 'outcome': 128})
            return result

        def chat(self, client, args, prompt):
            stage, validator = self.validators[prompt]
            path = self.draw_directory / (stage + '.json')
            state = read_json(path, {'prompt': prompt, 'responses': [], 'attempts': 0, 'invalid_draws': []})
            if state['prompt'] != prompt:
                raise ValueError('Saved judgment prompt differs')
            needed = 5 - len(state['responses'])
            failures = 0
            batch_limit = 5
            while needed:
                count = min(needed, batch_limit)
                rendered = self.tok.apply_chat_template([{'role':'system','content':prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = self.tok(rendered, return_tensors='pt', add_special_tokens=False).to(self.model.device)
                limit = self.token_limits[prompt]
                if inputs['input_ids'].shape[1] + limit > self.model.config.max_position_embeddings:
                    raise ValueError('Judge context budget exceeded')
                seed = self.sampling_seed + (10000 if stage == 'outcome' else 0) + state['attempts']
                torch.manual_seed(seed)
                started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        output = self.model.generate(**inputs, max_new_tokens=limit, num_return_sequences=count,
                            do_sample=True, temperature=.7, top_p=.8, top_k=20, use_cache=True,
                            pad_token_id=self.tok.eos_token_id)
                except torch.cuda.OutOfMemoryError:
                    del inputs
                    gc.collect(); torch.cuda.empty_cache()
                    if count == 1:
                        raise
                    batch_limit = max(1, count // 2)
                    continue
                sequences = output[:, inputs['input_ids'].shape[1]:]
                responses = self.tok.batch_decode(sequences, skip_special_tokens=True)
                self.calls.append({'stage':stage, 'batch':count, 'input_tokens':int(inputs['input_ids'].shape[1]),
                    'generated_tokens':int(sequences.shape[1]), 'seconds':time.perf_counter()-started,
                    'seed':seed, 'max_new_tokens':limit})
                state['attempts'] += count
                valid = 0
                for response in responses:
                    try:
                        validator(response)
                        state['responses'].append(response); valid += 1
                    except Exception as exc:
                        state['invalid_draws'].append({'text': response, 'error': str(exc)})
                del output, sequences, inputs
                write(path, state)
                needed = 5 - len(state['responses'])
                if valid < count:
                    failures += 1
                if failures >= 3 and needed:
                    raise ValueError(f'{stage}: failed to complete five valid draws after retries')
            offset = self.offsets.get(prompt, 0)
            self.offsets[prompt] = offset + 1
            return state['responses'][offset]

    return BatchJudge(model, tok)


def unit_finished(case, kind, retry_failed=False):
    saved=read_json(OUT/'units'/case_key(case)/(kind+'.json'))
    return bool(saved and (saved.get('status')=='complete' or not retry_failed))


def score_case(case, rows, kinds, judge_factory, retry_failed=False):
    import torch
    outputs = []
    folder = OUT / 'units' / case_key(case)
    for kind in kinds:
        unit = folder / f'{kind}.json'
        prior = read_json(unit)
        if prior and (prior.get('status') == 'complete' or not retry_failed):
            outputs.append(prior)
            continue
        row = rows.get(kind)
        if row is None:
            continue
        started = time.time()
        result = {'case_id':case['case_id'], 'kind':kind, 'source_id':case['source_id'],
                  'guideline_id':case['guideline_id'], 'language':case['language']}
        try:
            if row['status'] != 'complete':
                raise ValueError('Saved prediction failed')
            text = canonical(case, row['result']['text'])
            reused_path, score, old = find_reusable(case, text)
            was_reused = score is not None
            if score is None:
                rep_path = new_representation_path(case, text)
                judge = judge_factory()
                torch.cuda.reset_peak_memory_stats()
                draw_dir = OUT / 'draws' / case_key(case) / rep_path.stem
                case_index = case.get('_semantic_index', 0)
                score = judge.score(case, row['result']['text'], 20261102 + case_index * 100,
                                    draw_directory=draw_dir)
                score['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
                write(rep_path, score)
                reused_path = rep_path
            elif old:
                rep_path = new_representation_path(case, text)
                score = dict(score, reused_from=str(reused_path))
                write(rep_path, score)
                reused_path = rep_path
            if score['status'] != 'complete':
                raise ValueError('Semantic judge did not produce complete scores')
            result.update(status='complete', metrics=compute_metrics(score),
                representation=str(reused_path), reused=was_reused,
                peak_gb=score.get('peak_gb'), judge_seconds=score.get('seconds'))
        except Exception as exc:
            result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        result['seconds'] = time.time() - started
        write(unit, result)
        outputs.append(result)
    return outputs


def manifest():
    path = OUT / 'manifest.json'
    previous = read_json(path)
    if previous is not None:
        return previous
    cases = official.all_cases()
    result = {'created_at':time.time(), 'cases_file':str(official.CASES),
        'cases':len(cases), 'methods':METHODS, 'expected_units':len(cases)*len(METHODS),
        'configuration':CONFIG, 'predictions':str(official.OUT/'results'),
        'reuse':'Identical canonical prediction within the same case reuses all five judge draws.',
        'failure_policy':'Failures remain registered; report valid-only means and failure-penalized means.',
        'pilot_case_ids':read_json(official.OUT/'manifest.json')['pilot_ids']}
    write(path, result)
    return result


def strict_totals(cases):
    """Read saved strict scores once; duplicate appended rows use the latest record."""
    expected = {c['case_id'] for c in cases}
    rows = {k:{} for k in METHODS}
    for path in sorted((official.OUT/'results').glob('*.jsonl')):
        with path.open() as handle:
            for line in handle:
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('case_id') not in expected or row.get('kind') not in rows:
                    continue
                rows[row['kind']][row['case_id']] = bool(
                    row.get('status')=='complete' and row.get('score',{}).get('exact_path'))
    return {k:{'recorded':len(v), 'registered':len(cases), 'exact_path_correct':sum(v.values()),
               'PathAcc':sum(v.values())/len(cases) if len(v)==len(cases) else None}
            for k,v in rows.items()}


def update_manuscript(summary, manuscript=MANUSCRIPT, expected_cases=7458):
    """Fill complete columns by exact method label; retain external table rows."""
    import fcntl
    manuscript=Path(manuscript)
    if not manuscript.exists():
        return {'status':'not_found','updated_methods':[]}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'manuscript.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        original=manuscript.read_text()
        text=original
        updated=[]
        for section,metrics in [('MAIN',['PathAcc','DTW_PCC','Acc']),
                                ('PATH',['DTW_PCC','OT_PCC','LNDS_PCS'])]:
            start=f'<!-- ASTER_FINAL_{section}_START -->'
            end=f'<!-- ASTER_FINAL_{section}_END -->'
            if text.count(start)!=1 or text.count(end)!=1:
                return {'status':'markers_missing','updated_methods':[]}
            prefix,body=text.split(start,1)
            current,suffix=body.split(end,1)
            lines=current.split('\n')
            for kind,label in zip(PAPER_ORDER,PAPER_LABELS):
                matches=[i for i,line in enumerate(lines)
                         if line.startswith('|') and line.split('|')[1].strip()==label]
                if len(matches)!=1:
                    return {'status':'method_row_missing_or_duplicated','updated_methods':[]}
                i=matches[0]
                cells=[v.strip() for v in lines[i].split('|')[2:-1]]
                if len(cells)!=len(metrics):
                    return {'status':'unexpected_columns','updated_methods':[]}
                record=summary['methods'][kind]
                strict=summary['strict'][kind]
                strict_complete=strict['registered']==strict['recorded']==expected_cases
                semantic_complete=(strict_complete and record['registered']==record['recorded']==expected_cases
                                   and record['status']=='complete')
                values=dict(record['all_registered'],PathAcc=strict['PathAcc'])
                changed=False
                for j,metric in enumerate(metrics):
                    complete=strict_complete if metric=='PathAcc' else semantic_complete
                    if complete and values.get(metric) is not None:
                        value=f'{100*values[metric]:.2f}'
                        cells[j]='**'+value+'**' if kind=='no_js' else value
                        changed=True
                if changed:
                    lines[i]='| '+label+' | '+' | '.join(cells)+' |'
                    updated.append(kind)
            text=prefix+start+'\n'.join(lines)+end+suffix
        if text!=original:
            temp=manuscript.with_name(manuscript.name+f'.{os.getpid()}.tmp')
            temp.write_text(text)
            temp.replace(manuscript)
        return {'status':'updated' if text!=original else 'unchanged',
                'updated_methods':list(dict.fromkeys(updated))}


def aggregate():
    import numpy as np
    cases = official.all_cases()
    summary = {'status':'incomplete','configuration':CONFIG,'methods':{},'strict':strict_totals(cases)}
    penalties = {'DTW_PCC':1., 'Acc':0., 'OT_PCC':1., 'LNDS_PCS':0.}
    for kind in METHODS:
        rows = [read_json(OUT/'units'/case_key(c)/(kind+'.json')) for c in cases]
        saved = [r for r in rows if r is not None]
        valid = [r for r in saved if r['status']=='complete']
        complete = len(saved)==len(cases)
        summary['methods'][kind] = {'status':'complete' if complete else 'incomplete',
            'registered':len(cases), 'recorded':len(saved), 'scored':len(valid),
            'scoring_failures':len(saved)-len(valid),
            'valid_only':{k:float(np.mean([r['metrics'][k] for r in valid])) if valid else None for k in penalties},
            'all_registered':{k:float(sum(r['metrics'][k] if r and r['status']=='complete' else v
                                       for r in rows)/len(cases)) if complete else None for k,v in penalties.items()}}
    if all(v['status']=='complete' for v in summary['methods'].values()):
        summary['status']='complete'
    summary['manuscript']=update_manuscript(summary)
    write(OUT/'summary.json',summary)
    lines=['# Frozen predictions: local Qwen semantic evaluation','',
        'Five sampled judgments per pathway and outcome. OT fixed implementation reg=.05, tau=1; paper cost formula. LNDS threshold >=.9.','',
        '| Method | Recorded / registered | Scored | Failed | DTW-PCC (%) | Acc (%) | OT-PCC (%) | LNDS-PCS (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,record in summary['methods'].items():
        scores=record['all_registered']
        values=['—' if scores[k] is None else f"{100*scores[k]:.2f}" for k in penalties]
        lines.append(f"| {kind} | {record['recorded']}/{record['registered']} | {record['scored']} | {record['scoring_failures']} | "+' | '.join(values)+' |')
    lines += ['', 'Only fully covered methods receive table scores. Scoring failures use the registered conservative penalties; valid-only values are in summary.json. These are structured-path outputs, not the original paper free-text input setting.']
    (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    return summary


def pid_live(pid, module=MODULE):
    try:
        return module.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def claim(case, gpu):
    path=OUT/'claims'/(case_key(case)+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(2):
        try:
            with path.open('x') as handle:
                json.dump({'pid':os.getpid(),'gpu':gpu},handle)
            return path
        except FileExistsError:
            owner=read_json(path)
            if owner and not pid_live(owner['pid']):
                path.unlink(missing_ok=True)
            else:
                return None
    return None


def generation_running(gpu=None):
    info=read_json(official.OUT/'launch.json',{})
    workers=info.get('workers',{})
    pids=workers.values() if gpu is None else [workers.get(str(gpu),workers.get(gpu))]
    return any(pid is not None and official.live(pid) for pid in pids)


def worker(args):
    config=manifest()
    cases=official.all_cases()
    for i,c in enumerate(cases):
        c['_semantic_index']=i
    if args.wait_for_generation:
        while generation_running(args.worker):
            time.sleep(20)
    holder=[]
    def judge_factory():
        if not holder:
            holder.append(make_judge(args.memory_fraction))
        return holder[0]
    started=time.time()
    status=OUT/(f'pilot_gpu{args.worker}.json' if args.pilot else f'worker_gpu{args.worker}.json')
    if args.pilot:
        selected=args.case_id or (config['pilot_case_ids'][:args.case_count] if args.case_count > 1 else [config['pilot_case_ids'][args.worker]])
        report=[]
        for c in cases:
            if c['case_id'] in selected:
                report.extend(score_case(c,prediction_rows(c),['no_js'],judge_factory,args.retry_failed))
        write(status,{'status':'complete' if len(report)==len(selected) and all(r['status']=='complete' for r in report) else 'failed','gpu':args.worker,'elapsed':time.time()-started,
            'case_ids':selected,'results':report,'memory_fraction':args.memory_fraction,
            'model_loaded':bool(holder),'configuration':CONFIG})
        print(json.dumps({'pilot':str(status),'elapsed':time.time()-started,'model_loaded':bool(holder)}),flush=True)
        return
    finished=set()
    while True:
        did_work=False
        for phase,kinds in [('primary',['no_js']),('controls',METHODS[1:])]:
            for case in cases:
                key=(phase,case['case_id'])
                if key in finished:
                    continue
                if all(unit_finished(case,k,args.retry_failed) for k in kinds):
                    finished.add(key);continue
                rows=prediction_rows(case)
                todo=[k for k in kinds if k in rows and not unit_finished(case,k,args.retry_failed)]
                if not todo:
                    continue
                lock=claim(case,args.worker)
                if lock is None:
                    continue
                try:
                    write(status,{'status':'running','gpu':args.worker,'pid':os.getpid(),
                        'phase':phase,'case_id':case['case_id'],'elapsed':time.time()-started})
                    score_case(case,rows,todo,judge_factory,args.retry_failed)
                    did_work=True
                    if all(k in rows for k in kinds):
                        finished.add(key)
                finally:
                    lock.unlink(missing_ok=True)
            if phase=='primary' and did_work:
                break
        if not did_work:
            if generation_running() or any(pid_live((read_json(p,{}) or {}).get('pid',0)) for p in (OUT/'claims').glob('*.json')):
                time.sleep(20)
                continue
            break
    write(status,{'status':'complete','gpu':args.worker,'elapsed':time.time()-started})
    aggregate()


def launch(args):
    manifest()
    info=read_json(OUT/'launch.json',{})
    if any(pid_live(pid) for pid in info.get('workers',{}).values()):
        raise RuntimeError('Semantic workers are already running')
    workers={}
    for gpu in [0,1]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false')
        log=(OUT/f'worker_gpu{gpu}.log').open('a')
        command=[sys.executable,'-m',MODULE,'--worker',str(gpu),'--memory-fraction',str(args.memory_fraction)]
        if args.wait_for_generation:
            command.append('--wait-for-generation')
        if args.retry_failed:
            command.append('--retry-failed')
        workers[gpu]=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close()
    info={'status':'running','controller_pid':os.getpid(),'workers':{g:w.pid for g,w in workers.items()},
          'gpus':[0,1],'started_at':time.time(),'wait_for_generation':args.wait_for_generation,'configuration':CONFIG}
    write(OUT/'launch.json',info)
    codes={g:w.wait() for g,w in workers.items()}
    info.update(status='complete' if all(c==0 for c in codes.values()) else 'failed',exit_codes=codes)
    write(OUT/'launch.json',info)
    aggregate()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker','--gpu',dest='worker',type=int,choices=[0,1])
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--case-id',action='append')
    p.add_argument('--case-count',type=int,choices=[1,2],default=1)
    p.add_argument('--memory-fraction',type=float,default=1.)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--wait-for-generation',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.prepare:
        print(json.dumps(manifest(),ensure_ascii=False))
    elif a.worker is not None:
        worker(a)
    elif a.launch:
        launch(a)
    elif a.aggregate:
        print(json.dumps(aggregate(),ensure_ascii=False))


if __name__=='__main__':
    main()
