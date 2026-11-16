"""Corrected staged run, reusing provable old predictions and cached activations."""
import argparse, hashlib, json, os, shutil, subprocess, time, traceback
from pathlib import Path
import torch
from care_m2_inc.io import read_json, read_jsonl, write_json, sha256_file
from .data import digest
from .engine import Engine
from .candidates import candidate_contract, score_key, recover_legacy_top
from .repair import model_spec
from .study import load, select, primary, METHODS, evaluate, summarize, followup_decisions
from .smoke import run as smoke
from .cli import save_manifest

class ReusingEngine(Engine):
    def set_stage(self, previous, out, cap, validation):
        self.previous = Path(previous); self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.journal = self.out/'forward_journal.jsonl'; self.cache = {}; self.calls = 0; self.cap = cap
        self.old_manifest = read_json(self.previous/'run_manifest.json')
        journal = self.previous/'raw/forward_journal.jsonl'
        assert sha256_file(journal) == self.old_manifest['outputs']['raw/forward_journal.jsonl']
        self.legacy = {}; self.legacy_generations = {}
        for record in read_jsonl(journal):
            assert record['checksum'] == digest(record['payload'])
            if record['kind'] == 'score': self.legacy[record['key']] = record
            if record['kind'] == 'generate': self.legacy_generations[record['key']] = record
        if self.journal.exists():
            for record in read_jsonl(self.journal):
                assert record['checksum'] == digest(record['payload'])
                self.cache[record['key']] = record['payload']
        self.unique = validation['unique_candidate_decodings']
        assert all(self.fp['files'][k] == v for k, v in validation['tokenizer_files'].items())
        write_json(self.out/'model_fingerprint.json', self.fp)
        torch.cuda.reset_peak_memory_stats()
    def score(self, prompt, options, edits=None, biases=None, cache=True):
        sp = model_spec(edits or [], biases or [])
        cs = candidate_contract(self.tokenizer, prompt, options)
        key = score_key(self.modelhash, prompt, options, sp, cs)
        if cache and key not in self.cache:
            oldkey = digest([self.modelhash, prompt, options, sp]); old = self.legacy.get(oldkey)
            if old is not None:
                recovered = recover_legacy_top(old['payload'], cs, self.unique)
                if recovered is not None:
                    recovered['reuse_provenance'] = {'source_journal': str(self.previous/'raw/forward_journal.jsonl'),
                        'source_journal_sha256': self.old_manifest['outputs']['raw/forward_journal.jsonl'],
                        'source_key': oldkey, 'source_payload_sha256': old['checksum'], 'new_model_forward_count': 0}
                    self.log('score_reuse', key, recovered); self.cache[key] = recovered
        if (not cache or key not in self.cache) and self.calls >= self.cap:
            raise RuntimeError('approved stage forward budget exhausted')
        return super().score(prompt, options, edits, biases, cache)
    def capture(self, prompt, positions=None):
        key = digest([prompt, positions]); name = 'raw/activation_' + key + '.pt'
        if name in self.old_manifest['outputs']:
            path = self.previous/name
            assert sha256_file(path) == self.old_manifest['outputs'][name]
            self.log('capture_reuse', key, {'activation_file': str(path), 'sha256': sha256_file(path), 'forward_count': 0})
            return torch.load(path, map_location='cpu', weights_only=True)
        if self.calls >= self.cap: raise RuntimeError('approved capture budget exhausted')
        return super().capture(prompt, positions)
    def generate(self, prompt, max_new_tokens=12):
        key = digest([self.modelhash, prompt, 'generate', max_new_tokens])
        if key not in self.cache and key in self.legacy_generations:
            old = self.legacy_generations[key]
            rec = {**old['payload'], 'forward_count': 0, 'cache_hit': True,
                   'reuse_provenance': {'source_key': key, 'source_payload_sha256': old['checksum'],
                                       'source_journal': str(self.previous/'raw/forward_journal.jsonl')}}
            self.log('generate_reuse', key, rec); self.cache[key] = rec
        if key not in self.cache: raise RuntimeError('new generation not authorized by repair budget')
        return self.cache[key]

def code_hashes():
    return {str(p): sha256_file(p) for p in Path('src/care_m2_v2').glob('*.py')}

def dev(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    prepared = Path(args.reuse); previous = Path(args.previous)
    provenance = read_json(prepared/'provenance.json')
    for name, h in provenance['new_files'].items(): assert sha256_file(prepared/name) == h
    validation = read_json(prepared/'tokenizer_validation.json'); budget = read_json(prepared/'budget.json')
    write_json(root/'frozen_execution.json', {'scope': 'corrected_dev_then_separately_budgeted_test',
        'source_code_sha256': code_hashes(), 'source_run': str(previous), 'reuse_manifest_sha256': sha256_file(prepared/'provenance.json'),
        'data_manifest_sha256': sha256_file(Path(args.data)/'manifest.json'), 'dev_budget': budget,
        'git': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'argv': vars(args)})
    engine = ReusingEngine(args.model, root/'S1/raw')
    try:
        started = time.time(); engine.set_stage(previous/'N1', root/'S1/raw', 6, validation)
        smoke(engine, load(args.data, 'N1', 'train'), root/'S1')
        save_manifest(engine, root/'S1', 'S1', started, 'gpu_smoke_passed')
        total = engine.calls
        for task in ('N1', 'N2', 'N3', 'N5'):
            target = root/task; shutil.copytree(prepared/'reuse'/task, target)
            engine.set_stage(previous/task, target/'raw', budget['by_task'][task]['maximum_total_dev_forwards'], validation)
            started = time.time(); axes = torch.load(target/'axes.pt', map_location='cpu', weights_only=True)
            dev_rows = load(args.data, task, 'dev'); allrows = load(args.data, task, 'train') + dev_rows
            selection, preds = select(engine, dev_rows, axes, target, allrows)
            chosen = [p for p in preds if p['beta'] == selection['selected'][p['method']]]
            summary_dir = target/'dev_analysis'; summary_dir.mkdir()
            summary = summarize(task, chosen, summary_dir, 'dev')
            by = summary['metrics']; go = True
            if task == 'N1': go = by['route']['primary'] > by['direct_bias']['primary']
            if task in ('N2', 'N3'):
                go = by['route']['primary'] > by['baseline']['primary'] and by['route']['primary'] >= by['text']['primary'] and selection['selected']['route'] > 0
                if task == 'N2': go = go and by['iia_train']['primary'] > by['baseline']['primary'] and by['iia_train']['donor_copy']/max(1, by['iia_train']['donor_copy_eligible']) < by['iia_train']['correct']/max(1, by['iia_train']['total'])
                else: go = go and by['route']['q_wrong_decision'] <= by['text']['q_wrong_decision'] and by['route']['d_unnecessary_query'] <= by['text']['d_unnecessary_query']
            write_json(target/'dev_gate.json', {'go': go, 'source_split': 'dev', 'task': task,
                'N1_test_allowed': go if task == 'N1' else None, 'N4_component_pass': go if task in ('N2', 'N3') else None})
            save_manifest(engine, target, task, started, 'corrected_dev_complete')
            total += engine.calls
        write_json(root/'dev_complete.json', {'status': 'completed', 'new_forward_count': total,
            'frozen_selections': {t: sha256_file(root/t/'dev_selection.json') for t in ('N1','N2','N3','N5')},
            'test_started': False, 'source_code_sha256': code_hashes()})
    except Exception:
        (root/'failure.log').write_text(traceback.format_exc()); raise

def test(args):
    root = Path(args.out); previous = Path(args.previous); prepared = Path(args.reuse)
    frozen = read_json(root/'dev_complete.json'); plan = read_json(root/'test_budget.json')
    for task, h in frozen['frozen_selections'].items(): assert sha256_file(root/task/'dev_selection.json') == h
    assert code_hashes() == frozen['source_code_sha256'], 'code changed after freezing selection'
    validation = read_json(prepared/'tokenizer_validation.json')
    engine = ReusingEngine(args.model, root/'test_engine_metadata')
    summaries = {}; total = 0
    try:
        for task in ('N1','N2','N3','N5'):
            parent = root/task
            if task == 'N1' and not read_json(parent/'dev_gate.json')['go']:
                write_json(parent/'test_status.json', {'status': 'not_run_gate_fail', 'forward_count': 0}); continue
            target = parent/'test'; target.mkdir(exist_ok=False)
            for name in ('axes.pt','train_fit.json','data_manifest.json','dev_selection.json'): shutil.copyfile(parent/name, target/name)
            axes = torch.load(target/'axes.pt', map_location='cpu', weights_only=True)
            selection = read_json(target/'dev_selection.json')
            engine.set_stage(previous/task, target/'raw', plan['by_task'][task]['maximum_new_forwards'], validation)
            started = time.time(); rows = load(args.data, task)
            preds = evaluate(engine, [r for r in rows if r['split'] == 'test'], axes, selection, target, rows)
            summaries[task] = summarize(task, preds, target)
            if task == 'N2':
                from .knowledge import screen, evaluate_known
                knowledge, eligibility = screen(engine, target)
                known = evaluate_known(engine, knowledge, eligibility, selection, axes, target)
                if known:
                    (target/'knowledge_analysis').mkdir(); summarize(task, known, target/'knowledge_analysis')
            if task == 'N3': followup_decisions(engine, [r for r in rows if r['split']=='test'], preds, selection, axes, target, rows)
            save_manifest(engine, target, task, started, 'corrected_test_complete'); total += engine.calls
        n4go = all(read_json(root/t/'dev_gate.json')['go'] for t in ('N2','N3'))
        if n4go: raise RuntimeError('N4 unexpectedly eligible; requires separate joint budget')
        write_json(root/'N4_status.json', {'status': 'not_run_gate_fail', 'forward_count': 0,
            'N2_dev_pass': read_json(root/'N2/dev_gate.json')['go'], 'N3_dev_pass': read_json(root/'N3/dev_gate.json')['go']})
        ps = sorted((s['primary_comparison']['two_sided_signflip_p'],t) for t,s in summaries.items() if s.get('primary_comparison'))
        adjusted = {}; last = 0.
        for i,(p,t) in enumerate(ps): last = max(last,min(1.,p*(len(ps)-i))); adjusted[t] = last
        write_json(root/'final_summary.json', {'status': 'completed_with_registered_gate_stops', 'studies': summaries,
            'holm_adjusted_p': adjusted, 'dev_new_forwards': frozen['new_forward_count'], 'test_new_forwards': total,
            'total_new_forwards': frozen['new_forward_count']+total, 'A1_forwards': 0, 'historical_experiment_reruns': 0,
            'limits': 'corrected exploratory evaluation of previously seen test cases; no fresh held-out confirmation'})
    except Exception:
        (root/'test_failure.log').write_text(traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=['dev','test'])
    for f in ('data','previous','model','reuse','out'): p.add_argument('--'+f, required=True)
    args = p.parse_args(); (dev if args.phase == 'dev' else test)(args)
