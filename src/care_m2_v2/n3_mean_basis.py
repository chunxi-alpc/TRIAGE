"""N3-only mean-preserving rank-2 basis and bounded development evaluation."""
import argparse, copy, hashlib, json, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from care_m2_inc.io import read_json, read_jsonl, write_json, write_jsonl, sha256_file
from .data import BASE, digest
from .engine import Engine, LAYER, SEED
from .candidates import SCORING_CONTRACT, candidate_contract, score_key
from .repair import model_spec
from .study import load, runone, primary, summarize

GRID = [0., .5, 1.]
BUDGET = 48

def mean_preserving_basis(differences, mean):
    """First column is the mean; second maximizes remaining centered variance."""
    d = differences.detach().cpu().double(); m = mean.detach().cpu().double()
    if d.ndim != 2 or m.ndim != 1 or d.shape[1] != m.numel() or min(d.shape) < 2:
        raise ValueError('paired differences must be a nontrivial matrix matching the mean')
    if not torch.isfinite(d).all() or not torch.isfinite(m).all():
        raise ValueError('nonfinite training tensor')
    if m.norm() < 1e-8: raise ValueError('mean direction is undefined')
    if not torch.allclose(d.mean(0), m, atol=1e-5, rtol=1e-5):
        raise ValueError('mean does not match the paired training differences')
    first = m / m.norm()
    centered = d - d.mean(0)
    orthogonal = centered - (centered @ first)[:, None] * first[None, :]
    _, values, vh = torch.linalg.svd(orthogonal, full_matrices=False)
    if values[0] < 1e-8: raise ValueError('no independent second training direction')
    second = vh[0] - torch.dot(vh[0], first) * first
    second /= second.norm()
    if second[second.abs().argmax()] < 0: second = -second
    return torch.stack([first, second], dim=1).float(), values.float()

def code_hashes():
    return {str(p): sha256_file(p) for p in Path('src/care_m2_v2').glob('*.py')}

def snapshot(previous):
    return {str(p): sha256_file(p) for p in Path(previous).rglob('*') if p.is_file()}

def prepare(args):
    torch.set_num_threads(2)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    prior = Path(args.previous)/'N3'; manifest = read_json(prior/'run_manifest.json')
    source = {}
    for name in ('axes.pt','train_fit.json','dev_predictions.jsonl','dev_selection.json','raw/forward_journal.jsonl','raw/model_fingerprint.json'):
        path = prior/name; h = sha256_file(path)
        assert h == manifest['outputs'][name], ('source changed', name)
        source[str(path)] = h
    old_fit = read_json(prior/'train_fit.json'); assert old_fit['task']=='N3' and old_fit['split']=='train'
    train_rows = load(args.data, 'N3', 'train')
    assert set(old_fit['case_ids']) == {r['case_id'] for r in train_rows}
    assert old_fit['input_hash'] == digest(train_rows)
    axes = torch.load(prior/'axes.pt', map_location='cpu', weights_only=True)
    old = axes['DQ']; basis, values = mean_preserving_basis(old['differences'], old['mean'])
    new = copy.deepcopy(axes); new['DQ']['basis'] = basis
    # Existing singular_values still describe the unchanged centered differences.
    # The new basis is defined explicitly, not relabelled as two leading PCs.
    new['DQ']['orthogonal_residual_singular_values'] = values
    torch.save(new, root/'axes.pt')
    rows = load(args.data, 'N3', 'dev'); assert len(rows)==16
    predictions = read_jsonl(prior/'dev_predictions.jsonl')
    selection = read_json(prior/'dev_selection.json')['selected']
    chosen = [p for p in predictions if p['method'] in ('baseline','mean','text','route') and p['beta']==selection[p['method']]]
    assert len(chosen)==64
    write_jsonl(root/'reused_comparisons.jsonl', chosen)
    bycase = {r['case_id']:r for r in rows}; journal = {}
    for rec in read_jsonl(prior/'raw/forward_journal.jsonl'):
        assert rec['checksum']==digest(rec['payload'])
        if rec['kind'] in ('score','score_reuse'): journal[rec['key']] = rec
    seeds = []
    for p in chosen:
        if p['method']!='baseline': continue
        r = bycase[p['case_id']]
        key = score_key(p['model_hash'], r['prompt'], r['options'], p['spec'], p['candidate_tokens'])
        rec = journal[key]; payload = rec['payload']
        assert payload['scoring_contract']==SCORING_CONTRACT and payload['prediction']==p['prediction']
        payload = {**payload, 'forward_count':0, 'cache_hit':True, 'basis_study_reuse':{
            'source_journal':str(prior/'raw/forward_journal.jsonl'), 'source_key':key,
            'source_payload_sha256':rec['checksum'], 'source_journal_sha256':source[str(prior/'raw/forward_journal.jsonl')]}}
        seeds.append({'kind':'score_reuse','key':key,'payload':payload,'checksum':digest(payload)})
    write_jsonl(root/'baseline_score_cache.jsonl', seeds)
    mean = old['mean']; identity = basis.T@basis
    fit = {'task':'N3','split':'train','source_train_fit_sha256':source[str(prior/'train_fit.json')],
        'source_axes_sha256':source[str(prior/'axes.pt')], 'axes_sha256':sha256_file(root/'axes.pt'),
        'case_ids':old_fit['case_ids'], 'input_hash':old_fit['input_hash'], 'layer':LAYER, 'rank':2,
        'basis_rule':'unit paired mean; first PC of centered differences orthogonal to that mean',
        'centroids_unchanged':True,'differences_unchanged':True,'operator_unchanged':True,
        'old_mean_length_retained':float((old['basis']@(old['basis'].T@mean)).norm()/mean.norm()),
        'new_mean_length_retained':float((basis@(basis.T@mean)).norm()/mean.norm()),
        'mean_projection_error':float((basis@(basis.T@mean)-mean).norm()),
        'orthogonality_max_error':float((identity-torch.eye(2)).abs().max()),
        'orthogonal_residual_singular_values':values.tolist(),
        'original_singular_values_description':'unchanged centered paired-difference singular values, not scores of the new basis',
        'new_training_forwards':0,'uses_dev_or_test_to_fit':False}
    assert fit['mean_projection_error'] < 1e-4 and fit['orthogonality_max_error'] < 1e-5
    write_json(root/'train_fit.json', fit)
    shutil.copyfile(Path(args.data)/'manifest.json', root/'data_manifest.json')
    protected = snapshot(args.previous)
    write_json(root/'protected_source_sha256.json', protected)
    frozen = {'task':'N3','split':'dev','variant':'mean_preserving_rank2','grid':GRID,'rank':2,'layer':LAYER,'seed':SEED,
        'selection_rule':'maximize family-balanced development accuracy; ties smaller beta',
        'max_new_forwards':BUDGET,'main_grid_max':32,'selected_random_max':16,'test_forwards':0,'training_forwards':0,
        'source_run':args.previous, 'source_files':source,'source_code_sha256':code_hashes(),
        'model_hash':digest(read_json(prior/'raw/model_fingerprint.json')['files']),
        'dev_input_sha256':digest(rows),'dev_case_ids':sorted(bycase),
        'allowed_prompt_sha256':sorted({hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows}),
        'files':{name:sha256_file(root/name) for name in ('axes.pt','train_fit.json','data_manifest.json','reused_comparisons.jsonl','baseline_score_cache.jsonl','protected_source_sha256.json')},
        'reused_L0':str(Path(args.previous)/'S1/smoke.json'),
        'L0_reason':'same layer, rank, positions, scoring and hook implementation; only the basis tensor changes',
        'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(root/'frozen_execution.json', frozen)
    return fit

class DevEngine(Engine):
    def configure(self, frozen):
        self.frozen = frozen
        assert self.modelhash == frozen['model_hash']
    def score(self, prompt, options, edits=None, biases=None, cache=True):
        h = hashlib.sha256(prompt.encode()).hexdigest()
        if h not in self.frozen['allowed_prompt_sha256']:
            raise ValueError('only frozen N3 development prompts are permitted')
        sp = model_spec(edits or [], biases or [])
        key = score_key(self.modelhash,prompt,options,sp,candidate_contract(self.tokenizer,prompt,options))
        if (not cache or key not in self.cache) and self.calls >= self.frozen['max_new_forwards']:
            raise RuntimeError('48-forward development budget exhausted')
        return super().score(prompt,options,edits,biases,cache)
    def capture(self,*args,**kwargs): raise RuntimeError('new captures are outside this development study')
    def generate(self,*args,**kwargs): raise RuntimeError('free generation is outside this development study')

def run(args):
    root=Path(args.out); frozen=read_json(root/'frozen_execution.json')
    assert not (root/'raw').exists(), 'immutable run: use a new directory for any authorized retry'
    assert code_hashes()==frozen['source_code_sha256']
    for name,h in frozen['files'].items(): assert sha256_file(root/name)==h
    rows=load(args.data,'N3','dev');assert digest(rows)==frozen['dev_input_sha256']
    assert all(r['task']=='N3' and r['split']=='dev' for r in rows)
    (root/'raw').mkdir();shutil.copyfile(root/'baseline_score_cache.jsonl',root/'raw/forward_journal.jsonl')
    started=time.time();engine=None
    try:
        engine=DevEngine(args.model,root/'raw');engine.configure(frozen)
        axes=torch.load(root/'axes.pt',map_location='cpu',weights_only=True)
        grid=[];scores={}
        for beta in GRID:
            preds=[runone(engine,r,r['controls'],'route',beta,axes,rows) for r in rows]
            grid.extend(preds);scores[beta]=primary('N3',preds)
        beta=max(scores,key=lambda b:(scores[b],-b))
        selection={'task':'N3','split':'dev','scoring_contract':SCORING_CONTRACT,'variant':frozen['variant'],
            'grid':GRID,'scores':scores,'selected_route_beta':beta,'selected_random_beta':beta,
            'selection_rule':frozen['selection_rule'],'axes_sha256':sha256_file(root/'axes.pt'),
            'frozen_execution_sha256':sha256_file(root/'frozen_execution.json')}
        selection['sha256']=digest(selection);write_json(root/'dev_selection.json',selection)
        random=[runone(engine,r,r['controls'],'random',beta,axes,rows) for r in rows]
        reused=read_jsonl(root/'reused_comparisons.jsonl')
        comparisons=[{**p,'method':'old_route' if p['method']=='route' else p['method'],
            'comparison_reused':True,'forward_count':0,'cache_hit':True} for p in reused]
        chosen=[p for p in grid if p['beta']==beta]+random+comparisons
        write_jsonl(root/'dev_grid_predictions.jsonl',grid)
        write_jsonl(root/'dev_random_predictions.jsonl',random)
        write_jsonl(root/'dev_comparison_predictions.jsonl',chosen)
        summary=summarize('N3',chosen,root,'dev')
        summary.update(variant=frozen['variant'],selected_route_beta=beta,new_forward_count=engine.calls,
            test_status='not_run_user_scope',N2_status='unchanged',N5_status='unchanged',
            conclusion='exploratory development comparison on already-seen families; not a test confirmation')
        pairwise={}
        selected={p['case_id']:p for p in chosen if p['method']=='route'}
        for method in ('mean','text','old_route'):
            ref={p['case_id']:p for p in comparisons if p['method']==method}
            pairwise[method]={'corrected':sum(selected[k]['correct'] and not ref[k]['correct'] for k in selected),
                'damaged':sum(ref[k]['correct'] and not selected[k]['correct'] for k in selected),
                'net_correct_difference':sum(selected[k]['correct']-ref[k]['correct'] for k in selected),
                'families':{f:float(np.mean([float(selected[k]['correct'])-float(ref[k]['correct']) for k in selected if selected[k]['family_id']==f])) for f in sorted({p['family_id'] for p in chosen})}}
        summary['route_vs_comparisons']=pairwise
        write_json(root/'summary.json',summary)
        hooks=[e for p in grid if p['beta'] for e in p['hook']['residual']]
        checks={'max_48_forwards':engine.calls<=BUDGET,'no_new_training_or_test':True,
            'all_nonzero_edits_executed':all(e['actual_norm']>0 for e in hooks),
            'non_target_tokens_unchanged':all(e['non_target_changed']==0 for e in hooks),
            'baseline_reused_without_forward':all(p['cache_hit'] and p['forward_count']==0 for p in grid if p['beta']==0),
            'source_files_unchanged':snapshot(args.previous)==read_json(root/'protected_source_sha256.json')}
        assert all(checks.values()),checks
        records=read_jsonl(engine.journal)
        assert all(r['checksum']==digest(r['payload']) for r in records)
        assert sum(r['payload'].get('forward_count',0) for r in records)==engine.calls
        report={'status':'n3_dev_complete','process_exit_code':0,'new_forward_count':engine.calls,
            'elapsed_seconds':time.time()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'model':engine.fp,'checks':checks,'source_code_sha256':code_hashes(),'gpu':engine.fp['cuda_visible_devices'],
            'outputs':{str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}}
        write_json(root/'run_manifest.json',report)
    except Exception:
        (root/'failure.log').write_text(traceback.format_exc())
        write_json(root/'failure.json',{'status':'failed','exit_code':1,'new_forward_count':engine.calls if engine else 0})
        raise
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run'])
    p.add_argument('--data',default=str(BASE/'data_v2_ready'))
    p.add_argument('--previous',default=str(BASE/'runs/v3_repair_20261020T144100Z'))
    p.add_argument('--model',default='/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base')
    p.add_argument('--out',required=True);args=p.parse_args()
    print(json.dumps((prepare if args.phase=='prepare' else run)(args),ensure_ascii=False,indent=2))
