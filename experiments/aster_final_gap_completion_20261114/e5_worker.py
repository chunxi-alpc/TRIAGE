"""Missing interference scores only, reusing the trained E5 adapters and operators."""
import sys
import time
import types
from common import *
sys.path.insert(0, str(REPO5))

def record_score(method, uid, value, data, scores):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite method score')
    append(OUT / 'e5' / method / 'new_node_scores.jsonl',
           dict(uid=uid, method=method, score=value, raw_score=data.raw[uid], prediction=int(value > 0)))
    scores[method][uid] = value
    return value

def configuration(method):
    source = e5_source(method)
    record = dict(method=method, parameter_source=str(source), training_reused=True,
                  source_protocol='E1-balanced-2540', new_training=False,
                  source_natural_paths=str(source / 'test_path_predictions.jsonl'),
                  source_pair_predictions=str(source / 'test_pair_predictions.jsonl'),
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  expected_interference_paths=2418, expected_sources=40)
    for name in ['training.json', 'fit.json', 'selection.json']:
        if (source / name).exists():
            record[name[:-5]] = read(source / name)
    write(OUT / 'e5' / method / 'configuration.json', record)

def shared_paths(data, methods, score_missing, scores):
    """Co-traverse methods so an original prompt prefix serves their own final forwards."""
    from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import _execute_prepared_path
    completed = {m: {pathkey(r) for r in iterrows(OUT / 'e5' / m / 'interference_path_predictions.jsonl')} for m in methods}
    for (cid, variant), uids in data.groups.items():
        active = [m for m in methods if (cid, variant) not in completed[m]]
        if not active:
            continue
        case = data.cases[cid]
        nodes, root = data.trees[cid]
        positions = {m: root for m in active}
        visited = {m: {} for m in active}
        while True:
            requests = defaultdict(list)
            for m in active:
                gid = positions[m]
                if gid in nodes and gid not in visited[m]:
                    requests[gid].append(m)
            if not requests:
                break
            for gid, requested in requests.items():
                uid = uids[gid]
                missing = [m for m in requested if uid not in scores[m]]
                if missing:
                    computed = score_missing(uid, missing)
                    for method in missing:
                        record_score(method, uid, computed[method], data, scores)
                for m in requested:
                    s = scores[m][uid]
                    visited[m][gid] = s
                    positions[m] = nodes[gid]['yes' if s > 0 else 'no']
        for m in active:
            record = dict(method=m, case_id=cid, source_id=case['source_id'], variant=variant, interference=True,
                          **_execute_prepared_path(case, visited[m], nodes, root))
            append(OUT / 'e5' / m / 'interference_path_predictions.jsonl', record)
            completed[m].add((cid, variant))
            if len(completed[m]) < 4 or len(completed[m]) % 25 == 0:
                write(OUT / 'e5' / m / 'status.json', dict(state='running', paths_completed=len(completed[m]), expected=2418))

def adapters(kind, engine, data):
    import torch
    from experiments.aster_cr.baselines import Adapter
    methods = [kind + '_seed' + str(seed) for seed in [11, 22, 33]]
    pending = [m for m in methods if not read(OUT / 'e5' / m / 'done.json', {}).get('complete')]
    if not pending:
        return
    for m in pending:
        configuration(m)
    if kind == 'loreft':
        objects = {}
        for m in pending:
            ad = Adapter(engine, kind, 17, load=e5_source(m) / 'adapter.pt')
            ad.module.eval()
            ad.active = False
            objects[m] = ad
        scores = {m: data.cached_scores(m) for m in pending}
        def compute(uid, needed):
            for ad in objects.values():
                ad.active = False
            context = engine.prepare(data.prompts[uid])
            result = {}
            for m in needed:
                objects[m].active = True
                try:
                    with torch.no_grad():
                        result[m] = engine.run(context, capture=False)['ell']
                finally:
                    objects[m].active = False
            return result
        try:
            shared_paths(data, pending, compute, scores)
            for m in pending:
                data.summarize(m)
        finally:
            for ad in objects.values():
                ad.close()
    else:
        ad = Adapter(engine, kind, 17, load=e5_source(pending[0]) / 'adapter.pt')
        engine.model.eval()
        try:
            for index, method in enumerate(pending):
                if index:
                    ad.restore(e5_source(method) / 'adapter.pt')
                scores = {method: data.cached_scores(method)}
                def score(uid):
                    if uid not in scores[method]:
                        record_score(method, uid, ad.score(data.prompts[uid]), data, scores)
                    return scores[method][uid]
                data.complete_paths(method, score)
                data.summarize(method)
        finally:
            ad.close()

def representation(engine, data):
    import torch
    from experiments.aster_cr.comparators import Comparators, sea_class
    names = {'leace_pair': 'leace', 'iti_style': 'iti', 'sea_adaptation': 'sea'}
    pending = [m for m in names if not read(OUT / 'e5' / m / 'done.json', {}).get('complete')]
    if pending:
        comp = Comparators.__new__(Comparators)
        comp.eng, comp.ops, comp.sea_zero_norm_fallbacks = engine, {}, 0
        for method in pending:
            configuration(method)
            state = torch.load(e5_source(method) / 'operator.pt', map_location=engine.device, weights_only=False)
            if method == 'sea_adaptation':
                comp.sea = sea_class()(state['positive'], state['negative'], 'l2_norm', None)
            else:
                comp.ops[names[method]] = state
        scores = {m: data.cached_scores(m) for m in pending}
        residual_root = OUT / 'private_cache/sea_raw_final_residuals'
        residual_root.mkdir(parents=True, exist_ok=True)
        def compute(uid, needed):
            context = engine.prepare(data.prompts[uid])
            result = {}
            for method in needed:
                handle = None
                if method == 'sea_adaptation':
                    def capture(module, args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        np.save(residual_root / (uid + '.npy'), value[0, -1].detach().float().cpu().numpy())
                    handle = engine.layers[-1].register_forward_hook(capture)
                try:
                    with torch.no_grad():
                        result[method] = comp.edited(context, names[method])
                finally:
                    if handle is not None:
                        handle.remove()
            return result
        shared_paths(data, pending, compute, scores)
        for method in pending:
            data.summarize(method)
        write(OUT / 'e5/representation_runtime.json', dict(shared_prefix=True, methods=list(names),
              SEA_nonfinite_coordinate_fallbacks=comp.sea_zero_norm_fallbacks))
    method = 'strong_evidence_prompt'
    if not read(OUT / 'e5' / method / 'done.json', {}).get('complete'):
        from experiments.aster_e5_gaps_20261112.e1_prompt_controls import PromptEngine
        configuration(method)
        engine.strong = True
        engine.render = types.MethodType(PromptEngine.render, engine)
        scores = {method: data.cached_scores(method)}
        def score(uid):
            if uid not in scores[method]:
                record_score(method, uid, engine.score(data.prompts[uid], capture=False)['ell'], data, scores)
            return scores[method][uid]
        data.complete_paths(method, score)
        data.summarize(method)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['lora', 'lofit', 'loreft', 'representation'], required=True)
    args = parser.parse_args()
    import torch
    from experiments.aster_cr.engine import Engine
    torch.set_num_threads(4)
    started = time.time()
    data = E5Data(prompts=True)
    engine = Engine('qwen', device='cuda:0')
    engine.model.eval()
    if args.group == 'representation':
        representation(engine, data)
    else:
        adapters(args.group, engine, data)
    write(OUT / 'e5' / (args.group + '_worker_done.json'), dict(complete=True,
          seconds=time.time() - started, peak_gib=torch.cuda.max_memory_allocated(engine.device) / 2**30,
          counts=engine.counts, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), training=False))

if __name__ == '__main__':
    main()
