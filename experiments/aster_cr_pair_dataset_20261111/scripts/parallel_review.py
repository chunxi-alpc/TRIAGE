"""Aggregate disjoint single-round Codex review shards; no semantic judgments."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,datetime,fcntl,json,os,pathlib,subprocess,sys
SHARDS=ROOT/'review_sol/shards'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rows(p):
    if not p.exists():return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def read(p):return json.loads(p.read_text())
def save(p,obj):
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,p)
def collect():
    plan=read(SHARDS/'plan.json')
    baseline=rows(ROOT/'review_sol/historical_results.jsonl')
    combined=list(baseline); summaries=[]
    all_expected=set(r['pair_id'] for r in baseline)
    if len(all_expected)!=len(baseline):raise RuntimeError('Duplicate baseline IDs')
    for shard in plan['shards']:
        folder=pathlib.Path(shard['output_dir'])
        assignment=read(folder/'assignment.json')
        expected={pid for source in assignment['sources'] for pid in source['pair_ids']}
        rr=rows(folder/'historical_results.jsonl');got={r['pair_id'] for r in rr}
        if expected & all_expected:raise RuntimeError('Assignments overlap')
        if len(got)!=len(rr) or not got<=expected:raise RuntimeError('Shard output duplicates or crosses assignment')
        all_expected.update(expected);combined.extend(rr)
        summaries.append(dict(shard=shard['shard'],reviewer=shard['reviewer'],reviewed=len(rr),
                              assigned=len(expected),complete=(folder/'completed.json').exists() and got==expected,
                              status=dict(collections.Counter(r['status'] for r in rr))))
    ids=[r['pair_id'] for r in combined]
    if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate review across files')
    progress=dict(stage='historical_review_parallel',reviewed=len(combined),total=plan['total_pairs'],
                  percent=round(100*len(combined)/plan['total_pairs'],2),
                  baseline=len(baseline),status=dict(collections.Counter(r['status'] for r in combined)),
                  shards=summaries,updated_at=now())
    save(ROOT/'review_sol/parallel_progress.json',progress)
    save(ROOT/'review_sol/progress.json',progress)
    return combined,all_expected,progress
def merge(finalize):
    with (SHARDS/'merge.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        combined,expected,progress=collect()
        if not all(s['complete'] for s in progress['shards']):
            raise RuntimeError('All three complete flags and assigned results are required')
        candidates=rows(ROOT/'historical_pairs.jsonl')
        ordered=[r['pair_id'] for r in candidates]
        actual={r['pair_id']:r for r in combined}
        if set(ordered)!=expected or set(actual)!=set(ordered):
            raise RuntimeError('Historical input and single-round review IDs differ')
        wrong=[r['pair_id'] for r in combined if r.get('review_model')!='gpt-6.1-sol' or r.get('reasoning_effort')!='xhigh' or r.get('round')!=1]
        if wrong:raise RuntimeError('Formal review model/effort/round mismatch')
        path=ROOT/'review_results_historical.jsonl';temp=path.with_name(path.name+'.tmp')
        with temp.open('w',encoding='utf-8') as stream:
            for pid in ordered:stream.write(json.dumps(actual[pid],ensure_ascii=False)+'\n')
        os.replace(temp,path)
        save(SHARDS/'merge_complete.json',dict(status='complete',pairs=len(actual),updated_at=now(),shards=progress['shards']))
        if finalize and not (ROOT/'reports/completion_historical.json').exists():
            logfile=ROOT/'logs/parallel_historical_finalize.log'
            with logfile.open('w',encoding='utf-8') as out:
                result=subprocess.run([sys.executable,str(SCRIPTS/'validate.py'),'--stage','historical','--finalize'],
                                      cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Historical validation failed; details in '+str(logfile))
        return dict(merged=len(actual),finalized=(ROOT/'reports/completion_historical.json').exists(),updated_at=now())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['progress','merge'])
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.action=='progress':result=collect()[2]
    else:result=merge(args.finalize)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
