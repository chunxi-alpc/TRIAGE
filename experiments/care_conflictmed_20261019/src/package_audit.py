"""Verify and package observed artifacts. All execution is on the server."""
import json,hashlib,collections,datetime,subprocess,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
    release={}
    for name in ['ConflictMed.jsonl','ConflictMed_v2.jsonl']:
        rs=rows(ROOT/'sources'/name);idx={r['id']:r for r in rs}
        release[name]=dict(n=len(rs),unique_ids=len(idx),labels=dict(collections.Counter(r['label'] for r in rs)),diseases=dict(collections.Counter(r['disease'] for r in rs)),label_pair_mismatches=sum(idx[i]['label']!='correct' or idx[i+2145]['label']!='wrong' for i in range(2145)),paired_disease_mismatches=sum(idx[i]['disease']!=idx[i+2145]['disease'] for i in range(2145)),paired_bias_mismatches=sum(idx[i]['bias_type']!=idx[i+2145]['bias_type'] for i in range(2145)),mapping_to_advice='not assumed or inferred; files acquired but not used in model runs')
    cfg=json.loads((ROOT/'config.json').read_text());model=Path(cfg['model'])
    model_hashes={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in model.iterdir() if p.is_file() and (p.suffix in ['.json','.safetensors'] or p.name=='merges.txt')}
    dump(ROOT/'sources/model_files_sha256.json',model_hashes)
    d=torch.load(ROOT/'results/direction.pt',map_location='cpu',weights_only=True)
    compact={k:d[k] for k in ['direction','layer','train_ids']}
    compact['source_sha256']=sha(ROOT/'results/direction.pt')
    torch.save(compact,ROOT/'results/direction_weights.pt')
    copied=torch.load(ROOT/'results/direction_weights.pt',weights_only=True)
    assert torch.equal(copied['direction'],d['direction'])
    rev='5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e'
    source_manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p),url=f'https://huggingface.co/datasets/RDBH/DriftMed/resolve/{rev}/'+('README.md' if p.name=='DriftMed_README.md' else p.name)) for p in (ROOT/'sources').iterdir() if p.name in ['advice_pairs.jsonl','ConflictMed.jsonl','ConflictMed_v2.jsonl','DriftMed_README.md']}
    dump(ROOT/'sources/download_manifest.json',dict(revision=rev,transport='downloaded on network relay with curl, copied byte-for-byte by scp; all data preparation and model inference on server',server_network_issue='github.com DNS resolution failed; Hugging Face requests also unavailable',files=source_manifest))
    dump(ROOT/'audit/artifact_audit.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),release_checks=release,full_training_activation=dict(path=str(ROOT/'results/direction.pt'),bytes=(ROOT/'results/direction.pt').stat().st_size,sha256=sha(ROOT/'results/direction.pt'),git_included=False),portable_direction=dict(path='results/direction_weights.pt',sha256=sha(ROOT/'results/direction_weights.pt'),exact_vector_equality=True),script_change_after_main='Only Engine environment-output filename made configurable for continuation; default and inference unchanged. Main source was not pre-committed; config and data hashes were frozen before inference.',verification='Completed JSONL hashes, split isolation, coverage and actual nonzero prefill edits verified in analyze.py. Clinical label truth was not independently reviewed.',prior_review=dict(pr=10,url='https://github.com/chunxi-alpc/TRIAGE/pull/10',reviewed_head='6bae71cbb99d60503de0f1867c7c38ca4f1772b5',files=['experiments/care_steering_20261019/README.md','experiments/care_steering_20261019/REPORT.txt','experiments/care_steering_20261019/src/run_facts.py','experiments/care_steering_20261019/src/engine.py'],finding='Old medical target task supplies both old and new rules in context; cannot establish parametric knowledge conflict. Existing report already states this. Preserve its results and add a separate experiment.'),not_executed=['Native full scenario evaluation','Historical clinical guideline revision validation','Full CARE extraction and key-space intervention','Multiple models and direction seeds','DPO or SFT','Independent expert clinical review']))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','exports']) and p.name not in ['advice_pairs.jsonl','ConflictMed.jsonl','ConflictMed_v2.jsonl','DriftMed_README.md','direction.pt','ARTIFACT_MANIFEST.json','package.log']]
    dump(ROOT/'ARTIFACT_MANIFEST.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
    (ROOT/'exports').mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'exports/CARE_ConflictMedQA_results_20261019.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'ARTIFACT_MANIFEST.json']:z.write(p,arcname=str(p.relative_to(ROOT)))
    print(json.dumps(dict(release=release,files=len(files),archive_bytes=(ROOT/'exports/CARE_ConflictMedQA_results_20261019.zip').stat().st_size)))
if __name__=='__main__':main()
