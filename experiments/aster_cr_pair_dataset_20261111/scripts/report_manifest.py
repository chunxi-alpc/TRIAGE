"""Record actual Codex agent protocol and delivered dataset version."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import datetime,importlib.metadata,json,pathlib
def read(name,default=None):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else default
def main():
    hist=read('reports/completion_historical.json');final=read('reports/completion_all.json')
    frozen=read('reports/frozen_build_metadata.json',{})
    jobs=ROOT/'generation_parallel'
    author_claims=[json.loads(p.read_text()) for p in jobs.glob('w*_*/*author_claim.json')] if jobs.exists() else []
    reviewer_claims=[json.loads(p.read_text()) for p in jobs.glob('w*_*/*reviewer_claim.json')] if jobs.exists() else []
    stage='historical_review_running' if not hist else 'generation_or_new_candidate_review' if not final else 'dataset_complete'
    protocol=read('reports/experiment_protocol.json',{})
    scope=read('reports/execution_scope.json',{})
    if hist and not final and scope.get('current_window_scope')=='historical_review_only':stage='historical_review_complete'
    names=['historical_pairs.jsonl','f0_diagnostic_pairs.jsonl','generated_candidates.jsonl','accepted_historical_pairs.jsonl','rejected_historical_pairs.jsonl','accepted_pairs.jsonl','rejected_pairs.jsonl','review_results_historical.jsonl','review_results.jsonl','balanced_train.jsonl','full_pool_train.jsonl','pair_statistics.csv','source_coverage.csv','human_audit.csv']
    artifacts={}
    for n in names:
        p=ROOT/n
        if not p.exists():continue
        artifacts[n]=dict(bytes=p.stat().st_size)
    m=dict(dataset_name='ASTER-CR MEGA-CDP paired data',version='20261111-sol-agents-v2',stage=stage,experiment_root=str(ROOT),
        historical_data_root=str(DATA),historical_input_version='repair-v2',
        inventory=read('reports/inventory.json'),fit_source_ids=read('source_splits.json',{}).get('fit',[]),
        agent_a=dict(model='gpt-6.1-sol',reasoning_effort='high',agents=[r['owner'] for r in author_claims] or frozen.get('agent_a',{}).get('agents',[]),execution='user-created Codex agent; server data IO',generation_started=any(jobs.glob('w*_*/*author_edits.jsonl')) or frozen.get('agent_a',{}).get('generation_started',False),actual_snapshot_exposed=False),
        agent_b=dict(model='gpt-6.1-sol',reasoning_effort='xhigh',historical_agent='/root/agent_b_sol',new_candidate_agents=[r['owner'] for r in reviewer_claims] or frozen.get('agent_b',{}).get('new_candidate_agents',[]),execution='user-created independent Codex agent; server data IO',formal_semantic_rounds=1,blinded_to_generation_reasoning=True,actual_snapshot_exposed=False),
        protocol=protocol,execution_scope=scope,parallelism=dict(historical=protocol.get('parallelism',{}),new_candidates=read('generation_parallel/plan.json',frozen.get('parallelism',{}).get('new_candidates'))),reproduction=read('reports/reproduction_context.json'),
        gpu_resources=dict(gpus_used_for_formal_dataset_work=[],vram_gib=0),external_api_used=False,server_codex_cli_used=False,
        clinical_expert_review=False,human_audit=read('reports/human_audit_status.json',dict(actually_reviewed=0,human_pass_rate=None)),
        deterministic_validation=dict(script='scripts/validate.py',schema='pair.schema.json',unresolved_near_duplicates='UNCERTAIN'),
        dataset_a=dict(path='balanced_train.jsonl',selection='source, language and complexity stratification; same count per class from smallest accepted class; prioritize distinct cases before reusing a case'),
        dataset_b=dict(path='full_pool_train.jsonl',selection='all accepted A/P/C/F',future_training='class-balanced sampling or inverse-frequency weights; same model/features/objective/evaluation as Dataset A'),
        balanced_selection_revision=read('reports/balanced_selection_revision.json'),
        training_performed=False,model_structure_changed=False,paper_conclusions_changed=False,
        artifacts=artifacts,created_or_updated_at=datetime.datetime.now().astimezone().isoformat(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','jsonschema','matplotlib']})
    (ROOT/'dataset_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(stage)
if __name__=='__main__':main()
