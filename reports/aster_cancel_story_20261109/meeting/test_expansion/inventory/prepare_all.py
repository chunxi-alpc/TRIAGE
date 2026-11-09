
"""Prepare all released MEGA-CDP cases without changing the official manifest."""
from pathlib import Path
import argparse,json,sys,time
from collections import Counter,defaultdict

DEFAULT_OUT=Path('/home/gcx/aster_next_20261109/inventory')
DATA=Path('/home/gcx/TRIAGE/data/MEGA-CDP')
OLD=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
REPO=Path('/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
CLUSTERS=Path('/home/gcx/aster_manuscript_20261030/clinical01/source_clusters.json')
sys.path.insert(0,str(REPO))
from experiments.aster_representation.tree import tree_guideline

def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def available_dataset_inventory():
    config_path=REPO/'configs/datasets.json'
    config=json.loads(config_path.read_text())
    loader=str(REPO/'src/data_loader.py')+':load_dataset'
    common=dict(reader=loader,current_main_method='Frozen MEGA binary score correction; native nonbinary outputs need an explicit task adapter.')
    data=[
      dict(name='MEGA-CDP',splits={'train_val':34895,'official_test':7458},unit='generated case/reference path',ready=True,
           direct_evaluation=str(REPO/'experiments/aster_cr/paths.py')+':execute; current frozen inference/run.py',
           missing=[],paper_task=True),
      dict(name='NLI4CT-2024',splits={'train':1700,'dev':200,'test':5500,'practice_test':2142},unit='NLI statement',ready=True,
           direct_evaluation=str(REPO/'experiments/aster_cr/repair/external.py')+':nli4ct/evaluate_external; binary zero-shot adapter being prepared',
           missing=['Current main method must be reported as zero-shot transfer, not a new NLI fit.'],paper_task=True),
      dict(name='NLI4CT',splits={},unit='legacy 2023 version',ready=False,direct_evaluation=loader,
           missing=['Legacy download is partial/skipped; NLI4CT-2024 is complete and substitutes the native task.'],paper_task=False),
      dict(name='MedGUIDE',splits={'train':7747},unit='multiple-choice question',ready=True,direct_evaluation=loader,
           missing=['No currently frozen native multiple-choice output adapter for the main binary correction.'],paper_task=False),
      dict(name='MedCalc-Bench',splits={'train':10543,'test':1100},unit='numeric calculation question',ready=True,direct_evaluation=loader,
           missing=['Requires native numeric-answer scoring/tolerance; no direct path or NLI binary evaluation.'],paper_task=False),
      dict(name='MedPIC-Bench',splits={'test':467},unit='multi-select question',ready=True,
           direct_evaluation=str(REPO/'experiments/aster_cr/repair/external.py')+':medpic/evaluate_external',
           missing=['Prepared native items/pairs exist, but current main correction needs native multi-select output adaptation.'],paper_task=False),
      dict(name='FaithEval-inconsistent',splits={'test':1500},unit='contextual QA item',ready=True,direct_evaluation=loader,
           missing=['Text-answer task, no current main-method native QA output adapter.'],paper_task=False),
      dict(name='FaithEval-counterfactual',splits={'test':1000},unit='contextual QA item',ready=True,direct_evaluation=loader,
           missing=['Text-answer task, no current main-method native QA output adapter.'],paper_task=False),
      dict(name='FaithEval-unanswerable',splits={'test':2492},unit='contextual QA item',ready=True,direct_evaluation=loader,
           missing=['Answerability/text-output scoring must be adapted explicitly.'],paper_task=False),
      dict(name='ConflictMedQA',splits={'original':4290,'v2':4290},unit='QA version record',ready=True,direct_evaluation=loader,
           missing=['Original/v2 are versions, not automatically 8580 independent cases; native answer adapter absent.'],paper_task=False),
      dict(name='MedChangeQA',splits={'test':512},unit='QA row',ready=True,direct_evaluation=loader,
           missing=['Native newest-answer scoring needed; no current main-method output adapter.'],paper_task=False),
      dict(name='ClashEval',splits={'test':10179},unit='contextual QA row',ready=True,direct_evaluation=loader,
           missing=['Native original/modified-answer scoring needed; no current main-method output adapter.'],paper_task=False),
      dict(name='ConFiQA',splits={'MC':6000,'MR':6000,'QA':6000},unit='task-format record',ready=True,direct_evaluation=loader,
           missing=['Three task formats are not assumed independent cases; native output adapter needed.'],paper_task=False),
      dict(name='syllogism_validity',splits={},unit='logical validity example',ready=False,direct_evaluation=None,
           missing=['Only paper simulations found; no released real syllogism dataset or real evaluation manifest present.'],paper_task=True),
      dict(name='MedNLI',splits={},unit='NLI statement',ready=False,direct_evaluation=loader,
           missing=['No local dataset; existing download config marks credentialed access as skipped.'],paper_task=False),
    ]
    for item in data:
        entry=config.get(item['name'],{})
        item['path']=entry.get('project_path',entry.get('path'))
        item['reader']=loader
    return dict(scope='Paper tasks are MEGA-CDP, NLI4CT (available 2024 version), and syllogism validity; other local datasets are inventory, not claimed completed main-method evaluations.',
                records=data,counts_source=str(config_path),
                prepared_external_root=str(OLD/'external'),
                no_accuracy_files_read=True,no_new_datasets_downloaded=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    mapping=json.loads(CLUSTERS.read_text())['records']
    current_sources={r['source_id']:r['split'] for r in rows(OLD/'data/sources.jsonl')}
    current_cases={r['case_id']:r['split'] for r in rows(OLD/'data/path_cases.jsonl')}
    history=json.loads((OLD/'data/history_usage.json').read_text())
    historical={s:[v for v in why if ':official_test_clusters' not in v
                  and ':pr40_clusters' not in v and ':previous_mechanism_clusters' not in v]
                for s,why in history['excluded'].items()}
    historical={s:v for s,v in historical.items() if v}
    totals={};group_counts=Counter();case_role_counts=Counter();source_stats={}
    errors=[]
    with (out/'all_path_cases.jsonl').open('w',encoding='utf-8') as combined:
        for part in ['train_val','test']:
            raw={int(r['guideline_id']):r for r in rows(DATA/part/'decision_trees.jsonl')}
            paths={int(r['guideline_id']):r['paths'] for r in rows(DATA/part/'case_paths.jsonl')}
            cases=rows(DATA/part/'vignette_cases.jsonl');cache={}
            count=gold_nodes=all_nodes=0;part_sources=set();part_groups=Counter()
            # Only new products are written. Existing inventory/path_cases.jsonl stays intact.
            destination=out/('train_val_path_cases.jsonl' if part=='train_val'
                             else 'official_all_path_cases.jsonl')
            with destination.open('w',encoding='utf-8') as handle:
                for case in cases:
                    gid=int(case['guideline_id']);source_key=part+':'+str(gid)
                    source=mapping[source_key]['cluster'];source_role=current_sources.get(source,'none')
                    case_id='mega:'+part+':'+str(gid)+':'+str(case['path_id'])
                    case_role=current_cases.get(case_id,'none')
                    if gid not in cache:
                        try:
                            cache[gid]=tree_guideline(json.loads(raw[gid]['text'])['decision_tree'])
                        except (ValueError,KeyError,TypeError) as exc:
                            cache[gid]=None
                            errors.append(dict(dataset_split=part,guideline_id=gid,error=str(exc)))
                    if cache[gid] is None:
                        continue
                    guide,node_map=cache[gid]
                    path=paths[gid][int(case['path_id'])-1]
                    gold=[dict(gid='G'+str(node_map[str(x['index'])]['node']),
                               label=int(x['decision_outcome']=='yes'))
                          for x in path if 'decision_outcome' in x]
                    leaf='L'+str(node_map[str(path[-1]['index'])]['node'])
                    if part=='test':
                        split_group='official_test'
                    elif source_role=='fit':
                        split_group='fit'
                    elif source_role=='dev':
                        split_group='dev'
                    elif source_role=='test':
                        split_group='old_test'
                    else:
                        split_group='remaining_train_val'
                    history_group=('current_'+source_role if source_role!='none' else
                                   ('earlier_recorded_source' if source in historical
                                    else 'no_recorded_input_use'))
                    record=dict(
                        case_id=case_id,source_id=source,source_key=source_key,
                        dataset='MEGA-CDP',dataset_split='train_val' if part=='train_val' else 'official_test',
                        split=(source_role if source_role in ('fit','dev') and part=='train_val' else 'test'),
                        split_group=split_group,evaluation_group=split_group,usage_group=history_group,
                        history_group=history_group,current_source_role=source_role,current_case_role=case_role,
                        in_current_fit_dev=source_role in ('fit','dev'),in_old_test=source_role=='test',
                        is_current_fit_case=case_role=='fit',is_current_dev_case=case_role=='dev',
                        is_old_test_case=case_role=='test',known_earlier_input_use=source in historical,
                        guideline_id=gid,source_filename=raw[gid]['filename'],language=raw[gid].get('language'),
                        patient=case['text'],guideline=guide,definitions=[],gold_path=gold,gold_leaf=leaf)
                    line=json.dumps(record,ensure_ascii=False)+'\n'
                    handle.write(line);combined.write(line)
                    count+=1;gold_nodes+=len(gold);all_nodes+=len(guide['groups'])
                    group_counts[split_group]+=1;case_role_counts[case_role]+=1;part_groups[split_group]+=1;part_sources.add(source)
                    stat=source_stats.setdefault(source,dict(source_id=source,source_keys=set(),split_groups=set(),
                                                            original_splits=set(),current_source_role=source_role,cases=0))
                    stat['source_keys'].add(source_key);stat['split_groups'].add(split_group)
                    stat['original_splits'].add(part);stat['cases']+=1
            totals[part]=dict(path=str(destination),cases=count,trees=sum(v is not None for v in cache.values()),
                              source_clusters=len(part_sources),split_groups=dict(part_groups),
                              gold_decision_nodes=gold_nodes,all_reachable_case_node_contexts=all_nodes)
    with (out/'all_sources.jsonl').open('w',encoding='utf-8') as handle:
        for source,stat in sorted(source_stats.items()):
            for key in ['source_keys','split_groups','original_splits']:stat[key]=sorted(stat[key])
            handle.write(json.dumps(stat,ensure_ascii=False)+'\n')
    manifest=dict(
        dataset='MEGA-CDP',all_cases_path=str(out/'all_path_cases.jsonl'),
        train_val_cases_path=str(out/'train_val_path_cases.jsonl'),
        official_cases_path=str(out/'official_all_path_cases.jsonl'),
        original_official_manifest_unchanged=str(DEFAULT_OUT/'path_cases.jsonl'),
        totals=totals,all_cases=sum(v['cases'] for v in totals.values()),
        all_trees=sum(v['trees'] for v in totals.values()),all_source_clusters=len(source_stats),
        split_groups=dict(group_counts),exact_old_case_roles=dict(case_role_counts),
        role_definition='split_group fit/dev/old_test uses the existing unified source assignment; current_case_role and is_*_case separately retain exact prior-case membership.',
        source_mapping=str(CLUSTERS),conversion_errors=errors,training='none',
        scope='All released generated vignettes/reference paths; fit/dev/old_test groups are descriptive training/development evaluations, not independent held-out tests.',
        seconds=time.time()-started)
    dump(out/'all_primary_inventory.json',manifest)
    dump(out/'available_datasets.json',available_dataset_inventory())
    print(json.dumps(manifest,ensure_ascii=True))
if __name__=='__main__':main()

