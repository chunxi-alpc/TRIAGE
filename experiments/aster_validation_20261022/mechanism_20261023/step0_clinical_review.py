"""Generate blinded clinical review packets only in an authorized private directory."""
import argparse
import json
import random
from pathlib import Path
from step0_reanalysis import read_jsonl


def make_packets(cases, summary, output):
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('review directory must be empty')
    focus={r['id'] for rows in summary['cohorts'].values() for r in rows}
    focus.update(r['id'] for r in summary['structure_rows'] if r['sibling_actions_after']>1)
    focus.update(r['id'] for r in summary['top2_changed_rows'])
    remaining=sorted({c['id'] for c in cases}-focus)
    random.Random(20261024).shuffle(remaining)
    controls=set(remaining[:20]); chosen=focus|controls
    mapping=[]; packets=[]
    ordered=[(c,v) for c in sorted(cases,key=lambda c:c['id']) if c['id'] in chosen for v in 'ABCD']
    random.Random(20261025).shuffle(ordered)
    for n,(c,v) in enumerate(ordered,1):
        rid=f'R{n:04d}'
        mapping.append(dict(review_id=rid,id=c['id'],variant=v,control=c['id'] in controls))
        packets.append(dict(review_id=rid,patient=c['patient'],options=c['options'],
            rules=[{'condition':r['condition'],'action':r['action']} for r in c['variant_rules'][v]],
            ratings=dict(applicable_rule_indices=None,condition_applicability=None,
                action_reasonableness_by_rule=None,reasonable_option_labels=None,action_unique=None,
                rule_option_semantics=None,uncertainty_reason=None,evidence=None)))
    for reviewer in [1,2]:
        rows=list(packets);random.Random(20261025+reviewer).shuffle(rows)
        (output/f'reviewer{reviewer}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'coordinator_only_mapping.json').write_text(json.dumps(mapping,indent=2))
    return dict(status='packets_created_not_reviewed',focus_cases=len(focus),control_cases=len(controls),total_cases=len(chosen),items_per_reviewer=len(packets),reviewers_completed=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--private-output',type=Path,required=True);a=p.parse_args()
    if Path(__file__).resolve().parents[3] in a.private_output.resolve().parents:
        raise ValueError('private review packets must not be inside repository')
    print(json.dumps(make_packets(read_jsonl(a.cases),json.loads(a.summary.read_text()),a.private_output)))
if __name__=='__main__':main()
