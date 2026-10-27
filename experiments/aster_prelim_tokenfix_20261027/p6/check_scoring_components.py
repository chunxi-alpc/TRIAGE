"""Reproduce structural checks without running a judge or reading test data."""
import hashlib,json,sys,tempfile
from pathlib import Path
D=Path(__file__).resolve().parent
OLD=Path('/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution')
S=OLD/'runtime/vendors/MEGA-CDP-ssh/mega_cdp/scripts/evaluate'
sys.path.insert(0,str(S))
from parse_prediction_output_singleturn import parse_sections
from pathway_metrics import dynamic_time_warping
from score_cache import get_or_score,cache_key
checks={}
checks['dtw_identical_zero']=dynamic_time_warping([[0,1],[1,0]])[1]==0
checks['dtw_all_mismatch_one']=dynamic_time_warping([[1,1],[1,1]])[1]==1
checks['parser_en']=parse_sections('# Clinical Decision Process\n1. Condition observed.\n# Clinical Decision Outcome\nAction.', 'en')==(['Condition observed.'],'Action.')
checks['parser_zh']=parse_sections('# 临床决策过程\n1. 条件成立。\n# 临床决策结论\n采取措施。','zh')==(['条件成立。'],'采取措施。')
try:
 parse_sections('invalid', 'en');checks['parser_rejects_invalid']=False
except ValueError:checks['parser_rejects_invalid']=True
calls=[]
def score():calls.append(1);return {'PathCost':0.25}
with tempfile.TemporaryDirectory(prefix='p6-score-cache-') as t:
 a=get_or_score(t,'case','output',['reference'],{'judge':'synthetic'},score)
 b=get_or_score(t,'case','output',['reference'],{'judge':'synthetic'},score)
 checks['same_tuple_scores_once']=len(calls)==1 and a[1]==b[1] and b[2]
 c=get_or_score(t,'case','output',['different'],{'judge':'synthetic'},score)
 checks['changed_reference_new_cache']=len(calls)==2 and c[1]!=a[1]
 checks['changed_config_new_key']=cache_key('case','output',['reference'],{'judge':'different'})!=a[1]
assert all(checks.values()),checks
audit=json.loads((D/'results/scorer_audit.json').read_text())
audit['checks']=checks
audit['reproduction_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(D/'results/scorer_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'status':audit['status'],'clinical_outputs_scored':0}))
