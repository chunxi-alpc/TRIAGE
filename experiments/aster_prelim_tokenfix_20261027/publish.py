"""Publish only repaired-run artifacts and a link from the preserved historical run."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
D=Path(__file__).resolve().parent
paths={'P1':'results/gate_status.json','P2':'results/p2/gate_status.json','P3':'results/p3/gate_status.json','P4':'results/p4/gate_status.json','P5':'results/p5/gate_status.json'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--push',action='store_true');a=p.parse_args();states={};details={}
 for stage,rel in paths.items():
  path=D/rel
  if path.exists():z=json.loads(path.read_text());states[stage]=z[stage];details[stage]=z
  else:
   prefixes=['qwen','llama'] if stage=='P1' else [stage.lower()+'_'+x for x in ['qwen','llama']]
   states[stage]='RUNNING' if any((D/'private'/x/'manifest.json').exists() for x in prefixes) else 'NOT_RUN'
 failure=D/'private/runner_failure.json'
 if failure.exists():
  err=json.loads(failure.read_text());states[err['stage']]='ERROR_STOP'
 states['P6']='READY_FOR_DATA_PREPARATION' if states['P5']=='PASS_PRUNE_GATE' else 'NOT_RUN'
 status={'revision':'tokenfix_v3','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':states,'test_inference_count':0,'freeze':'results/freeze.json','old_run_preserved':'../aster_prelim_repaired_20261027','data_validity':'specified repair contracts passed; not independent human clinical adjudication'}
 (D/'results/overall_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 lines=['# 编码修复后 P1–P6 实际进度','','旧数据、模型输出与失败记录保存在 [修订开发集旧运行](../aster_prelim_repaired_20261027)。本轮从 P1 重验：Llama 全部重新推理；Qwen 仅在全部词元及权重一致性核对后复用。','','| 阶段 | 当前状态 |','|---|---|']+[f'| {k} | {v} |' for k,v in states.items()]
 lines+=['','NC/IC/AC/SW 各 169 例，CS 17 例；4 个开发树。两模型均使用 FP32。测试集推理为 0。','', '具体修复和标签范围见 [编码修复说明](TOKENIZATION_FIX.zh.md)，冻结指纹见 [freeze.json](results/freeze.json)。','', 'P6 不自动跳过用户要求的人工核验和数据规模门槛；前置通过后由定时任务继续准备。', '', '更新时间（UTC）：'+status['updated_at']]
 (D/'RUN_STATUS.zh.md').write_text('\n'.join(lines)+'\n')
 inv={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in D.rglob('*') if f.is_file() and not any(x in f.relative_to(D).parts for x in ['private','__pycache__']) and f.name!='artifact_sha256.json'}
 (D/'results/artifact_sha256.json').write_text(json.dumps(inv,indent=2)+'\n')
 if a.push:
  repo=D.parents[1];assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='exp/aster-p1-p6-20261027'
  subprocess.run(['git','add',str(D.relative_to(repo)),'experiments/aster_prelim_repaired_20261027/TOKENIZATION_FIX_POINTER.zh.md'],cwd=repo,check=True)
  if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:subprocess.run(['git','commit','-m','Record repaired development inputs and sequential preliminary runs'],cwd=repo,check=True)
  subprocess.run(['git','push','origin','exp/aster-p1-p6-20261027'],cwd=repo,check=True)
 print(json.dumps(status))
if __name__=='__main__':main()
