"""Collect the revised 500-test-case protocol without claiming queued stages finished."""
import datetime,json,shutil
from pathlib import Path
from zoneinfo import ZoneInfo
exp=Path(__file__).resolve().parent;root=Path('/home/gcx/aster_validation_20261022/private/run500');out=exp/'results/run500';out.mkdir(parents=True,exist_ok=True)
for name in ['dataset_manifest.json','model_fingerprints.json','condition_labels.json','positive_preparation.json']:
 if (root/name).exists():shutil.copyfile(root/name,out/name)
design=json.loads((root/'option_order/design.json').read_text());design.pop('inverse_label_maps');(out/'option_order_design.json').write_text(json.dumps(design,indent=2))
status={'checked_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'test_cases':500,'development_cases_excluded':40,'option_rotations':2,'positive_cases':500,'models':{}}
for model in ['llama','qwen']:
 dest=out/model;dest.mkdir(exist_ok=True)
 for f in root.glob(model+'*.manifest.json'):shutil.copyfile(f,dest/f.name)
 stages={}
 for name in ['status','positive_status','cad_fix_status']:
  f=root/f'{model}.{name}';stages[name]=f.read_text().strip() if f.exists() else 'not_started'
 stages['main_complete']=(dest/'full/summary.json').exists();stages['option_order_complete']=(dest/'option_order/summary.json').exists()
 stages['positive_complete']=stages['positive_status']=='complete'
 for arm in ['baseline','seka']:
  f=root/'positive'/model/arm/'efficacy_metrics.json'
  stages['positive_complete'] &= f.exists()
  if f.exists():
   (dest/'positive').mkdir(exist_ok=True);shutil.copyfile(f,dest/'positive'/f'{arm}_efficacy_metrics.json')
 stages['cad_fix_complete']=stages['cad_fix_status']=='complete' and (dest/'cad_fix/summary.json').exists()
 stages['complete']=stages['cad_fix_complete'] and stages['main_complete'] and stages['option_order_complete'] and stages['positive_complete'];status['models'][model]=stages
status['requested_run_complete']=all(x['complete'] for x in status['models'].values());(out/'STATUS.json').write_text(json.dumps(status,indent=2))
lines=['# 500 道测试题实验','',f"检查时间：{status['checked_at']}",'范围：原固定留出划分的前500题，排除40道开发题；不是根据模型准确率筛题。原3645题任务的已有结果保留，500题对应主输出原样复用。','CAD/prune_cad全部重跑128 token；两种选项重排覆盖同一500题；SEKA独立正对照500题。','']
for model,stages in status['models'].items():lines.append(model+': '+json.dumps(stages,ensure_ascii=False))
lines+=['','原始输出、病例及详细日志保留服务器 private/run500；本目录仅保存可发布数值和配置。','原始选项顺序和重排的动作一致率按动作身份比较。K/U仅代表一次闭卷原动作选择，不能证明参数记忆。','原先未具备条件的预测适用性、Context-DPO/CANOE、模型自选关键句不属于已完成结果。']
(out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
(exp/'results/REPORT.zh.md').write_text('# 当前执行范围已调整\n\n用户将主实验改为500道留出测试题。最新状态和结果见 [run500/REPORT.zh.md](run500/REPORT.zh.md)。旧目录保留诊断和历史记录，不能据此声称3645例已完成。\n')
print(json.dumps(status))
