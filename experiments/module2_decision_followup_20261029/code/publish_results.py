from pathlib import Path
import subprocess, json, shutil, hashlib, zipfile
root=Path('/home/gcx/TRIAGE-worktrees')
repo=root/'applicability-decision-20261029'
run=root/'applicability-decision-20261029-run02'
code=root/'applicability-decision-20261029-code'
dest=repo/'experiments/module2_decision_followup_20261029'
archive=root/'triage_decision_results_20261029.zip'
assert json.loads((run/'pipeline_exit.json').read_text())['exit_code']==0
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='232c45c132ba5712edf760e598e0c03741a382f442a90decf107650fd83ff2b9'
assert not dest.exists(),'Publication directory already exists'
dest.mkdir()
for name in ['code','summaries','figures','provenance','artifacts']: (dest/name).mkdir()
for p in code.iterdir():
 if p.is_file() and p.suffix in ['.py','.sh'] and not p.name.startswith('transfer'):
  shutil.copy2(p,dest/'code'/p.name)
shutil.copy2(code/'protocol.json',dest/'PROTOCOL.json')
shutil.copy2(run/'final_review/REPORT.zh.txt',dest/'REPORT.zh.txt')
for p in (run/'final_review').iterdir():
 if p.suffix in ['.csv','.json']: shutil.copy2(p,dest/'summaries'/p.name)
 elif p.suffix in ['.png','.pdf']: shutil.copy2(p,dest/'figures'/p.name)
for source,target in [('prepare/manifest.json','prepare_manifest.json'),('select/frozen.json','frozen_selection.json'),('localize/heads.json','localized_heads.json'),('smoke.json','smoke.json'),('pipeline_exit.json','pipeline_exit.json')]:
 shutil.copy2(run/source,dest/'provenance'/target)
for p in run.glob('*/status.json'): shutil.copy2(p,dest/'provenance'/f'{p.parent.name}_status.json')
shutil.copy2(archive,dest/'artifacts'/archive.name)
summary=json.loads((run/'confirm/summary.json').read_text())
labels={'baseline':'无干预','projected_mean':'原投影均值','optimized_subspace':'优化子空间','score_bias':'直接偏置','reversed_score_bias':'反向直接偏置','optimized_score_bias':'优化直接偏置'}
table=['| 方法 | 准确率 | 配对全对率 |','| --- | ---: | ---: |']
for row in summary:
 table.append(f"| {labels.get(row['method'],row['method'])} | {row['accuracy']:.2%} | {row['paired_accuracy']:.2%} |")
readme="""# 规则选择干预：2026-10-29 完整后续实验

**已完成，预定成功标准未达到。** 主要候选优化子空间为 49/96（51.04%），无干预及预选强直接偏置均为 48/96（50%）。增益的指南级配对 97.5% 区间为 [0.00, 4.17] 个百分点，未建立优于随机方向的证据。

使用 Qwen3-0.6B-Base 固定修订、FP32，在 115 服务器 GPU 0/1 完成。原实验的 12 个来源指南全部排除；新数据按来源指南分为 24/12/12。确认集为 12 个指南、96 个唯一正负提示、48 个最小对。原实验与原确认集未修改。

"""+'\n'.join(table)+"""

## 文件入口

- [完整中文研究报告](REPORT.zh.txt)
- [全部代码、逐例数据、日志及失败记录的完整压缩包](artifacts/triage_decision_results_20261029.zip)
- [冻结实验协议](PROTOCOL.json)、[实现代码](code/run_stages.py)
- [逐方法确认指标](summaries/heldout_metrics.csv)、[逐指南结果](summaries/heldout_by_guideline.csv)
- [单头/多头与增强/抑制/联合消融](summaries/heldout_ablation_metrics.csv)
- [任务与分词审计](summaries/scoring_audit.csv)、[因果定位汇总](summaries/localization_summary.csv)
- [覆盖与冻结摘要核验](summaries/completion_validation.json)
- [确认前冻结的选择](provenance/frozen_selection.json)、[数据与模型指纹](provenance/prepare_manifest.json)
- [文件与压缩包清单](PUBLICATION_MANIFEST.json)

![留出准确率增益](figures/heldout_accuracy.png)

图中为描述性 95% 指南级配对自助区间；主要两项比较采用协议指定的 97.5% 区间。全部配置在确认之前选定，随机及置乱种子均报告。

## 完成范围与解释

完成拟合/选择基线审计 288 条、反事实定位 14,688 条、训练日志 480 条、选择评估 76,128 条、确认主比较 1,152 条、确认消融 3,456 条。覆盖完整，无重复或缺失单元；冻结文件摘要一致。模型计算约 47.3 分钟，不含下载。

原单词元 A/B 审计在 288 个拟合/选择提示上全部选择 B。带空格答案评分也接近随机水平；这些结果不可视为自然病历或自由处置生成成绩。干预采用适用性真值路由，属于算子上界实验。

新确认集未用于本轮训练或选择，但其他项目对这些指南的历史接触情况尚未确认。本轮不声称假设成立。主要候选仅改对一例；原投影均值和一个随机方向的观察成绩更高。

## 复现

基础源码固定为 6078465acdbe9676678aa1db143d9954e5ed26ed。运行代码要求一个单独的该提交工作副本，通过 --repo 指定；本发布提交包含运行产物，不应被误当作原始基础源码。

在 115 服务器使用 /home/gcx/gcx261014 环境（PyTorch 2.8.0+cu128、Transformers 4.57.6），按 prepare → audit → localize → train → select → confirm → report 执行。示例：

    git worktree add --detach /path/to/base-6078465 6078465acdbe9676678aa1db143d9954e5ed26ed
    bash experiments/module2_decision_followup_20261029/code/run_server.sh /path/to/base-6078465 /path/to/new-run

模型为 Qwen/Qwen3-0.6B-Base，修订 da87bfb608c14b7cf20ba1ce41287e8de496c0cd。模型权重不入库；文件 SHA-256、原始开发数据修订与校验值见指纹。所有运行输出必须使用新目录。

压缩包保留 run02/ 的全部代码执行结果、日志、数据划分、训练系数与冻结配置，code/ 的实际运行代码，以及 run01_preparation_failure/ 的首次准备失败记录。首次失败是源码尚未展开导致导入失败；未覆盖该记录。原实验冻结算子的副本位于包内 original_reference/。

部分脚本保留原服务器路径，以记录实际执行；一般复现入口为上面的参数化脚本。最终整理脚本只读取完成的结果，没有再次运行模型。
"""
(dest/'README.md').write_text(readme)
with zipfile.ZipFile(archive) as z: archive_files=[{'path':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist()]
files={}
for p in sorted(dest.rglob('*')):
 if p.is_file(): files[str(p.relative_to(dest))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
manifest={'base_commit':'6078465acdbe9676678aa1db143d9954e5ed26ed','run_id':'applicability-decision-20261029-run02','status':'completed; prespecified success criterion not met','files':files,'complete_archive_members':archive_files,'model_weights_included':False}
(dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for key in ['user.name','user.email']:
 value=subprocess.check_output(['git','-C','/home/gcx/TRIAGE','config','--get',key],text=True).strip()
 subprocess.check_call(['git','-C',str(repo),'config',key,value])
print(json.dumps({'destination':str(dest),'files':len(files)+1,'archive_members':len(archive_files),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())},ensure_ascii=False))
