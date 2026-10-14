#!/usr/bin/env python3
"""Consolidate completed official resource downloads and existing validations."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from resume_requested_downloads import atomic
names=['MEGA-CDP','MedCalc-Bench','ConFiQA','FaithEval-unanswerable','ConflictMedQA','MedChangeQA','ClashEval']
datasets=json.loads((ROOT/'configs/datasets.json').read_text())
models=json.loads((ROOT/'configs/models.json').read_text())['models']
pending=[n for n in names if datasets.get(n,{}).get('status')!='downloaded' or not datasets[n].get('complete')]
if pending:raise RuntimeError('Downloads incomplete: '+', '.join(pending))
nli=json.loads((ROOT/'results/medical_nli_smoke.json').read_text())
assert models['medical_nli']['status']=='downloaded' and nli['status']=='passed'
reports={}
for name in ['text_datasets_validation','github_datasets_validation','conflictmedqa_validation','continuation_integration_validation']:
    file=ROOT/'results'/(name+'.json')
    if not file.exists():raise FileNotFoundError(file)
    reports[name]=json.loads(file.read_text())
loader=json.loads((ROOT/'results/datasets_status.json').read_text())
assert all(loader.get(n,{}).get('verification')=='passed' for n in names)
report={'status':'completed','completed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'server':'gcx@10.29.128.115','project':'/home/gcx/TRIAGE','environment':'/home/gcx/gcx261014','environment_redeployed':False,'images_requested':False,'datasets':{n:datasets[n] for n in names},'dataset_validations':reports,'loader_validation':{n:loader[n] for n in names},'medical_nli':{'configuration':models['medical_nli'],'verification':nli},'scope':'official downloads and read checks; three NLI examples only; no model training or full benchmark'}
atomic(ROOT/'results/continuation_download_report.json',report)
atomic(ROOT/'results/resume_download_status.json',{'status':'completed','completed_at':report['completed_at'],'datasets':names,'medical_nli':'passed','images_requested':False,'report':str(ROOT/'results/continuation_download_report.json')})
p=ROOT/'README.md';text=p.read_text()
old='完整数据：MedGUIDE、MedPIC-Bench、NLI4CT-2024、FaithEval 的 inconsistent/counterfactual 子集。MEGA-CDP 和 MedCalc-Bench 的原 test 部分完整可读，train 部分未完整。FaithEval-unanswerable 因下载入口限流未完成，ConFiQA 官方下载文件链接返回 404。实际范围以服务器配置为准。'
new='完整数据：MedGUIDE、MedPIC-Bench、NLI4CT-2024、FaithEval 三个子集、MedCalc-Bench train/test、ConFiQA QA/MR/MC、ConflictMedQA 两个官方版本、MedChangeQA、ClashEval。MEGA-CDP 的 train_val/test 原始 JSONL 和 2,274 份指南 Markdown 已完整，按用户要求仅下载文本；Markdown 文件总数 2,275 包括根 README。实际路径、来源、版本和范围见服务器配置。'
assert old in text;text=text.replace(old,new)
old='MEGA-CDP 本次完成 test 的 4 个原 JSONL，train_val 尚未完整；Markdown 指南取得 152/2,275 个，截图未下载。完整下载可加 `--full-mega` 续传。MedCalc-Bench 的 test 完整，train 尚未完成。MedNLI 需要 PhysioNet 认证、培训和数据使用协议，本次跳过。'
new='此次续传入口为 `scripts/resume_requested_downloads.py --stage core`，复用原缓存与断点，只处理本次指定数据和医学 NLI 模型；MEGA-CDP 仅包含文本。MedCalc-Bench train 10,543 条、test 1,100 条；FaithEval-unanswerable 2,492 条；MedChangeQA 512 条；ConflictMed 原版及 v2 各 4,290 条。ClashEval 使用官方 GitHub 的 `ba892d17f9b125264494689b7ce5d8ab22ed821d` 版本，保留原始 Parquet，不混入模型响应文件。MedNLI 仍需要 PhysioNet 认证、培训和数据使用协议。\n\n新数据读取示例：`load_dataset("MedChangeQA", split="test")`、`load_dataset("ConflictMedQA", split="v2")`、`load_dataset("ClashEval", split="test")`；MedChangeQA 的参考标签读取原始 Newest Label，Outdated Label 保留在 raw；ConFiQA 与 ClashEval 同时保留作者提供的两套上下文和答案。此次完整性报告见 `results/continuation_download_report.json`，模型三分类验证见 `results/medical_nli_smoke.json`。'
assert old in text;text=text.replace(old,new)
p.write_text(text)
print(json.dumps({'status':'completed','datasets':len(names),'medical_nli':'passed'}))

