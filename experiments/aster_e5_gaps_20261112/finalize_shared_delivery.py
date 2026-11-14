"""Refresh E5 delivery records from completed artifacts only; no model execution."""
from pathlib import Path
import csv,json,time
ROOT=Path('/home/gcx/aster-e5-gaps-20261112')
OUT=ROOT/'reports/aster_e5_gaps_20261112'
QUEUE=Path('/home/gcx/aster-gpu-queue-20261113')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
groups={
'output_controls':OUT/'output_controls_e1/completion.json',
'loreft':OUT/'loreft_e1/completion.json',
'lora':OUT/'native_supervised_e1/lora/completion.json',
'lofit':OUT/'native_supervised_e1/lofit/completion.json',
'representation':OUT/'representation_e1_plan/completion.json',
'prompt':OUT/'prompt_controls_e1/completion.json'}
completed={}
for key,p in groups.items():
    if p.exists():
        value=read(p)
        completed[key]=value.get('state',value.get('status','complete')) in ('complete','completed','success') or value.get('complete') is True
    else:completed[key]=False
with (OUT/'section_4_5_new_e1.csv').open() as f:table=list(csv.DictReader(f))
queue=read(QUEUE/'status.json')
external=[j for j in queue['jobs'].values() if j['id'].startswith('codex03-') and j.get('gpu_count')]
external_complete=bool(external) and all(j['state']=='complete' for j in external)
own_complete=all(completed.values())
status=dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),codex02_required_groups_complete=own_complete,
    codex03_external_groups_complete=external_complete,E5_status='\u672c\u8f6e\u57fa\u7ebf\u8865\u7f3a\u5b8c\u6210' if own_complete and external_complete else '\u90e8\u5206\u5b8c\u6210',
    completion_by_group=completed,completed_numeric_rows=len(table),new_results_table='section_4_5_new_e1.csv',
    historical_results_tables=['section_4_5_data.csv','historical/section_4_5_historical_table.csv','additional_historical/table_4_5_historical.csv'],
    objective_protocol_notes='See per-method native or task-adapted objective records',task_adaptations_explicitly_labelled=True,Qwen_IT_ASTER_retraining=0)

status.update(Qwen_NLI4CT_reruns=0,Qwen_MedPIC_reruns=0,manuscript_changed=False,
    historical_reports_overwritten=False,historical_test_note='40 sources / 806 paths have historical use; not a new independent test; 7458-path coverage is not inferred.',
    shared_queue=str(QUEUE),predictions_and_CIs='Per-method completion artifacts and section_4_5_new_e1.csv; archived historical predictions and intervals retained.')
write(OUT/'delivery_status_current.json',status)
write(OUT/'new_runs_current.json',dict(updated_at=status['updated_at'],completed_numeric_rows=len(table),completed_results=table,
    required_groups=completed,job_records=[j for j in queue['jobs'].values() if j['id'].startswith('codex02-')],
    excluded_runs=dict(ASTER_Qwen_IT_training=0,NLI4CT_Qwen_reruns=0,MedPIC_Qwen_reruns=0)))
def group(name):
    n=name.casefold()
    for token,key in [('loreft','loreft'),('lofit','lofit'),('lora','lora')]:
        if token in n:return key
    if any(x in n for x in ('leace','iti','sea')):return 'representation'
    if '\u5f3a\u8bc1\u636e' in name or 'strong' in n or n=='cc' or 'contextual' in n:return 'prompt'
    if any(x in n for x in ('temperature','\u6e29\u5ea6','\u622a\u8ddd','\u76d1\u7763\u7ebf\u6027','linear','affine')):return 'output_controls'
    return None
records=[]
with (OUT/'compatibility_matrix.csv').open() as f:
    for old in csv.DictReader(f):
        name=old['method'];key=group(name)
        result=dict(method=name,historical_evidence=old['existing_evidence'],historical_training_protocol=old['training_protocol'])
        result['current_status']=('E1\u8865\u7f3a\u5df2\u5b8c\u6210' if completed[key] else '\u5c1a\u672a\u5b8c\u6210\uff1b\u53ea\u4fdd\u7559\u5b9e\u9645\u7ed3\u679c') if key else '\u590d\u7528\u5df2\u4fdd\u5b58\u7684\u5386\u53f2\u6216\u51bb\u7ed3E1\u7ed3\u679c'
        result['artifact']=str(groups[key].relative_to(OUT)) if key else 'reused_results.json'
        n=name.casefold()
        if 'loreft' in n:note='\u4fee\u590d\u540e\u91cd\u65b0\u62df\u5408\uff1bbinary supervised CE\u4efb\u52a1\u9002\u914d\uff0c\u4e0d\u542bA/P\u6216C/F\u5dee\u5206\u635f\u5931'
        elif 'lofit' in n:note='Qwen\u7ed3\u6784\u9002\u914d\uff1b\u4fdd\u7559\u4e24\u9636\u6bb5\u5168\u8bcd\u8868\u7b54\u6848\u4ea4\u53c9\u71b5\u76ee\u6807'
        elif 'lora' in n:note='\u672c\u5730\u5355\u5c42rank1\u914d\u7f6e\uff1b\u5168\u8bcd\u8868\u7b54\u6848\u4ea4\u53c9\u71b5'
        elif 'leace' in n:note='LEACE-pair\u4efb\u52a1\u9002\u914d\uff1b\u4f7f\u7528A/P\u4e0e\u5171\u4eab\u5b9a\u4f4d\u5934\uff0c\u72ec\u7acb\u6807\u8bb0'
        elif 'iti' in n:note='ITI-style\u4efb\u52a1\u9002\u914d\uff1b\u6b63\u786e/\u9519\u8bef\u5b8c\u6210\u6fc0\u6d3b\u4e0e\u539f\u751fprobe/shift'
        elif 'sea' in n:note='E1\u8865\u62df\u5408\uff1b\u5b98\u65b9AdapterLayer\u4e0e\u539f\u751f\u8c31\u76ee\u6807\uff1b\u5386\u53f2\u5b8c\u6574\u7ed3\u679c\u4fdd\u7559'
        elif 'cad' in n:note='CAD-style\u4e0a\u4e0b\u6587\u9002\u914d\uff1b\u4f18\u5148\u590d\u7528\u5386\u53f2\u5b8c\u6574\u8bc4\u4ef7'
        else:note='\u8be6\u89c1audit_reference\u4e0e\u5404\u65b9\u6cd5\u534f\u8bae\uff1b\u5df2\u5b8c\u6210\u6570\u636e\u76f4\u63a5\u590d\u7528'
        result['protocol_note']=note;records.append(result)
with (OUT/'compatibility_matrix_current.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
text=('# E5 \u5f53\u524d\u4ea4\u4ed8\u72b6\u6001\n\n'+status['E5_status']+'\u3002\u672c\u8f6e\u5b9e\u9645\u65b0\u589e\u8868\u542b '+str(len(table))+' \u884c\uff08\u542b\u591a\u4e2a\u968f\u673a\u79cd\u5b50\u884c\uff09\u3002\n\n'
    '\u6700\u65b0\u517c\u5bb9\u77e9\u9635\uff1acompatibility_matrix_current.csv\u3002\u65b0\u589e\u8fd0\u884c\u4e0e\u9010\u65b9\u6cd5\u7ed3\u679c\uff1anew_runs_current.json\u3001section_4_5_new_e1.csv\u3002\u5b8c\u6210\u72b6\u6001\uff1adelivery_status_current.json\u3002\u9010\u4f8b\u9884\u6d4b\u548c95%\u6765\u6e90\u805a\u7c7b\u7f6e\u4fe1\u533a\u95f4\u4fdd\u7559\u5728\u5404\u65b9\u6cd5\u76ee\u5f55\uff0c\u5386\u53f2\u62a5\u544a\u4fdd\u6301\u539f\u540d\u3002\n\n'
    '40\u6765\u6e90\u3001806\u8def\u5f84\u5177\u6709\u5386\u53f2\u4f7f\u7528\u8bb0\u5f55\uff0c\u4e0d\u80fd\u79f0\u4e3a\u5168\u65b0\u72ec\u7acb\u6d4b\u8bd5\uff1b\u4e0d\u63a8\u65ad7458\u8def\u5f84\u7ed3\u679c\u3002\u8bba\u6587\u6b63\u6587\u672a\u4fee\u6539\u3002\n')
(OUT/'E5_CURRENT_STATUS.zh.md').write_text(text)
p=ROOT/'docs/ASTER/experiments.zh.md'
addition='\n\n### E5 \u7edf\u4e00\u961f\u5217\u6700\u7ec8\u6c47\u603b\uff082026-11-13\uff09\n\n'+status['E5_status']+'\u3002\u5b9e\u9645\u65b0\u589e\u7ed3\u679c\u3001\u517c\u5bb9\u77e9\u9635\u3001\u65b0\u589e\u8fd0\u884c\u6e05\u5355\u548c\u9010\u65b9\u6cd5\u5b8c\u6210\u72b6\u6001\u89c1 [\u5f53\u524d\u4ea4\u4ed8\u8bb0\u5f55](../../reports/aster_e5_gaps_20261112/E5_CURRENT_STATUS.zh.md)\u3002\u7b2c4.5\u8282\u6570\u503c\u4ec5\u6765\u81ea\u5df2\u5b8c\u6210\u65b9\u6cd5\uff1b\u5386\u53f240\u6765\u6e90\u3001806\u8def\u5f84\u4ecd\u6807\u8bb0\u4e3a\u6709\u5386\u53f2\u4f7f\u7528\u8bb0\u5f55\u7684\u8bc4\u4ef7\u3002\n'
s=p.read_text()
if '### E5 \u7edf\u4e00\u961f\u5217\u6700\u7ec8\u6c47\u603b\uff082026-11-13\uff09' not in s:p.write_text(s+addition)
print(json.dumps(status,ensure_ascii=False))
