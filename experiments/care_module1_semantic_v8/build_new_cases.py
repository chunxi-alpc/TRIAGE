#!/usr/bin/env python3
"""Freeze 12 new authored workflow families, with executable three-valued references.
No evaluated model participates in authoring. These are fictional, nonclinical tasks.
Gold/reference atoms are written separately from prediction cases.
"""
import collections,copy,datetime,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from src.module1.care_v6 import eval_logic,tri_and,tri_not

def AND(*a):return {'op':'AND','args':list(a)}
def OR(*a):return {'op':'OR','args':list(a)}
def NOT(a):return {'op':'NOT','args':[a]}
def atom(text,t,f,u):return {'text':text,'wording':{'T':t,'F':f,'U':u}}

# Each item is a new semantic family, not a claim of a new Boolean operator.
FAMILIES=[
 dict(key='recent_specimen_accession',name='近期标本与当前登记',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the current specimen was collected within the past 48 hours and its accession matches the current patient',exception_phrase='the laboratory has rejected that specimen',
  context='The coordinator is preparing the current specimen transfer. The previous sample was collected last month.',atoms={
  'C1':atom('The specimen currently being transferred was collected within the past 48 hours.','The specimen now being transferred was collected 18 hours ago.','The specimen now being transferred was collected 72 hours ago.','The time of collection for the specimen now being transferred is not recorded.'),
  'C2':atom('The accession for the specimen currently being transferred matches the current patient.','The accession number was checked against this patient and matches.','The accession number belongs to a different patient.','No one has checked whose accession number is on the container.'),
  'E1':atom('The laboratory has rejected the specimen currently being transferred.','The laboratory has rejected this specimen.','The laboratory has explicitly accepted this specimen without rejection.','The laboratory has not yet reported whether it accepts or rejects this specimen.')}),
 dict(key='current_callback_permission',name='回访许可有效期',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the patient\'s own callback permission is still active and was recorded within the past 30 days',exception_phrase='the patient has asked the service to suspend calls',
  context='A nurse is arranging a routine administrative callback. The patient\'s sister gave permission for her own calls yesterday.',atoms={
  'C1':atom('The current patient has active permission for callback calls.','The patient personally confirmed that permission to call remains active.','The patient personally confirmed that permission to call is no longer active.','The record contains only the sister\'s permission; the patient\'s permission status is not documented.'),
  'C2':atom('The current patient\'s callback-permission record was entered within the past 30 days.','The patient\'s permission-status entry is dated 12 days ago.','The patient\'s permission-status entry is dated 45 days ago.','The patient\'s permission-status entry has no date.'),
  'E1':atom('The current patient has asked the service to suspend callback calls.','The patient has asked the service to suspend all callback calls.','The patient explicitly says there is no request to suspend callback calls.','The suspension-request field has not been completed.')}),
 dict(key='imaging_device_clearance',name='影像设备清单与豁免',base=AND(NOT('C1'),'C2'),exception=AND('E1',NOT('E2')),flip='C1',flip_true='F',fixed={'C2':'T'},
  requirement='the patient has no implanted device and the current equipment questionnaire is complete',exception_phrase='a specialist review is required and no signed review waiver is in force',
  context='This invented scheduling policy routes equipment paperwork; it does not decide whether a scan is medically safe. The father has a cardiac implant.',atoms={
  'C1':atom('The current patient has an implanted device.','The patient has an implanted device.','The patient confirms having no implanted device.','Only the father\'s device is described; the patient\'s device status is absent.'),
  'C2':atom('The current patient\'s equipment questionnaire is complete.','The patient\'s equipment questionnaire has been completed today.','The patient\'s equipment questionnaire is still incomplete.','Whether the patient\'s equipment questionnaire is complete is not stated.'),
  'E1':atom('A specialist review is required for the current patient\'s equipment paperwork.','The scheduling team marked specialist review as required.','The scheduling team explicitly marked specialist review as not required.','The need for specialist review has not been entered.'),
  'E2':atom('A signed waiver of the required specialist review is currently in force.','A signed waiver of this review is currently in force.','The record explicitly says that no review waiver is in force.','The record does not say whether a review waiver is in force.')}),
 dict(key='medication_review_after_discharge',name='用药清单复核与后续出院',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the patient\'s medication list has been verified today and the patient brought their own medication containers',exception_phrase='the patient has had a hospital discharge after that verification',
  context='The desk is routing medication-list paperwork. The mother\'s medication list was verified this morning.',atoms={
  'C1':atom('The current patient\'s medication list was verified today.','The pharmacist verified the patient\'s own medication list today.','The pharmacist explicitly says the patient\'s medication list has not been verified today.','The note confirms only the mother\'s review and says nothing about a review of the patient\'s list.'),
  'C2':atom('The current patient brought their own medication containers.','The patient brought their own labeled medication containers.','The patient brought no medication containers.','The note does not say whether the patient brought medication containers.'),
  'E1':atom('The current patient had a hospital discharge after today\'s medication-list verification.','The patient was discharged from hospital later today, after the list verification.','The patient\'s last discharge was before the verification; there has been no discharge since.','The note mentions a discharge today but does not establish whether it was before or after the verification.')}),
 dict(key='result_contact_channels',name='结果通知渠道与临时限制',base=OR('C1','C2'),exception='E1',flip='C1',fixed={'C2':'F'},
  requirement='the patient currently permits either telephone notification or portal notification',exception_phrase='a temporary communication restriction is active on the patient\'s account',
  context='The service is selecting an administrative channel for a result notification. An expired restriction appears in last year\'s correspondence.',atoms={
  'C1':atom('The current patient currently permits telephone notification.','The patient currently permits notification by telephone.','The patient explicitly does not permit notification by telephone.','Current telephone-notification permission is not documented.'),
  'C2':atom('The current patient currently permits portal notification.','The patient currently permits notification through the portal.','The patient explicitly does not permit notification through the portal.','Current portal-notification permission is not documented.'),
  'E1':atom('A temporary communication restriction is currently active on the patient\'s account.','The account has an active temporary communication restriction.','The account explicitly shows no active temporary communication restriction.','One current note says the restriction is active and another current note says it is not; neither supersedes the other.')}),
 dict(key='remote_device_ownership',name='远程设备所属人与最近上传',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the uploaded readings come from a device assigned to this patient and were uploaded within the past 24 hours',exception_phrase='the device is currently listed as recalled',
  context='The coordinator is filing remote-monitoring records. The patient and spouse each have a separate device.',atoms={
  'C1':atom('The device producing the readings is assigned to the current patient.','The readings came from the device assigned to the patient.','The readings came from the device assigned to the spouse, not the patient.','The upload gives a device number, but its owner has not been identified.'),
  'C2':atom('The readings were uploaded within the past 24 hours.','These readings were uploaded 6 hours ago.','These readings were uploaded 40 hours ago.','The upload timestamp is missing.'),
  'E1':atom('The device producing the readings is currently listed as recalled.','This device is currently on the recall list.','This device is explicitly listed as not recalled.','The recall status of this device has not been checked.')}),
 dict(key='interpreter_patient_preference',name='口译需求与本人意愿',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the patient requests an interpreter and a professional interpreter for the requested language is available',exception_phrase='the patient has subsequently declined professional interpretation for this appointment',
  context='The receptionist is arranging language support for the patient. The accompanying brother needs an interpreter for his own appointment.',atoms={
  'C1':atom('The current patient has requested an interpreter for this appointment.','The patient personally requested an interpreter for this appointment.','The patient personally said that no interpreter was requested for this appointment.','The note records only the brother\'s request and gives no interpreter preference for the patient.'),
  'C2':atom('A professional interpreter for the requested language is available for this appointment.','A professional interpreter for the relevant language is available for this appointment.','No professional interpreter for the relevant language is available for this appointment.','Professional interpreter availability has not been recorded.'),
  'E1':atom('The current patient subsequently declined professional interpretation for this appointment.','After the initial arrangements, the patient declined professional interpretation for this appointment.','The patient confirms that there has been no subsequent refusal of professional interpretation.','The later conversation is noted but does not say whether the patient declined professional interpretation.')}),
 dict(key='consent_signatory_withdrawal',name='签署人与签署后撤回',base=OR('C1','C2'),exception='E1',flip='C1',fixed={'C2':'F'},
  requirement='the current form was signed either by the patient or by the patient\'s legally authorized representative',exception_phrase='permission on this form was withdrawn after it was signed',
  context='This fictional document-routing exercise concerns paperwork only. A relative signed a different form at a previous visit.',atoms={
  'C1':atom('The current form was signed by the patient.','The patient signed the current form.','The patient did not sign the current form.','The signature on the current form is illegible and the signer has not been identified.'),
  'C2':atom('The current form was signed by the patient\'s legally authorized representative.','The current form bears the signature of the patient\'s legally authorized representative.','No legally authorized representative signed the current form.','It is not known whether a legally authorized representative signed the current form.'),
  'E1':atom('Permission on the current form was withdrawn after its signing.','After the current form was signed, permission on that form was withdrawn.','The record explicitly confirms that permission on this form has not been withdrawn after signing.','There is a withdrawal note, but it does not identify which form or whether withdrawal followed this signature.')}),
 dict(key='checkin_departure_room_override',name='当前签到离场与房间例外',base=AND('C1',NOT('C2')),exception=AND('E1',NOT('E2')),flip='C2',flip_true='F',fixed={'C1':'T'},
  requirement='the patient has checked in for today\'s appointment and has not left the facility since checking in',exception_phrase='the assigned room is closed and no approved replacement room is available',
  context='The coordinator is directing the patient to an appointment room. Yesterday\'s visit ended normally.',atoms={
  'C1':atom('The current patient has checked in for today\'s appointment.','The patient checked in for today\'s appointment at 09:00.','The patient has not checked in for today\'s appointment.','Today\'s check-in status is not recorded.'),
  'C2':atom('The current patient has left the facility since checking in today.','The patient left the facility after today\'s check-in.','The patient remains in the facility and has not left since today\'s check-in.','The desk cannot establish whether the patient has left since today\'s check-in.'),
  'E1':atom('The room assigned to today\'s appointment is closed.','The assigned room is closed today.','The assigned room is open today.','The assigned room\'s current opening status is unknown.'),
  'E2':atom('An approved replacement room is available for today\'s appointment.','An approved replacement room is available for this appointment.','No approved replacement room is available for this appointment.','The desk has not confirmed whether an approved replacement room is available.')}),
 dict(key='archived_slides_current_loan',name='存档玻片归属与当前借出',base=AND('C1','C2'),exception='E1',flip='C1',fixed={'C2':'T'},
  requirement='the archived slides belong to this patient and the original slide set has a complete catalog entry',exception_phrase='the slides are currently on loan to another service',
  context='The archive is processing a slide-transfer request. A completed loan from last year remains in the file.',atoms={
  'C1':atom('The archived slides in this request belong to the current patient.','The identifiers confirm that these slides belong to this patient.','The identifiers confirm that these slides belong to the patient\'s twin, not this patient.','The identifiers cannot establish whether the slides belong to the patient or the twin.'),
  'C2':atom('The original slide set for this request has a complete catalog entry.','The archive confirms that this original slide set has a complete catalog entry.','The archive confirms that this original slide set has no complete catalog entry.','The archive has not recorded whether the original slide set has a complete catalog entry.'),
  'E1':atom('The slides in this request are currently on loan to another service.','The slide set is currently on loan to another service.','The old loan ended with return of the slides, and no current loan is active.','A loan is mentioned without a return status; it is unclear whether it is still active.')}),
 dict(key='transport_driver_alternative',name='交通安排与本人留宿请求',base=OR('C1','C2'),exception='E1',flip='C1',fixed={'C2':'F'},
  requirement='either a driver is confirmed for the patient or independent travel has been administratively approved',exception_phrase='the patient has requested an overnight accommodation booking',
  context='This invented policy assigns a transport-desk queue and makes no clinical fitness-to-travel judgment. The sister has a separate travel booking.',atoms={
  'C1':atom('A driver is confirmed to transport the current patient.','A driver is confirmed specifically for the patient.','The desk confirms that no driver is arranged for the patient.','A driver is confirmed for the sister, but the patient\'s driver status is absent.'),
  'C2':atom('Independent travel has been administratively approved for the current patient.','The transport desk has approved independent travel for this patient.','The transport desk has not approved independent travel for this patient.','The administrative approval status for independent travel is not documented.'),
  'E1':atom('The current patient requested an overnight accommodation booking.','The patient requested an overnight accommodation booking.','The patient explicitly declined overnight accommodation and made no booking request.','The record does not state whether the patient requested overnight accommodation.')}),
 dict(key='diary_review_new_entry',name='日记时间与复核后新增记录',base=AND('C1','C2'),exception=AND('E1',NOT('E2')),flip='C1',fixed={'C2':'T'},
  requirement='the patient\'s own diary contains entries on two consecutive days and the patient has requested a discussion',exception_phrase='the current diary was already reviewed and no new entry has been added after that review',
  context='The research coordinator is scheduling a discussion of a nonclinical study diary. The partner keeps a separate diary.',atoms={
  'C1':atom('The current patient\'s own diary contains entries on two consecutive days.','The patient\'s own diary has entries on Monday and Tuesday of this week.','The patient\'s own diary has entries only on Monday and Thursday of this week.','Only the partner\'s diary dates are available; dates in the patient\'s diary are missing.'),
  'C2':atom('The current patient requested a discussion of the diary.','The patient asked to discuss the diary.','The patient explicitly said no diary discussion was requested.','The patient\'s preference about a diary discussion is not documented.'),
  'E1':atom('The current diary has already been reviewed.','The coordinator has already reviewed the current diary.','The coordinator confirms that the current diary has not yet been reviewed.','Whether the current diary has been reviewed is not recorded.'),
  'E2':atom('A new entry was added to the current diary after its review.','A new diary entry was added after that review.','No new diary entry has been added after the review.','The note mentions another entry but does not establish whether it was added before or after the review.')}),
]

def dump_jsonl(path,rows):
 path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def make_cases():
 cases=[];gold=[];families=[]
 for n,f in enumerate(FAMILIES,1):
  fid=f'v8_new_{n:02d}_{f["key"]}'; ids=list(f['atoms']);flip=f['flip'];good=f.get('flip_true','T');bad='F' if good=='T' else 'T'
  neutral={k:'F' for k in ids};neutral.update(f['fixed']);neutral[flip]=good
  p1t=neutral.copy();p1f=dict(neutral,**{flip:bad})
  has_waiver='E2' in ids
  if has_waiver:
   p2t=dict(neutral,E1='T',E2='T');p2f=dict(neutral,E1='T',E2='F');u2=dict(neutral,E1='T',E2='U');p2changed='E2'
  else:
   p2t=neutral.copy();p2f=dict(neutral,E1='T');u2=dict(neutral,E1='U');p2changed='E1'
  u1=dict(neutral,**{flip:'U'})
  assignments=[('p01_T',p1t,'p01',flip),('p01_F',p1f,'p01',flip),('p02_T',p2t,'p02',p2changed),('p02_F',p2f,'p02',p2changed),('u01',u1,None,flip),('u02',u2,None,p2changed)]
  condition_atoms=[dict(condition_id=k,text=a['text'],criterion_kind='inclusion') for k,a in f['atoms'].items() if k.startswith('C')]
  exception_atoms=[dict(condition_id=k,text=a['text'],criterion_kind='exclusion') for k,a in f['atoms'].items() if k.startswith('E')]
  # Natural language includes the complete requirement in both adjacent branches.
  route=f'Route {n:02d} Ready';alternate=f'Route {n:02d} Preparation';exception=f'Route {n:02d} Exception'
  texts=[f'Assign {route} if {f["requirement"]}, unless {f["exception_phrase"]}.',
   f'Assign {alternate} if it is false that {f["requirement"]}, unless {f["exception_phrase"]}.',
   f'If {f["exception_phrase"]}, assign {exception}; this exception takes precedence over the other two branches.']
  rules=[dict(rule_id='R_TARGET',text=texts[0],action=route,conditions=condition_atoms,exceptions=exception_atoms,logic=f['base'],exception_logic=f['exception']),
   dict(rule_id='R_ALTERNATIVE',text=texts[1],action=alternate,conditions=condition_atoms,exceptions=exception_atoms,logic=NOT(f['base']),exception_logic=f['exception']),
   dict(rule_id='R_EXCEPTION',text=texts[2],action=exception,conditions=exception_atoms,exceptions=[],logic=f['exception'],exception_logic=OR())]
  guidance='These are fictional administrative routing rules, not medical guidance.\n'+'\n'.join(r['rule_id']+': '+r['text'] for r in rules)+'\nIf a unique route cannot be established, request review.'
  for variant,states,pair,changed in assignments:
   caseid=f'{fid}_{variant}';patient=f['context']+' '+' '.join(f['atoms'][k]['wording'][states[k]] for k in ids)
   question='Which administrative routing action follows for this patient from the stated observations and all three candidate rules?'
   option_map={'A':route,'B':alternate,'C':exception,'D':'Request review'}
   prompt='This is an invented administrative simulation, not medical advice. Use only stated observations about the current patient and the relevant time. Missing or unresolved conflicting information is unknown, not false.\n\nQuestion: '+question+'\nPatient: '+patient+'\nGuidance:\n'+guidance+'\nOptions:\n'+'\n'.join(k+'. '+v for k,v in option_map.items())+'\nReturn exactly one option letter.\nAnswer:'
   spans={}
   for rule in rules:
    start=prompt.index(rule['rule_id']+': ');a=prompt.index(rule['action'],start);spans[rule['rule_id']]=[a,a+len(rule['action'])]
   outputs=[dict(rule_id=r['rule_id'],state=tri_and([eval_logic(r['logic'],states),tri_not(eval_logic(r['exception_logic'],states))])) for r in rules]
   actions=[r['action'] for r,p in zip(rules,outputs) if p['state']=='T'];answer=next((k for k,v in option_map.items() if len(actions)==1 and v==actions[0]),'D')
   cases.append(dict(case_id=caseid,target_rule_id='R_TARGET',dataset_id='care_module1_semantic_v8_new72',family_id=fid,split='new72',patient=patient,question=question,query=question,guidance=guidance,rules=copy.deepcopy(rules),option_map=option_map,prompt=prompt,action_spans=spans,pair_id=f'{fid}_{pair}' if pair else None,pair_eligible=pair is not None,variant=variant,changed_atom=changed,is_clinical=False))
   gold.append(dict(case_id=caseid,family_id=fid,pair_id=f'{fid}_{pair}' if pair else None,pair_eligible=pair is not None,variant=variant,changed_atom=changed,reference_state=outputs[0]['state'],gold_answer=answer,gold_action=option_map[answer],reference_structure={'source':'authored_executable_reference','rules':outputs,'atoms':states},reference_method='Author-specified observation truth states, evaluated by the same established three-valued Boolean semantics; no evaluated-model labeling.'))
  families.append(dict(family_id=fid,name_zh=f['name'],rule_texts=texts,base_logic=f['base'],exception_logic=f['exception'],newness='New subject-specific workflow propositions and narratives; logical operators are intentionally shared with earlier experiments.'))
 return cases,gold,families

def main():
 out=HERE/'data';out.mkdir(exist_ok=True)
 if (out/'manifest_new72.json').exists():raise SystemExit('Frozen data already exist; do not regenerate in place.')
 cases,gold,families=make_cases()
 # One lightweight consistency pass: sizes, minimal-pair assignments, and expected target labels.
 assert len(cases)==72 and len(families)==12
 for i in range(0,72,6):
  gs=gold[i:i+6];assert [r['reference_state'] for r in gs]==['T','F','T','F','U','U']
  for a,b in [(gs[0],gs[1]),(gs[2],gs[3])]:
   aa=a['reference_structure']['atoms'];bb=b['reference_structure']['atoms'];assert sum(aa[k]!=bb[k] for k in aa)==1
 for c in cases:
  assert all(c['prompt'][s:e]==next(r['action'] for r in c['rules'] if r['rule_id']==rid) for rid,(s,e) in c['action_spans'].items())
 dump_jsonl(out/'new72_cases.jsonl',cases);dump_jsonl(out/'new72_gold.jsonl',gold);dump_jsonl(out/'new72_families.jsonl',families)
 review='''# 新增72例数据简要阅读记录\n\n全部12个规则模板及四个代表家族（1、3、8、12）的六种叙述已在冻结前作者/Codex逐句文本阅读检查（无独立人类专家审核）。标签先由作者设置观察命题真值，再由既有三值逻辑计算；没有调用被评估模型制定或修改标签。每个家族两对病例各只改一个关键条件句，另两例分别缺失入口条件和例外条件。读查重点为当前患者/家属、先后时间、否定及豁免的含义一致；无需额外审查流程。\n\n这些是人工编写的虚构医疗行政流程，非真实病例、非真实临床指南、无独立临床专家审核；12个新的语义家族不声称是12种新的布尔运算。既有24个mechanism家族和V4/V5/V6公开规则均已对照，没有沿用其年龄、肿瘤、授权/暂停命题或临床阈值。\n\n预测文件保留自然语言及作者提供的规则原子/逻辑以供A2使用，但不含参考真值、患者结构化事实、目标答案。A0/A1必须只取question、patient、rule_id/text/action。金标单独存new72_gold.jsonl。A2的原子分解及逻辑树来自规则作者，构成输入优势；不能把差异全部归因为模型算法。\n\n时间相关命题按规则所写的相关时点判断，历史描述仅在其确实支持该时间命题时有用。U保留信息缺失和未解决矛盾，不按效果删除。\n'''
 (out/'new72_review.zh.md').write_text(review)
 paths=['new72_cases.jsonl','new72_gold.jsonl','new72_families.jsonl','new72_review.zh.md']
 manifest=dict(dataset_id='care_module1_semantic_v8_new72',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=72,families=12,rules_per_case=3,total_rule_judgments=216,pairs=24,u_scenarios=24,split='new72',synthetic=True,is_clinical=False,independent_clinical_validation=False,authoring='Human-readable narratives authored before predictions; executable T/F/U references; no evaluated model used for labeling.',reference_rule_states=dict(collections.Counter(r['state'] for g in gold for r in g['reference_structure']['rules'])),files={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in paths},prediction_allowlist=['case_id','question','patient','rules.rule_id','rules.text','rules.action'],a2_extra_input=['rules.conditions','rules.exceptions','rules.logic','rules.exception_logic'],review='new72_review.zh.md',frozen_before_model_predictions=True,dataset_version=2,prefreeze_revision='Before any new72 predictions, improved grammatical wording of complete rule clauses and clarified archive catalog completeness to avoid conflating it with loan status; initial draft retained in new72_prefreeze_draft_v1.')
 (out/'manifest_new72.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
