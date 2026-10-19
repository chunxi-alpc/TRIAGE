#!/usr/bin/env python3
"""Author-frozen synthetic symptom narratives; no real clinical validation.
Run on the experiment server before inference. Gold never enters model inputs.
"""
import argparse, collections, datetime, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Each family: requirement, exception, full narrative context, second condition,
# three versions of first condition, three versions of exception, Boolean operator.
FAMILIES=[
('headache_daily_activity','headaches interrupt ordinary activities and at least two episodes occurred this week','the person reports that headaches began for the first time in their life this week',
 'Mara describes a band-like ache while sorting invoices. Her sister has migraine, but this account concerns Mara. She has had three episodes since Monday.',
 'At least two headache episodes occurred this week.',
 ['During each episode she puts her work aside until the ache settles.','She keeps working and doing her usual activities throughout each episode without interruption.','She describes the ache but gives no account of whether it interrupts what she is doing.'],
 ['She remembers the same sort of headache on several occasions last year.','She is clear that she had never experienced any headache before this week.','The conversation ends before anyone asks about headaches earlier in her life.'],'AND'),
('breathlessness_stairs','breathlessness occurs when climbing stairs and settles when the person stops','breathlessness also occurs while the person is sitting quietly',
 'Joel is discussing his walk to an upstairs office. He says the breathing trouble settles after he pauses. His colleague has asthma; Joel gives no diagnosis for himself.',
 'The breathlessness settles when the person stops.',
 ['One flight of stairs leaves him short of breath.','He climbs the stairs without any breathlessness; the trouble occurs only during running.','He describes running but nobody asks what happens on stairs.'],
 ['Sitting quietly, he breathes comfortably and has no breathlessness.','Even while sitting quietly he has episodes of breathlessness.','His breathing while sitting quietly is not discussed.'],'AND'),
('postural_lightheadedness','lightheadedness starts on standing and improves on sitting back down','the person has lost consciousness during one of these episodes',
 'Anika describes a brief floating feeling when moving around the kitchen. Whenever it comes on, sitting back in the chair makes it fade. Her husband once fainted, but she is describing her own episodes here.',
 'The lightheadedness improves on sitting back down.',
 ['The feeling begins just as she rises from her chair.','Standing up never brings it on; it starts only after she has already been walking for several minutes.','She cannot say whether it starts as she stands or later in the walk.'],
 ['Throughout every episode she has remained awake and aware.','During one episode she blacked out and later found herself on the floor.','She remembers being helped into a chair but cannot establish whether she blacked out.'],'AND'),
('solid_food_swallowing','solid food sticks during swallowing and liquids pass without sticking','a previous swallowing referral remains open',
 'Ben describes meals over the last fortnight. Water and soup go down freely without a sticking sensation. His father had swallowing trouble years ago; this note concerns Ben.',
 'Liquids pass without sticking.',
 ['Bread and pieces of meat repeatedly seem to lodge on the way down.','Bread, meat, and all other solid foods pass without sticking; his complaint is a sour taste afterward.','He talks about soup and water, but swallowing solid food is not described.'],
 ['The referral office confirms that he has no open swallowing referral.','The referral office confirms that his earlier swallowing referral is still open.','An old swallowing referral is mentioned, but its current status cannot be established.'],'AND'),
('night_hand_sensation','hand tingling wakes the person from sleep or shaking the hand relieves tingling','the same complaint is already being assessed in an active specialist appointment series',
 'Leah describes pins and needles in one hand after using a sewing machine. Shaking the hand has never relieved it. Her mother wears a wrist splint, but Leah does not discuss her mother’s symptoms further.',
 'Shaking the hand relieves the tingling.',
 ['The tingling also wakes her from sleep several nights a week.','It has never woken her from sleep and occurs only when she is awake.','She cannot tell whether she wakes because of the hand or notices it after waking for another reason.'],
 ['The booking team confirms that no active specialist appointment series is assessing this hand complaint.','An active specialist appointment series is already assessing this same hand complaint.','She mentions specialist visits, but neither their purpose nor whether they are ongoing is known.'],'OR'),
('urinary_discomfort','passing urine is painful and urinary frequency has increased from the person’s usual pattern','the person is currently away and has requested assessment through another service instead',
 'Owen is discussing a change over three days. He now visits the toilet roughly twice as often as he ordinarily does. His partner’s unrelated symptoms are not part of this account.',
 'Urinary frequency has increased from the person’s usual pattern.',
 ['He describes a burning pain each time he passes urine.','He specifically says passing urine causes no pain or burning.','He describes the number of trips but does not say how passing urine feels.'],
 ['He is at home and has made no request to use another service.','He is away from home and has requested assessment through a different service instead.','He says he is away, but the note does not record whether he requested another service.'],'AND'),
('rash_new_product','a rash began after first using a new skin product and the rash is on skin where that product was applied','the person has already stopped the product and the rash has completely disappeared',
 'Nia describes red patches on her forearms, exactly where she applied a cream she had not used before. Her child has eczema, but the patches in this account are Nia’s.',
 'The rash is on skin where the new product was applied.',
 ['Her skin was clear before the first application; the patches appeared the next morning.','The patches were already present for a week before she first tried the cream.','She remembers both the patches and the new cream but cannot place their beginnings in order.'],
 ['She has stopped using the cream, but the patches remain visible.','She has stopped using the cream and the patches have disappeared completely.','She has stopped the cream, but nobody records whether any patches remain.'],'AND'),
('bowel_pattern_change','stools have become looser than usual or bowel movements have become more frequent than usual','the person has a scheduled bowel review and asks to discuss this change at that review instead',
 'Pavel is describing his bowel pattern around a recent investigation. The number of bowel movements has stayed at his usual one each morning. His brother’s bowel history is mentioned separately.',
 'Bowel movements have become more frequent than usual.',
 ['The stools have changed from his usual formed stools to loose stools.','The stools have kept their usual formed consistency without becoming looser.','The note gives the number of bowel movements but not their consistency compared with usual.'],
 ['He has no bowel review scheduled and has made no request to discuss the change at another review.','He has a bowel review scheduled and asks to discuss this change there instead.','He has a bowel review scheduled, but it is not recorded whether he asks to discuss this change there instead.'],'OR'),
('ankle_swelling_asymmetry','one ankle is visibly swollen and the other ankle is not swollen','the person has already arranged a same-day assessment elsewhere for this swelling',
 'Ruth describes her ankles after taking off her boots. She is explicit that the left ankle has its normal outline and is not swollen. Her spouse has longstanding swollen feet.',
 'The other ankle is not swollen.',
 ['Her right ankle looks puffy and is visibly larger than its usual outline.','Her right ankle also has its usual outline with no visible swelling.','The note describes the left ankle clearly but gives no usable description of the right ankle.'],
 ['She confirms that she has no assessment arranged elsewhere for this swelling today.','She has already booked an assessment at another service today specifically for this swelling.','She mentions an appointment elsewhere today, but its purpose is not recorded.'],'AND'),
('voice_recovery','the person’s voice has become hoarse and has not returned to its usual sound','the person asks to postpone this voice assessment until after an upcoming performance',
 'Tariq, who sings in a community choir, discusses the sound of his voice. He says it has not returned to its usual sound. His throat is not painful, and his daughter’s recent sore throat is a separate matter.',
 'The voice has not returned to its usual sound.',
 ['His speaking voice has become rough and hoarse.','His voice is quieter but he specifically says it has never sounded rough or hoarse.','He says his voice is different but cannot describe whether it is hoarse.'],
 ['He wants the voice assessment now and has not asked to delay it for a performance.','He asks for this voice assessment to wait until after his forthcoming performance.','He mentions a forthcoming performance but does not say whether he wants the assessment delayed.'],'AND'),
('observed_sleep_pauses','another person has observed pauses in breathing during sleep or the person wakes gasping','a sleep assessment for these symptoms has already been completed and no new symptoms have appeared since',
 'Uma discusses restless nights. She says she has never woken gasping. Her father snores, but it is Uma’s sleep that the observer is describing.',
 'The person wakes gasping.',
 ['Her partner has watched her stop breathing briefly during sleep and then resume.','Her partner has watched her sleep and explicitly reports no pauses in breathing.','Her partner mentions noisy sleep but says nothing about pauses in breathing.'],
 ['She confirms that no sleep assessment for these symptoms has been completed.','A sleep assessment for these symptoms was completed, and there have been no new symptoms since.','She remembers a sleep assessment but cannot say whether these symptoms were assessed or began afterward.'],'OR'),
('hand_grip_function','grip has become weaker than usual and the weakness affects opening jars','the person has already received and accepted a booking for assessment of this same grip problem',
 'Wei talks about handling kitchen items. Opening jars is now difficult because he cannot maintain enough grip, rather than because the lid is slippery. His brother’s hand injury is unrelated.',
 'Weakness affects opening jars.',
 ['He says his grip is distinctly weaker than it used to be.','He says his grip has always been this weak and has not become weaker than usual.','He describes his current grip but gives no comparison with his usual strength.'],
 ['The booking office confirms that he has no accepted booking for this grip problem.','He has received and accepted a booking to assess this same grip problem.','A booking letter is mentioned, but it is unclear what it concerns or whether he accepted it.'],'AND'),
]
def tri_and(a,b): return 'F' if 'F' in (a,b) else 'U' if 'U' in (a,b) else 'T'
def tri_or(a,b): return 'T' if 'T' in (a,b) else 'U' if 'U' in (a,b) else 'F'
def neg(a): return {'T':'F','F':'T','U':'U'}[a]
def write_jsonl(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=HERE/'data');args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest_natural72.json').exists():raise SystemExit('Frozen dataset exists; use a different --output-dir for reproduction.')
 cases=[];gold=[];families=[]
 for i,(key,req,exc,context,c2text,first,exception,op) in enumerate(FAMILIES,1):
  fid=f'v81_natural_{i:02d}_{key}'; c2='T' if op=='AND' else 'F'
  ruletexts=[f'Use the standard assessment queue if {req}, unless {exc}.',f'Use the alternative enquiry queue if it is false that {req}, unless {exc}.',f'Use the exception queue if {exc}; this takes precedence over the other two queues.']
  rules=[dict(rule_id=rid,text=t,action=a) for rid,t,a in zip(['R_TARGET','R_ALTERNATIVE','R_EXCEPTION'],ruletexts,['Standard assessment queue','Alternative enquiry queue','Exception queue'])]
  variants=[('p01_T',0,0,'p01','C1'),('p01_F',1,0,'p01','C1'),('p02_T',0,0,'p02','E1'),('p02_F',0,1,'p02','E1'),('u01',2,0,None,'C1'),('u02',0,2,None,'E1')]
  for variant,ci,ei,pair,changed in variants:
   # Pair 2 uses a separate narrative ordering, avoiding duplicate T records.
   if variant.startswith('p02'):patient=exception[ei]+' '+context+' '+first[ci]
   else:patient=context+' '+first[ci]+' '+exception[ei]
   cid=f'{fid}_{variant}';atoms={'C1':['T','F','U'][ci],'C2':c2,'E1':['F','T','U'][ei]};base=(tri_and if op=='AND' else tri_or)(atoms['C1'],c2)
   states=[tri_and(base,neg(atoms['E1'])),tri_and(neg(base),neg(atoms['E1'])),atoms['E1']]
   case=dict(case_id=cid,dataset_id='care_module1_v8_1_natural72',family_id=fid,split='natural72',target_rule_id='R_TARGET',patient=patient,question='Using only the narrative and the fictional routing policy, determine whether each complete rule applies.',rules=rules,pair_id=f'{fid}_{pair}' if pair else None,pair_eligible=bool(pair),variant=variant,changed_atom=changed,is_clinical=False)
   cases.append(case);gold.append(dict(case_id=cid,family_id=fid,pair_id=case['pair_id'],pair_eligible=bool(pair),variant=variant,changed_atom=changed,reference_state=states[0],reference_structure={'source':'author_truth_assignments_with_three_valued_logic','atoms':atoms,'rules':[dict(rule_id=r['rule_id'],state=s) for r,s in zip(rules,states)]}))
  families.append(dict(family_id=fid,requirement=req,exception=exc,condition2=c2text,base_operator=op,condition2_state=c2,rule_texts=ruletexts))
 assert len(cases)==72 and len({x['patient'] for x in cases})==72
 for i in range(0,72,6):assert [g['reference_state'] for g in gold[i:i+6]]==['T','F','T','F','U','U']
 write_jsonl(out/'natural72_cases.jsonl',cases);write_jsonl(out/'natural72_gold.jsonl',gold);write_jsonl(out/'natural72_families.jsonl',families)
 (out/'natural72_notes.zh.md').write_text('''# 新自然语言验证集\n\n72 个合成叙述，12 个新症状主题，每主题 6 例、每例 3 条完整规则。每主题包含两对 T/F 反事实病例及两个 U 病例；每对仅改变一个关键条件的句子。第一对改变必要条件或可选条件，第二对改变例外。第二对调整叙述顺序，避免与第一对的 T 病例完全重复。\n\n这是合成自然语言验证，不是临床验证。分流规则由作者虚构，不是诊疗指南，不评价真实诊断或治疗效果。主题与 V8 新72的行政流程主题不同；仍有刻意控制的配对结构，不能视作自然采集病例。每个主题只有六种变体，统计独立性应以主题为单位考虑。\n\n参考答案来自作者预先设置的命题真值与三值逻辑；缺失信息或无法确定时间关系为 U。作者检查叙述与参考是否一致，没有独立临床专家审核。模型仅接收 question、patient、rules.rule_id/text/action，不接收 gold、family definitions、variant、changed_atom 或配对信息。\n\n本数据在推理前冻结。不根据模型输出修改或筛除病例。复现脚本可指定新的 --output-dir，内容文件哈希可复核；冻结时间随新一次复现改变。\n''')
 names=['natural72_cases.jsonl','natural72_gold.jsonl','natural72_families.jsonl','natural72_notes.zh.md']
 manifest=dict(dataset_id='care_module1_v8_1_natural72',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=72,families=12,rules_per_case=3,total_rule_judgments=216,pairs=24,u_scenarios=24,synthetic=True,is_clinical=False,independent_clinical_validation=False,frozen_before_model_predictions=True,prediction_allowlist=['question','patient','rules.rule_id','rules.text','rules.action'],reference_rule_states=dict(collections.Counter(s['state'] for g in gold for s in g['reference_structure']['rules'])),files={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names},builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'manifest_natural72.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'cases':72,'rules':216,'pairs':24,'manifest':str(out/'manifest_natural72.json')}))
if __name__=='__main__':main()
