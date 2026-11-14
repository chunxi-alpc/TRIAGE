#!/usr/bin/env python3
"""CPU-only post-hoc matcher sensitivity; never changes data or learned directions."""
import collections
import json
import re
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(name):return [json.loads(s) for s in (ROOT/name).read_text().splitlines() if s.strip()]
def normalize(s):
    s=unicodedata.normalize('NFKD',s).casefold()
    s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub('[^a-z0-9]+',' ',s).strip()
def matches(raw,expected):
    answer,text=normalize(expected),normalize(raw)
    return bool(answer) and (answer==text or len(answer)>=4 and re.search(r'(?:^| )'+re.escape(answer)+r'(?: |$)',text) is not None)
facts={r['item_id']:r for r in read('data/facts.jsonl') if r['split']=='train'}
rows=read('results/fact_filter.jsonl')
groups=collections.defaultdict(list)
for row in rows:groups[row['item_id']].append(row)
attrition=collections.Counter();stage=collections.defaultdict(collections.Counter)
recovered=[];lost=[];original=[];verified=[]
for iid in facts:
    rr=groups.get(iid,[]);data={x['stage']:x for x in rr}
    assert len(data)==len(rr),f'Duplicate stage: {iid}'
    for row in rr:
        counts=stage[row['stage']]
        counts['attempted']+=1;counts['passed']+=bool(row['correct']);counts['failed']+=not row['correct']
        revised=matches(row['raw_output'],row['expected']);counts['nfkd_passed']+=revised
        if bool(row['correct'])!=revised:
            record={k:row[k] for k in ['item_id','stage','expected','raw_output']}
            record['question']=facts[iid]['question']
            (recovered if revised else lost).append(record)
    if 'clean0' not in data:category='not_attempted'
    elif not data['clean0']['correct']:category='first_clean_failed'
    elif any(f'clean{j}' in data and not data[f'clean{j}']['correct'] for j in range(1,5)):category='later_clean_variant_failed'
    elif 'conflict' not in data:category='incomplete_after_clean'
    elif not data['conflict']['correct']:category='conflict_did_not_follow'
    else:category='original_eligible';original.append(iid)
    attrition[category]+=1
    if all(f'clean{j}' in data and matches(data[f'clean{j}']['raw_output'],data[f'clean{j}']['expected']) for j in range(5)) and 'conflict' in data and matches(data['conflict']['raw_output'],data['conflict']['expected']):verified.append(iid)
for row in recovered:row['newly_verified_eligible']=row['item_id'] in set(verified)-set(original)
examples=[]
for iid in ['fact_0003','fact_0001','fact_0059','fact_0015','fact_0028']:
    row=next((r for r in groups[iid] if not r['correct']),None)
    if row:
        examples.append({**{k:row[k] for k in ['item_id','stage','expected','raw_output']},'question':facts[iid]['question'],'original_answer':facts[iid]['answer_original']})
result={'train_candidates':len(facts),'observed_filter_rows':len(rows),'attrition':dict(attrition),
        'stages':{k:dict(v) for k,v in stage.items()},
        'normalization':'Unicode NFKD + casefold + remove combining marks, then original ASCII/token-boundary matching rule',
        'normalization_recovered_observed_rows':len(recovered),'normalization_recovered_cases':len({r['item_id'] for r in recovered}),
        'recovered_examples':recovered,'normalization_lost_observed_rows':lost,
        'original_eligible':len(original),'verified_eligible_using_only_recorded_stages':len(verified),
        'newly_verified_eligible_ids':sorted(set(verified)-set(original)),'no_longer_verified_eligible_ids':sorted(set(original)-set(verified)),
        'requires_unobserved_followup_ids':sorted({r['item_id'] for r in recovered}-set(verified)),
        'manual_review_examples':examples,
        'caution':'Recovered early-stage cases are not qualified direction-training examples: later clean variants and/or conflict probes were never run. Original outputs and direction are unchanged.'}
(ROOT/'audit/filter_attrition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['筛选淘汰复核（只读CPU分析）','',
       f"候选{len(facts)}条；首次clean失败{attrition['first_clean_failed']}；后续4种指令变体再失败{attrition['later_clean_variant_failed']}；全部5种clean通过后冲突答案未匹配{attrition['conflict_did_not_follow']}；最终入选{attrition['original_eligible']}。",'',
       '阶段 | 实际运行 | 原匹配通过 | 原匹配失败 | NFKD修正通过']
for name in ['clean0','clean1','clean2','clean3','clean4','conflict']:
    c=stage[name];lines.append(f"{name} | {c['attempted']} | {c['passed']} | {c['failed']} | {c['nfkd_passed']}")
lines+=['',f"NFKD去音标+casefold只改变{len(recovered)}条已观察记录的判定，没有撤销原通过记录。其余匹配规则保持原样，没有增加别名词典，也没有用语义模型改标签。",f"仅根据实际保存的所有必需阶段，合格训练对仍为{len(verified)}，新增已验证合格数为{len(set(verified)-set(original))}。fact_0003的后续clean1–4及conflict从未运行，因此不能推测它最终通过。",'',
        '可人工核验短表（仅比较已保存的标签与原始输出，不替代外部事实核验）','ID | 阶段 | 该阶段目标 | 模型原始输出']
for r in examples:lines.append(f"{r['item_id']} | {r['stage']} | {r['expected']} | {r['raw_output']}")
lines+=['','fact_0003是确定的Unicode书写差异：Istanbul与İstanbul，标准化后可匹配。',
        "fact_0001问Which city...，标签为Antigua，输出为St. John's；问题措辞与目标标签值得人工核验。没有外部核验前，不将其计为匹配器修复或模型事实错误。",
        '两条conflict例子分别输出原答案Copenhagen/Indianapolis，而修改目标为Copenhangover/Indiopolis，Unicode修正不会改变其未跟随修改答案的判定。',
        '', '解释边界：筛选确实严格，但这批候选的主要损失在首次问题：228/256未匹配。额外四个指令变体只淘汰7/28个首轮通过者。不能仅凭11/256断言模型没有相关知识，也不能把主要损失都归因于五遍全对；答案别名、源标注、问题难度和模型知识覆盖仍需进一步核验。',
        '这里没有重新运行任何模型、推测未执行阶段、修改原始输出、重新学习方向或更改主实验结果。']
(ROOT/'audit/filter_attrition.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'attrition':dict(attrition),'recovered_observed_rows':len(recovered),'newly_verified_eligible':len(set(verified)-set(original))}))
