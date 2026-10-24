import hashlib,json
from pathlib import Path
E=Path(__file__).parent;O=E/'results';P=Path('/home/gcx/cs44_explicit_check_20261024/private')
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
s=json.loads((O/'summary.json').read_text());g=s['groups'];workers=[json.loads((O/f'worker{k}.json').read_text()) for k in range(2)];focus=g['focus18'];rows=read(O/'scores.jsonl');details={r['cell_key']:r for k in range(2) for r in read(P/f'outputs{k}.jsonl')}
lines=['# 44 例裁剪前显式核对：结果','本轮固定PR28的44例和176格，原对照全部直接复用。新增核对只看患者原文与两候选条件；金标、K/U、原闭卷答案、18例关注标签均只用于事后评分。',f"实际新增核对 {s['check_calls']} 次，新增最终回答 {s['new_final_calls']} 次，其余 {s['final_cached_cells']}/176 个最终回答复用缓存；无法解析或无法确定的核对 {s['fallback_cells']} 格保留原两候选，没有金标回退。",'## 原有两规则对照 → 显式核对后裁剪','|阈值／顺序|全部44|K20|U24|关注K18|','|---|---:|---:|---:|---:|']
for key,name in [('original/current','原阈值／当前'),('original/reversed','原阈值／反转'),('changed/current','新阈值／当前'),('changed/reversed','新阈值／反转')]:
 vals=[]
 for st in ['all','K','U','focus18']:
  x=g[st]['cells'][key];vals.append(f"{x['baseline_correct']}→{x['correct']} / {x['n']}")
 lines.append('|'+name+'|'+'|'.join(vals)+'|')
lines+=['## 18例固定旧答案的K病例',f"新阈值下，两种顺序均正确的病例为 {focus['changed_both_correct']}/18，至少一种顺序正确为 {focus['changed_any_correct']}/18；四格全部正确为 {focus['all4_correct']}/18。原阈值两种顺序均保持正确为 {focus['original_both_correct']}/18。仍在四格固定旧答案为 {focus['fixed_old_all4']}/18。病例级与格子级指标分开，不把36个新阈值格子当作36个病例。",'## 损伤与恢复，以及错误发生在哪一步','|组／阈值／顺序|救回|损伤|条件核对正确|核对正确但最终错|未确定回退|','|---|---:|---:|---:|---:|---:|']
for st in ['K','U','focus18']:
 for key,x in g[st]['cells'].items():lines.append(f"|{st}/{key}|{x['gained']}|{x['lost']}|{x['check_correct']}/{x['n']}|{x['right_check_wrong_answer']}|{x['fallback']}|")
lines+=['## 阈值翻转与顺序稳定性','|组别|阈值配对双侧均正确|只交换顺序就改变答案|四格全对病例|','|---|---:|---:|---:|']
for st in ['all','K','U','focus18']:
 x=g[st];lines.append(f"|{st}|{x['correct_threshold_pairs']}/{x['threshold_pairs']}|{x['order_changed_pairs']}/{x['order_pairs']}|{x['all4_correct']}/{x['cases']}|")
lines+=['同一病例贡献两次阈值配对和两次顺序配对；这是病例内比较，不增加独立样本数。','## 主要发现及剩余错误','18例固定旧答案的K病例在新阈值下两种顺序均恢复18/18；其中16/18还能在原阈值两种顺序都答对，另外2例有原阈值损伤。因此达到救回新阈值错误的目标，但没有完全实现保留原阈值表现。U原阈值保持24/24，新阈值当前顺序17→22、反转22→24；当前顺序有1例正确→错误，净提升不能掩盖这项损伤。','相对于冻结规则与金标，剩余5个错误格子分属4病例：medguide_03504在原阈值两顺序把7 mm结节总径替换为3 mm实性成分，选错分支；medguide_03777的反转条件核对同时提及总径7 mm与实性成分2 mm，仍称结节<6 mm；medguide_07067把4 mm判成小于3.6 mm；medguide_04050把3–5 mm范围整体判成小于4.5 mm。显式写出比较能提高表现，但不保证测量对象选择与算术正确。当前没有核对选对分支后最终仍答错的格子。','## 格式解析修正','初始解析仅接受行首Branch标记，漏掉19条正文内明确的Branch: A/B。修正为读取最后一次明确标记后，直接重新解析原始核对文本，并全部复用对应最终答案缓存，新增模型调用0。另有1格仍未给出可识别选择，按既定规则保留两候选。初始输出、worker哈希、重解析记录均保留；没有按金标或是否答对决定解析结果。','## 方法与解释边界','核对要求显式说明患者类型、数量、大小，并与提示中的阈值比较；模型选择A/B后，使用原裁剪提示作答。没有把核对解释拼入最终回答，也未使用人工指定patient_value或source_path。条件中原有的随访信息保留。核对模型看到两条条件，最终回答模型看到所选分支及原选项。','这轮改变了分支选择方式，同时最终上下文从两条规则变成一条；对照只能说明整个“显式核对→裁剪”流程效果，不能单独归因于数值比较文字。44例都已召回旧/新两个目标分支，不能说明在完整800例上能解决候选遗漏。K/U和阈值方向、结节类型仍有构成差异。','## 核对样例与剩余错误','verifier_examples.jsonl 给出少量核对输出、所选分支与事后评分。全部核对原文保存在服务器private目录。错误分析只在生成结束后合并金标。','## 执行与验证',f"GPU0/1各88次核对；耗时分别 {workers[0]['seconds']:.1f}/{workers[1]['seconds']:.1f} 秒，峰值分配显存 {workers[0]['peak_gib']:.2f}/{workers[1]['peak_gib']:.2f} GiB。核对160 token上限，最终回答8 token；Qwen3-8B BF16 eager greedy。",'校验176格完整唯一、worker输入不含评估字段、全部旧对照直接读取、实际调用数、原始结果哈希，以及逐格救回−损伤等于净变化。未修改本次提示或根据结果调参，未启动后续实验。']
# A few transparent factual checks, never the full patient input.
examples=[];seen=set()
for pred in [lambda r:r['focus18'] and r['threshold_state']=='changed' and r['correct'],lambda r:r['focus18'] and r['threshold_state']=='changed' and not r['correct'],lambda r:r['stratum']=='U' and r['baseline_correct'] and not r['correct']]:
 for r in [r for r in rows if pred(r)][:2]:
  if r['cell_key'] not in seen:examples.append({**r,'check_completion':details[r['cell_key']]['check_completion']});seen.add(r['cell_key'])
(O/'verifier_examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples));(O/'REPORT.zh.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n')
for group in g.values():
 for c in group['cells'].values():assert c['correct']-c['baseline_correct']==c['gained']-c['lost']
# Worker input shape: labels exist only in the post-hoc evaluation file.
for k in range(2):
 for r in read(P/f'requests{k}.jsonl'):assert set(r)=={'cell_key','check_key','check_prompt','routes','options'}
(O/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(O.iterdir())+[E/'trial.py',E/'analyze.py',E/'report.py',E/'run.sh',E/'README.zh.md',E/'reparse.py',E/'test_parse.py'] if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print('Report and count checks complete')
