"""Build a human review package from completed editing drafts, never certify pairs."""
import collections,csv,difflib,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
D=Path(__file__).resolve().parent
B=D.parent
C=B/'private/p6_candidates_v1'
R=B/'private/p6_action_drafts_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert (R/'DONE.json').exists(),'Drafting must finish before building the complete review package'
 cases=rows(C/'candidates.jsonl');raw=rows(R/'raw_outputs.jsonl')
 assert len(cases)==len(raw)==160
 outputs={x['pair_id']:x for x in raw};patches={x['pair_id']:x for x in rows(R/'patches.jsonl')}
 assert len(outputs)==160
 out=B/'private/p6_human_review_v2';out.mkdir(exist_ok=False)
 shutil.copytree(C/'guidelines',out/'native_guidelines')
 shutil.copytree(C/'dossiers',out/'source_dossiers')
 (out/'pairs').mkdir();(out/'decoy_guidelines').mkdir()
 with (C/'review.csv').open(newline='') as f:
  reader=csv.DictReader(f);fields=list(reader.fieldnames);reviews=list(reader)
 records=[]
 for c in cases:
  pid=c['pair_id'];r=outputs[pid];patch=patches.get(pid)
  guide=(C/c['native_guideline_file']).read_text()
  lines=[f'# P6 人工核验候选 {pid}','','**未核验。AI 草稿可能选错分支、包含条件或改错参考路径，不能直接用于实验。**','',
   f'病例资料：[患者、原始路径和映射](../source_dossiers/{pid}.md)',
   f'完整原生指南：[原文](../native_guidelines/{c["source_filename"]}.md)','',
   '## 草稿状态',r['status'],r.get('error',''),'','## 模型草稿（供核验，不是事实判断）',r.get('output') or '无输出']
  record={'pair_id':pid,'draft_status':r['status'],'human_verified':False,'patient_facts':c['patient_facts'],'question':c['question'],'native_reference_path':c['native_reference_path'],'native_guideline_file':'native_guidelines/'+c['source_filename']+'.md','decoy_guideline_file':None,'patch':patch}
  if patch:
   assert hashlib.sha256(guide.encode()).hexdigest()==patch['native_guideline_sha256']
   edited=guide
   for edit in sorted(patch['guide_edits'],key=lambda x:x['start'],reverse=True):
    assert guide[edit['start']:edit['end']]==edit['before']
    edited=edited[:edit['start']]+edit['after']+edited[edit['end']:]
   assert hashlib.sha256(edited.encode()).hexdigest()==patch['decoy_guideline_sha256']
   target=out/'decoy_guidelines'/(pid+'.md');target.write_text(edited)
   record['decoy_guideline_file']='decoy_guidelines/'+pid+'.md'
   diff=''.join(difflib.unified_diff(guide.splitlines(True),edited.splitlines(True),fromfile='native',tofile='decoy',n=3))
   lines+=['','## 完整诱饵指南',f'[待核验改写](../decoy_guidelines/{pid}.md)','','## 逐项文本改动','~~~diff',diff,'~~~','','## 原生参考路径',json.dumps(c['native_reference_path'],ensure_ascii=False,indent=2),'','## 改写参考路径（待核验）',json.dumps(patch['decoy_reference_path'],ensure_ascii=False,indent=2)]
  else:
   lines+=['','尚未生成可定位的文本改动。请记录不可配对原因，或人工给出精确的处置替换及对应参考路径；不要直接认可模型输出。']
  lines+=['','## 核验要求','在 review.csv 填写审阅人、日期和每项判断。确认患者事实、问题、条件、例外及定义不变；只改同一节点竞争分支的处置与相应参考路径；核对目标 span、可接受路径及多节点/跨片段属性。可以拒绝并说明原因，不能按后续模型表现删选。']
  (out/'pairs'/(pid+'.md')).write_text('\n'.join(lines)+'\n');records.append(record)
 with (out/'review.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(reviews)
 (out/'pairs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
 (out/'README.zh.md').write_text('# P6 人工核验包\n\n共 160 个来源候选、20 份开发指南。pairs/ 中逐例列出原文、AI 草稿、文本差异和参考路径。review.csv 尚未填写人工判断。源码可定位不等于语义正确；缺草稿和失败项全部保留。至少 120 对、20 份指南和 40 对多节点/跨片段通过人工核验后，才能认定规模达标。\n\n请保留失败项和修改历史。人工修改处置或参考路径后必须重新生成文件指纹；本包不是冻结数据，也未执行 P6 方法比较。\n')
 stats=dict(collections.Counter(r['status'] for r in raw))
 summary={'status':'HUMAN_REVIEW_PACKAGE_PREPARED_NOT_FROZEN','created_at':datetime.now(timezone.utc).isoformat(),'candidate_count':len(cases),'guideline_count':len({x['guideline_id'] for x in cases}),'source_multinode_candidates':sum(x['multinode_candidate'] for x in cases),'exact_text_drafts_pending_human':len(patches),'draft_statuses':stats,'human_verified_pairs':0,'P6_method_inputs':0,'test_files_read':False,'private_package':str(out),'source_candidates_sha256':sha(C/'candidates.jsonl'),'raw_outputs_sha256':sha(R/'raw_outputs.jsonl'),'patches_sha256':sha(R/'patches.jsonl'),'review_table_sha256':sha(out/'review.csv')}
 dump(out/'manifest.json',summary);dump(D/'results/human_review_package.json',summary)
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
