from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,json,sys
from pathlib import Path
FOLDER=ROOT/'review_sol'
import historical_review_io as review_io
B=vars(review_io)
def load(index):
 return json.loads((FOLDER/('source_%03d.json'%index)).read_text())
def prepare(index):
 data=DATA
 nodes={x['group_id']:x for x in B['read_rows'](data/'nodes.jsonl') if x['split']=='fit'}
 views={x['view_id']:x for x in B['read_rows'](data/'views.jsonl') if x['split']=='fit'}
 pairs=[x for x in B['read_rows'](data/'pairs.jsonl') if x['split']=='fit' and x['kind'] in ['A','P','C','F']]
 sources=list(dict.fromkeys(nodes[p['group_id']]['source_document'] for p in pairs))
 path=sources[index];grouped=collections.defaultdict(list)
 for pair in pairs:
  if nodes[pair['group_id']]['source_document']==path:grouped[pair['group_id']].append(pair)
 pack={'source_index':index,'source_document_path':path,'source_document_full_text':source_document_path(path).read_text(),'trees':[],'edits':[],'groups':[]}
 def intern(seq,value):
  if value not in seq:seq.append(value)
  return seq.index(value)
 for gid,pp in grouped.items():
  node=nodes[gid];base=B['split_prompt'](views[pp[0]['left']]['prompt']);leaves=B['leaf_actions'](base['tree'])
  group={'group_id':gid,'case_id':node['case_id'],'patient':node['patient'],'current_node':node['gid'],'current_condition':base['condition'],'declared_original_label':node['original_label'],'ancestor_conditions':node['ancestors'],'tree_ref':intern(pack['trees'],base['tree']),'pairs':[]}
  for pair in pp:
   left=B['split_prompt'](views[pair['left']]['prompt']);right=B['split_prompt'](views[pair['right']]['prompt'])
   group['pairs'].append({'pair_id':pair['pair_id'],'expected_kind':pair['kind'],'declared_left_label':pair['left_label'],'declared_right_label':pair['right_label'],'left_edit_ref':intern(pack['edits'],B['edits_for_side'](base,left,leaves)),'right_edit_ref':intern(pack['edits'],B['edits_for_side'](base,right,leaves))})
  pack['groups'].append(group)
 FOLDER.mkdir(exist_ok=True)
 (FOLDER/('source_%03d.json'%index)).write_text(json.dumps(pack,ensure_ascii=False,indent=2))
 return {'source_index':index,'groups':len(pack['groups']),'pairs':sum(len(g['pairs']) for g in pack['groups']),'trees':len(pack['trees']),'edits':len(pack['edits']),'source_chars':len(pack['source_document_full_text'])}
def persist(index,batch,decisions):
 pack=load(index);groups={g['group_id']:g for g in pack['groups']};pairs={p['pair_id']:(g,p) for g in pack['groups'] for p in g['pairs']}
 output=FOLDER/'historical_results.jsonl'
 done={r['pair_id'] for r in B['read_rows'](output)} if output.exists() else set()
 action_reasons={1:'Only leaf actions are deleted; all protected conditions, definitions and patient facts remain identical.',2:'Only leaf actions are replaced by administrative text; all protected conditions, definitions and patient facts remain identical.',3:'Only the inspected original leaf actions are permuted with leaf identifiers retained; all protected conditions, definitions and patient facts remain identical.'}
 ids=[]
 for decision in decisions:
  selected=[(groups[decision['group_id']],p) for p in groups[decision['group_id']]['pairs'] if p['expected_kind']=='A'] if 'group_id' in decision else [pairs[decision['pair_id']]]
  for group,pair in selected:
   if pair['pair_id'] in done:raise ValueError('already reviewed: '+pair['pair_id'])
   checks=dict(decision['checks']);checks['duplicate']='DEFERRED_DETERMINISTIC'
   reason=decision['reason']
   if 'group_id' in decision:reason+=' '+action_reasons[pair['right_edit_ref']]
   status='FAIL' if 'FAIL' in checks.values() else 'UNCERTAIN' if 'UNCERTAIN' in checks.values() or decision['left'] is None or decision['right'] is None else 'PASS'
   item={'status':status,'checks':checks,'independent_left_label':decision['left'],'independent_right_label':decision['right'],'reason':reason}
   result=B['validated_result'](item,pair,group,'gpt-6.1-sol',None);result.update(reasoning_effort='xhigh',reviewer='agent_b_sol',review_batch=batch,source_index=index)
   B['append_json'](output,result);ids.append(pair['pair_id']);done.add(pair['pair_id'])
 B['append_json'](FOLDER/'read_batches.jsonl',{'batch':batch,'source_index':index,'pair_ids':ids,'review_model':'gpt-6.1-sol','reasoning_effort':'xhigh','round':1,'time':B['now']()})
 (FOLDER/'completed_pair_ids.json').write_text(json.dumps(sorted(done),indent=2))
 stats=collections.Counter(r['status'] for r in B['read_rows'](output))
 B['write_json'](FOLDER/'progress.json',{'reviewed':len(done),'total':5022,'status':dict(stats),'last_source':index,'updated_at':B['now']()})
 return {'persisted':len(ids),'total_persisted':len(done),'status':dict(stats)}
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','source','structure','groups','persist','cases','nodes']);p.add_argument('index',type=int);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int);p.add_argument('--batch');a=p.parse_args()
 if a.mode=='prepare':result=prepare(a.index)
 elif a.mode=='persist':result=persist(a.index,a.batch,json.load(sys.stdin))
 else:
  pack=load(a.index)
  if a.mode=='source':print(pack['source_document_full_text'][a.start:a.end]);result=None
  elif a.mode=='structure':result={'trees':pack['trees'],'edits':pack['edits']}
  elif a.mode=='cases':result={g['case_id']:g['patient'] for g in pack['groups']}
  elif a.mode=='nodes':
   conditions=[];nodes=[]
   for g in pack['groups'][a.start:a.end]:
    node={k:v for k,v in g.items() if k not in ['patient','ancestor_conditions']};anc=[]
    for a_cond in g['ancestor_conditions']:
     text={'gid':a_cond['gid'],'condition':a_cond['condition']}
     if text not in conditions:conditions.append(text)
     anc.append({'condition_ref':conditions.index(text),'label':a_cond['label']})
    node['ancestor_conditions']=anc;nodes.append(node)
   result={'ancestor_condition_texts':conditions,'nodes':nodes}
  else:result=pack['groups'][a.start:a.end]
  B['append_json'](FOLDER/'read_operations.jsonl',{'source_index':a.index,'field':a.mode,'start':a.start,'end':a.end,'time':B['now']()})
 if result is not None:print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
