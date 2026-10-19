"""Freeze new synthetic probes from unused complete MEGA-CDP guideline trees.

Read only decision_trees.jsonl and historic result identifiers. Never read dataset
patient vignettes, case_paths, gt_paths, patient answers or model predictions.
Third-party full guideline/action text remains in server-only private/ files.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import re

BASE=Path(__file__).resolve().parent
# Hand-prepared node decomposition, verified against complete source tree edges.
SPECS={
42:{'leaf':3,'edges':[(1,'yes'),(2,'yes')],'logic':{'op':'AND','args':['c1',{'op':'OR','args':['c2','c3']},'c4']},
 'atoms':[
  {'node':1,'fragment':'undergoing transfer within the hospital','text':'The patient is undergoing transfer within the hospital.','value':'T','true':'The patient is being transferred between units within the hospital.','false':'The patient is not being transferred within the hospital.'},
  {'node':1,'fragment':'anaesthetised','text':'The patient is anaesthetised during the current transfer.','value':'F','true':'The patient is anaesthetised during the current transfer.','false':'The patient is not anaesthetised during the current transfer.'},
  {'node':1,'fragment':'sedated','text':'The patient is sedated during the current transfer.','value':'T','true':'The patient is sedated during the current transfer.','false':'The patient is not sedated during the current transfer.'},
  {'node':2,'fragment':'an airway device remain in place during transfer','text':'An airway device remains in place during the current transfer.','value':'T','true':'An airway device remains in place during this transfer.','false':'No airway device is in place during this transfer.'}],
 'negation':'It is not correct that the airway device was removed; the device remains in place during this transfer.',
 'missing':'Whether an airway device remains in place during this transfer is not documented.',
 'audit':'Source node1 means transfer AND (anaesthetised OR sedated); source node2 requires airway device. Preserve OR explicitly. Original observations set anaesthesia F and sedation T, so either mode is not silently required. All observations refer to the same transfer.'},
44:{'leaf':10,'edges':[(1,'yes'),(2,'no')],'logic':{'op':'AND','args':['c1','c2',{'op':'NOT','args':['c3']}]},
 'atoms':[
  {'node':1,'fragment':'biopsy sample','text':'The current breast specimen is a biopsy sample.','value':'T','true':'The current breast specimen is a biopsy sample.','false':'The current breast specimen is a surgical resection, not a biopsy.'},
  {'node':1,'fragment':'pre-neoadjuvant therapy (pre-NAT)','text':'The current breast specimen was obtained before neoadjuvant therapy.','value':'T','true':'The current breast specimen was obtained before any neoadjuvant therapy.','false':'The current breast specimen was obtained after neoadjuvant therapy.'},
  {'node':2,'fragment':'histological type according to the latest WHO Classification of Breast Tumors been determined on the biopsy','text':'The biopsy histological type has been determined according to the latest WHO Classification of Breast Tumors.','value':'F','true':'The biopsy histological type has been determined and documented according to the latest WHO Classification of Breast Tumors.','false':'The biopsy histological type has explicitly not been determined according to the latest WHO Classification of Breast Tumors.'}],
 'negation':'It is not true that WHO histological type classification was completed; the biopsy histological type remains undetermined.',
 'missing':'The record does not say whether WHO histological type classification was completed for this biopsy.',
 'audit':'Source node1 is pre-NAT biopsy, decomposed into specimen type and treatment-relative time without numerical thresholds. Source node2 asks whether classification HAS been determined; its NO edge is represented by NOT(c3). Explicit incomplete classification differs from unknown documentation. Source leaf10 is an action, never a patient fact.'},
80:{'leaf':3,'edges':[(1,'yes'),(2,'yes')],'logic':{'op':'AND','args':['c1','c2','c3']},
 'atoms':[
  {'node':1,'fragment':'pregnant woman','text':'The patient is pregnant.','value':'T','true':'The woman is currently pregnant.','false':'The woman is not pregnant.'},
  {'node':1,'fragment':'Rh D negative blood type','text':'The patient has Rh D negative blood type.','value':'T','true':'The patient has confirmed Rh D negative blood type.','false':'The patient has confirmed Rh D positive blood type.'},
  {'node':2,'fragment':'booking (early pregnancy) visit','text':'The current appointment is the booking visit in early pregnancy.','value':'T','true':'The current appointment is explicitly the booking visit in early pregnancy.','false':'The current appointment is a later follow-up visit, explicitly not the booking visit.'}],
 'negation':'The current appointment is not a later follow-up visit; it is explicitly the booking visit in early pregnancy.',
 'missing':'Whether this appointment is the booking visit or a later follow-up visit is not documented.',
 'audit':'Source node1 is pregnant AND RhD negative, explicitly decomposed. All authored patients are women, so sex is a fixed disclosed population context rather than inferred. Source node2 booking event is stated directly, not derived from gestational age.'},
53:{'leaf':43,'edges':[(1,'no')],'logic':{'op':'NOT','args':['c1']},
 'atoms':[
  {'node':1,'fragment':'diabetes','text':'The patient has diabetes.','value':'F','true':'The patient has a confirmed diagnosis of diabetes.','false':'The patient has been assessed and explicitly does not have diabetes.'}],
 'negation':'It is not correct to describe this patient as diabetic; diabetes is explicitly absent.',
 'missing':'The record contains no assessment establishing whether the patient has diabetes.',
 'audit':'Source root asks whether the patient has diabetes; its explicit NO edge leads to leaf43, which directs use of other pain guidance. The condition remains positive diabetes and the rule uses NOT(c1); this is a source-defined fallback, not a clinical claim about treatment efficacy.'}}


def sha(v):return hashlib.sha256(v if isinstance(v,bytes) else v.encode()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def jsonl(p,v):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in v))

def truth(expr,values):
 if isinstance(expr,str):return values.get(expr,'U')
 args=[truth(x,values) for x in expr['args']]
 if expr['op']=='NOT':return {'T':'F','F':'T','U':'U'}[args[0]]
 if expr['op']=='AND':return 'F' if 'F' in args else 'U' if 'U' in args else 'T'
 if expr['op']=='OR':return 'T' if 'T' in args else 'U' if 'U' in args else 'F'
 raise ValueError(expr)

def proofs(logic,values):
 label=truth(logic,values)
 if label=='U':return [[]]
 ids=list(values);result=[]
 for size in range(1,len(ids)+1):
  for subset in itertools.combinations(ids,size):
   if any(set(prev)<=set(subset) for prev in result):continue
   if truth(logic,{k:values[k] for k in subset})==label:result.append(list(subset))
 return [[int(c[1:]) for c in group] for group in result]

def walk(n,trace=()):
 if n.get('leaf'):yield trace,n
 else:
  for b in ('yes','no'):
   if isinstance(n.get(b),dict):yield from walk(n[b],trace+((n,b),))

def source_span(raw,description,node_index,fragment=None):
 indices=list(re.finditer(r'"index"\s*:\s*(\d+)',raw))
 positions=[i for i,m in enumerate(indices) if int(m.group(1))==node_index]
 if len(positions)!=1:raise ValueError('Source node index is not unique')
 i=positions[0];lo=indices[i].end();hi=indices[i+1].start() if i+1<len(indices) else len(raw)
 found=[m for m in re.finditer(r'"(?:[^"\\]|\\.)*"',raw) if lo<=m.start()<hi and json.loads(m.group())==description]
 if len(found)!=1:raise ValueError('Source node string is not unique')
 match=found[0];start,end=match.start()+1,match.end()-1
 if fragment:
  if fragment not in description:raise ValueError(f'Missing source fragment: {fragment}')
  # Selected fragments are ASCII without JSON escapes; refuse ambiguous encoding.
  target=json.dumps(fragment,ensure_ascii=False)[1:-1];body=raw[start:end]
  if body.count(target)!=1:raise ValueError('Source fragment not unique or encoded differently')
  start+=body.index(target);end=start+len(target)
 return {'start':start,'end':end,'unit':'character','source':'original_MEGA_CDP_tree_text','encoding':'JSON_string_content'}

def history_audit(root):
 rows=[];used=set()
 for path in sorted(root.rglob('*')):
  if not path.is_file() or path.suffix not in {'.json','.jsonl'}:continue
  raw=path.read_bytes()
  try:text=raw.decode()
  except UnicodeDecodeError:continue
  ids=set(map(int,re.findall(r'"guideline_id"\s*:\s*(\d+)',text)))
  ids.update(map(int,re.findall(r'(?:case_v2_|case_|mega_)(\d+)_',text)))
  ids.update(map(int,re.findall(r'cached_gid(\d+)',text)))
  used.update(ids);rows.append({'path':str(path),'sha256':sha(raw),'bytes':len(raw),'guideline_ids':sorted(ids)})
 return rows,used

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/gcx/TRIAGE/data/MEGA-CDP/train_val/decision_trees.jsonl'));p.add_argument('--history',type=Path,default=Path('/home/gcx/TRIAGE/results'));p.add_argument('--output',type=Path,default=BASE/'data');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Refuse to overwrite frozen exploratory data')
 audit,used=history_audit(a.history)
 if set(SPECS)&used:raise ValueError(f'Guidelines overlap existing experiments: {sorted(set(SPECS)&used)}')
 raw_file=a.source.read_bytes();chosen={r['guideline_id']:r for r in map(json.loads,raw_file.decode().splitlines()) if r['guideline_id'] in SPECS}
 if set(chosen)!=set(SPECS):raise ValueError('Selected complete trees missing')
 a.output.mkdir(parents=True);private=a.output/'private';private.mkdir()
 rules=[];inputs=[];labels=[];pairs=[];manifests=[];reviews=[];docs=[]
 for ordinal,(gid,spec) in enumerate(SPECS.items(),1):
  source=chosen[gid];raw=source['text'];tree=json.loads(raw)['decision_tree'];trace,leaf=next((t,n) for t,n in walk(tree) if n['index']==spec['leaf'])
  assert [(n['index'],b) for n,b in trace]==spec['edges']
  nodes={n['index']:n for n,b in trace};rid=f'E{ordinal:02}';text='Prepared atomic conditions (combine with the supplied explicit logic):\n';conditions=[]
  for i,atom in enumerate(spec['atoms'],1):
   start=len(text);text+=atom['text'];end=len(text);text+='\n'
   conditions.append({'condition_id':f'c{i}','field':f'mega{gid}_atom{i}','text':atom['text'],'criterion_kind':'inclusion','source_span':{'start':start,'end':end,'unit':'character','source':'offline_normalized_rule_pack'},'original_source_span':source_span(raw,nodes[atom['node']]['description'],atom['node'],atom['fragment']),'source_node_index':atom['node'],'source_node_fragment_sha256':sha(atom['fragment'])})
  text+='Logic: '+json.dumps(spec['logic'],sort_keys=True)+'\nRecommendation: ';astart=len(text);text+=leaf['conclusion'];action_span={'start':astart,'end':len(text),'unit':'character','source':'offline_normalized_rule_pack'}
  original_action=source_span(raw,leaf['conclusion'],leaf['index'])
  provenance={'dataset':'MEGA-CDP','split':'train_val','guideline_id':gid,'filename':source['filename'],'leaf_index':leaf['index'],'source_tree_sha256':sha(raw),'source_file_sha256':sha(raw_file),'source_tree_edges':spec['edges'],'patient_reference_paths_used':False,'reference_answers_used':False,'source_license':'TBD; full trees/actions server-only','scope':'all stated tree-branch preconditions; explicit atomic decomposition','fixed_population_context':'woman' if gid==80 else None}
  rule={'rule_id':rid,'text':text,'conditions':conditions,'logic':spec['logic'],'exceptions':[],'exception_logic':{'op':'OR','args':[]},'action':leaf['conclusion'],'action_span':action_span,'source_span':{'start':0,'end':len(text),'unit':'character','source':'offline_normalized_rule_pack'},'original_action_span':original_action,'provenance':provenance};rules.append(rule);docs.append({'rule_id':rid,'guideline_id':gid,'text':raw})
  manifests.append({'rule_id':rid,'provenance':provenance,'logic':spec['logic'],'condition_count':len(conditions),'conditions':[{k:v for k,v in c.items() if k not in {'text','field','criterion_kind'}}|{'prepared_text_sha256':sha(c['text'])} for c in conditions],'original_action_span':original_action,'action_sha256':sha(leaf['conclusion']),'rule_pack_sha256':sha(text),'action_span':action_span})
  values={f'c{i}':x['value'] for i,x in enumerate(spec['atoms'],1)};assert truth(spec['logic'],values)=='T'
  reviews.append({'rule_id':rid,'guideline_id':gid,'manual_audit':spec['audit'],'atomic_condition_count':len(conditions),'logic':spec['logic'],'source_tree_sha256':sha(raw),'labels':'authored explicit observations + strong three-valued logic','clinical_expert_review':False,'reference_case_path_read':False})
  original=[x['true' if x['value']=='T' else 'false'] for x in spec['atoms']];last=spec['atoms'][-1];opp='false' if last['value']=='T' else 'true';k=len(original);pairid=f'EP{ordinal:02}'
  variants={'original':original,'counterfactual':original[:-1]+[last[opp]],'partial_missing':original[:-1]+[spec['missing']],'negation':original[:-1]+[spec['negation']],'multiple_values':original[:-1]+["A different patient's record states: "+last[opp]+" That record does not describe this patient. For this patient, the verified finding is: "+original[-1]],'contradiction':["Signed report A for this patient's same assessment: "+original[-1],"Signed report B for this patient's same assessment: "+last[opp]+' Both reports remain signed; neither is withdrawn or superseded.']}
  for variant,sentences in variants.items():
   vals=dict(values)
   if variant=='counterfactual':vals[f'c{k}']='F' if last['value']=='T' else 'T'
   elif variant=='partial_missing':vals[f'c{k}']='U'
   elif variant=='contradiction':vals={c:'U' for c in vals}
   label=truth(spec['logic'],vals);assert label==('F' if variant=='counterfactual' else 'U' if variant in {'partial_missing','contradiction'} else 'T')
   row={'sample_id':f'{pairid}_{variant}','rule_id':rid,'sentences':sentences};inputs.append(row)
   labels.append({'sample_id':row['sample_id'],'rule_id':rid,'pair_id':pairid,'variant':variant,'label':label,'gold_atomic_states':vals,'acceptable_evidence_proofs':proofs(spec['logic'],vals),'all_relevant_evidence_ids':list(range(1,k+1)) if label in {'T','F'} else [1,2] if variant=='contradiction' else list(range(1,k)),'conflict_evidence_ids':[1,2] if variant=='contradiction' else [],'synthetic_patient':True,'gold_source':'authored observations and explicit AND/OR/NOT; no patient reference pathway','clinical_expert_review':False,'input_sha256':sha(json.dumps(row,sort_keys=True))})
  pairs.append({'pair_id':pairid,'rule_id':rid,'original_id':f'{pairid}_original','counterfactual_id':f'{pairid}_counterfactual'})
 for name,rows in [('inputs_public.jsonl',inputs),('labels_public.jsonl',labels),('pairs_public.jsonl',pairs),('rule_manifest_public.jsonl',manifests),('annotation_audit.jsonl',reviews)]:jsonl(a.output/name,rows)
 jsonl(private/'rules.jsonl',rules);jsonl(private/'source_trees.jsonl',docs);jsonl(private/'history_files.jsonl',audit)
 dump(a.output/'history_audit.json',{'root':str(a.history),'file_count':len(audit),'selected_guideline_ids':list(SPECS),'matching_selected_ids':sorted(set(SPECS)&used),'used_guideline_ids':sorted(used),'history_file_audit_sha256':sha(json.dumps(audit,sort_keys=True)),'limit':'No matching IDs in available result JSON/JSONL and cached case IDs; unavailable external history not audited.'})
 dump(a.output/'PROTOCOL.json',{'experiment':'new synthetic exploratory logic probes','samples':len(inputs),'rules':len(rules),'atomic_conditions':sum(len(r['conditions']) for r in rules),'trialmatch_requests':sum(len(r['conditions']) for r in rules)*6,'counterfactual_pairs':len(pairs),'labels':dict(collections.Counter(x['label'] for x in labels)),'models':'same frozen V5 prompt, thresholds and precision as regression','independence':'New unused complete trees and entirely authored synthetic patient facts, no clinical gold','clinical_expert_review':False,'reference_patient_paths_or_answers_used':False,'dataset_files_read':[str(a.source)],'third_party_license':'TBD; source trees and actions remain server-only','evidence_policy':'Same strict V5 sufficient-proof policy. Missing U expects empty evidence; conflict U exactly two conflict reports.','evidence_limit':'Short explicit synthetic facts, not long-note retrieval','schema_review':'Before any exploratory inference, compound prototype was replaced by explicit atomic AND/OR/NOT rules; see pre_run_schema_review.json and superseded_freeze.json.'})
 freeze={'status':'frozen_before_model_inference','created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(Path('data') / p.relative_to(a.output)):sha(p.read_bytes()) for p in sorted(a.output.rglob('*')) if p.is_file()},'builder_sha256':sha(Path(__file__).read_bytes()),'no_model_output_read':True,'schema_review_sha256':sha((BASE/'pre_run_schema_review.json').read_bytes()),'superseded_freeze_sha256':sha((BASE/'superseded_freeze.json').read_bytes())}
 dump(a.output/'FREEZE.json',freeze)
 print(json.dumps({'samples':len(inputs),'rules':len(rules),'atomic_conditions':sum(len(r['conditions']) for r in rules),'trialmatch_requests':sum(len(r['conditions']) for r in rules)*6,'labels':dict(collections.Counter(x['label'] for x in labels)),'unused_guideline_ids':list(SPECS),'status':'frozen'}))

if __name__=='__main__':main()
