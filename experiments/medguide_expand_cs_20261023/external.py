"""Keep external tasks separate; reproduce source formulas before synthetic edits."""
import argparse,ast,collections,csv,difflib,hashlib,json,re
from pathlib import Path
from prepare import write,sha

def weight_rule(bmi,actual,ideal,cut=25):
 return ('underweight',actual) if bmi<18.5 else ('normal',min(ideal,actual)) if bmi<cut else ('adjusted',ideal+0.4*(actual-ideal))
def calc(e,cut=25):
 def val(k):return float(e[k][0])
 def unit(k):return e[k][1].lower()
 w=val('weight');h=val('height');cr=val('creatinine')
 if unit('weight') in ('lb','lbs','pounds'):w*=0.45359237
 elif unit('weight')!='kg':raise ValueError('weight unit '+unit('weight'))
 hu=unit('height')
 if hu=='cm':h/=100
 elif hu in ('in','inches'):h*=0.0254
 elif hu!='m':raise ValueError('height unit '+hu)
 cu=unit('creatinine')
 if cu in ('µmol/l','μmol/l','umol/l'):cr*=0.011312
 elif cu=='mg/l':cr/=10
 elif cu!='mg/dl':raise ValueError('creatinine unit '+cu)
 female=e['sex'].lower()=='female';ibw=(45.5 if female else 50)+2.3*(h*39.3701-60);bmi=w/h**2;branch,weight=weight_rule(bmi,w,ibw,cut)
 return {'bmi':bmi,'branch':branch,'weight_used':weight,'ideal_weight':ibw,'actual_weight':w,'crcl':(140-val('age'))*weight*(0.85 if female else 1)/(cr*72)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.private);out=Path(a.output);summary={}
 file=root/'ncbi__MedCalc-Bench__test_data_11_18_final.csv';rows=list(csv.DictReader(file.open()));records=[]
 for r in rows:
  if 'bmi' not in r['Question'].lower() or 'adjusted body weight' not in r['Question'].lower():continue
  base={'source_row':r['Row Number'],'note_id':r['Note ID'],'calculator':r['Calculator Name'],'source_file_sha256':sha(file),'source_url':'https://huggingface.co/datasets/ncbi/MedCalc-Bench','revision':'83a1a56d6b53f8d4c0e03f0ba9699becf524a595','license':'CC-BY-SA-4.0; derived records retain same license','published_gold':float(r['Ground Truth Answer'])}
  try:
   e=ast.literal_eval(r['Relevant Entities']);old=calc(e);bmi=old['bmi'];newcut=bmi+0.5 if bmi>=25 else max(18.50001,bmi-0.5);new=calc(e,newcut);keepcut=(25+bmi)/2;keep=calc(e,keepcut)
   base.update(entities=e,old=old,new=new,new_threshold=newcut,preserving_threshold=keepcut,preserving=keep,within_source_tolerance=float(r['Lower Limit'])<=old['crcl']<=float(r['Upper Limit']),branch_flipped=old['branch']!=new['branch'],value_changed=abs(old['crcl']-new['crcl'])>1e-8,preserve_ok=old['branch']==keep['branch'] and abs(old['crcl']-keep['crcl'])<1e-8,synthetic=True,comparison='numeric calculator answer and BMI branch; separate from MedGUIDE MCQA; BMI<18.5 underweight remains fixed')
  except (KeyError,ValueError,TypeError) as e:base['error']=str(e)
  records.append(base)
 write(out/'medcalc_bmi.jsonl',records);summary['medcalc']={'test_rows':len(rows),'bmi_cases':len(records),'formula_only_cases_excluded':sum('adjusted body weight' in r['Question'].lower() and 'bmi' not in r['Question'].lower() for r in rows),'computed':sum('old' in r for r in records),'within_source_tolerance':sum(r.get('within_source_tolerance',False) for r in records),'branch_flips':sum(r.get('branch_flipped',False) for r in records),'value_changes':sum(r.get('value_changed',False) for r in records),'preserve_ok':sum(r.get('preserve_ok',False) for r in records),'errors':[r for r in records if 'error' in r],'source_version':'requested ncbi v1.2; card points to newer Verified repository; do not mix versions'}
 driftfile=root/'RDBH__DriftMed__advice_pairs.jsonl'
 if driftfile.exists():
  dr=[json.loads(s) for s in driftfile.read_text().splitlines()];pairs=[]
  for r in dr:
   c=r.get('correct','');w=r.get('wrong','');cn=re.findall(r'\d+(?:\.\d+)?',c);wn=re.findall(r'\d+(?:\.\d+)?',w)
   if cn!=wn:pairs.append({'id':r.get('id'),'disease':r.get('disease'),'correct_numeric_tokens':cn,'wrong_numeric_tokens':wn,'changed_spans':[{'correct':c[i:j],'wrong':w[k:l]} for tag,i,j,k,l in difflib.SequenceMatcher(None,c,w).get_opcodes() if tag!='equal'],'source_sha256':sha(driftfile),'source_url':'https://huggingface.co/datasets/RDBH/DriftMed','interpretation':'published correct/wrong advice pair, not verified chronological guideline revision','comparable_to':'rule-text numeric perturbations; no patient scalar or gold branch inferred'})
  write(out/'driftmed_numeric_pairs.jsonl',pairs);summary['driftmed']={'advice_pairs':len(dr),'numeric_change_candidates':len(pairs),'interpretation':'manual incorrect variants; not evidence of historical threshold change','patient_execution':'not yet supported: advice pairs do not provide patient numeric values with executable branches'}
 elif (root/'RDBH__DriftMed__ConflictMed_v2.jsonl').exists():
  file=root/'RDBH__DriftMed__ConflictMed_v2.jsonl';dr=[json.loads(s) for s in file.read_text().splitlines()];byid={r['id']:r for r in dr};pairs=[];counts=collections.Counter();failed=[]
  for r in dr:
   if r['label']!='correct':continue
   other=byid.get(r['id']+2145)
   if not other or other['label']!='wrong' or any(r[k]!=other[k] for k in ('disease','bias_type','change_category')):failed.append(r['id']);continue
   counts[r['change_category']]+=1
   if r['change_category']!='Diagnostic & Threshold Adjustments':continue
   oldnums=re.findall(r'\d+(?:\.\d+)?',r['text']);newnums=re.findall(r'\d+(?:\.\d+)?',other['text'])
   pairs.append({'correct_id':r['id'],'wrong_id':other['id'],'disease':r['disease'],'bias_type':r['bias_type'],'category':r['change_category'],'correct_numbers':oldnums,'wrong_numbers':newnums,'numeric_tokens_changed':oldnums!=newnums,'source_sha256':sha(file),'correct_text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'wrong_text_sha256':hashlib.sha256(other['text'].encode()).hexdigest(),'revision':'5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e','interpretation':'source correctness pair; numeric differences may include background values; not chronological guideline update or patient-specific branch gold','license':'Research Use Only per source card; source scenario text not redistributed'})
  write(out/'driftmed_threshold_candidates.jsonl',pairs);summary['driftmed']={'scenario_rows':len(dr),'validated_pairs':sum(counts.values()),'pairing_failures':failed,'diagnostic_threshold_pairs':len(pairs),'pairs_with_numeric_changes':sum(r['numeric_tokens_changed'] for r in pairs),'interpretation':'manually crafted incorrect variants; numeric changes are candidates requiring predicate extraction; no invented patient branch gold','source_revision':'5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e','source_file_sha256':sha(file),'source_url':'https://huggingface.co/datasets/RDBH/DriftMed','comparison':'separate correctness task; do not pool with MedGUIDE MCQA or MedCalc numeric answers'}
 else:summary['driftmed']={'status':'download unavailable; see external_sources.json','interpretation':'official card describes manually crafted incorrect variants, not chronological real guideline revisions','card_url':'https://huggingface.co/datasets/RDBH/DriftMed/blob/5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e/README.md'}
 (out/'external_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
