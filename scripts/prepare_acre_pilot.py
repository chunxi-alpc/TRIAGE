#!/usr/bin/env python3
"""Offline strict numeric sibling screening. Reference metadata never enters inference.

Inspect mode does NOT generate a dataset. Accepted pairs additionally need
explicit incompatible-action review; different text alone is insufficient.
"""
import sys,json,re,hashlib,argparse,collections,operator,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.span_mapping import locate
OPS={'>':operator.gt,'<':operator.lt,'>=':operator.ge,'<=':operator.le,'≥':operator.ge,'≤':operator.le}
FIELDS={
 'age':(r'age|aged|年龄|年齡|岁',r'years?|months?|岁|年|个月'),
 'potassium':(r'(?:serum\s+)?potassium|血钾|血清钾|钾水平',r'mEq/L|mmol/L'),
 'egfr':(r'eGFR|estimated glomerular filtration rate|估算肾小球滤过率',r'mL/min(?:/1\.73\s*m[²2])?'),
 'crcl':(r'(?:creatinine\s+clearance)|肌酐清除率|CrCl',r'mL/min'),
 'sbp':(r'(?:systolic\s+blood\s+pressure)|SBP|收缩压',r'mmHg'),
 'lvef':(r'LVEF|left ventricular ejection fraction|左(?:心)?室射血分数',r'%'),
 'hba1c':(r'HbA1c|糖化血红蛋白',r'%'),
 'psa_dt':(r'PSA\s*(?:doubling\s*time|倍增时间)',r'months?|个月|月'),
 'platelets':(r'(?:platelet\s+count)|血小板计数|血小板',r'(?:×|x)\s*10\^9/L'),
 'alc':(r'ALC|absolute lymphocyte count',r'(?:×|x)\s*10\^9/L')}

def numeric_spec(text):
    # A primitive threshold only; compounded conditions cannot be verified by one edit.
    if re.search(r'\band\b|\bor\b|\be\.g\.|\bwith\b|\bwithout\b|且|或|和|例如',text,re.I):return None
    matches=list(re.finditer(r'(>=|<=|>|<|≥|≤)\s*(\d+(?:\.\d+)?)',text))
    if len(matches)!=1:return None
    m=matches[0]
    for field,(alias,unit) in FIELDS.items():
        if re.search(alias,text[:m.start()],re.I):
            if not re.search(unit,text[m.end():],re.I):continue
            return {'field':field,'operator':m[1],'threshold':float(m[2]),'unit':re.search(unit,text[m.end():],re.I)[0]}
    return None

def observations(text, spec):
    alias,unit=FIELDS[spec['field']]
    if spec['field']=='age':
        pattern=r'(?P<value>\d+(?:\.\d+)?)\s*(?:[- ]?)(?P<unit>years?|months?|岁|年|个月)(?:[- ]?old)?'
    else:
        pattern=r'(?:'+alias+r')(?:\s*(?:is|was|of|at|measured\s+at|为|是|:|：|=|约|approximately|currently))*\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>'+unit+r')'
    obs=list(re.finditer(pattern,text,re.I))
    # No competing repeated measurement, converted value, range, or threshold.
    if len(obs)!=1:return None
    m=obs[0]
    if m['unit'].lower().replace(' ','')!=spec['unit'].lower().replace(' ',''):return None
    start=m.start('value');end=m.end('value')
    a=max(text.rfind('.',0,m.start()),text.rfind('。',0,m.start()),text.rfind('\n',0,m.start()))+1
    # Decimal dots must not be sentence delimiters.
    stops=[x.start() for x in re.finditer(r'(?<!\d)\.(?!\d)|。|\n',text[m.end():])]
    b=m.end()+stops[0]+1 if stops else len(text)
    sentence=text[a:b]
    if re.search(r'\b(?:below|above|low|high|severe|eligible|ineligible|start|avoid|recommend|less|greater|not|under|over|months)\b|低于|高于|低下|升高|不符合|符合|建议|重度|小于|大于|未满',sentence,re.I):return None
    # Every mention of this field must be in this one measurement, excluding ages.
    if spec['field']!='age' and len(list(re.finditer(alias,text,re.I)))!=1:return None
    return {'start':start,'end':end,'value':float(m['value']),'evidence_start':m.start(),'evidence_end':m.end(),'evidence':m[0]}

def source_data():
    cfg=json.loads((ROOT/'configs/datasets.json').read_text())['MEGA-CDP'];d=Path(cfg['path'])
    load=lambda name:[json.loads(s) for s in (d/'train_val'/(name+'.jsonl')).read_text().splitlines()]
    return cfg,load('decision_trees'),load('case_paths'),load('vignette_cases')

def inspect():
    cfg,trees,paths,cases=source_data();historical=set(json.loads((ROOT/'configs/acre_pilot.json').read_text())['historical_guidelines'])
    pathmap={r['guideline_id']:r for r in paths};casemap={(r['guideline_id'],r['path_id']):r for r in cases};accepted=[];excluded=[];counts=collections.Counter()
    for ti,rec in enumerate(trees):
        gid=rec['guideline_id']
        if gid in historical:continue
        try: tree=json.loads(rec['text'])['decision_tree']
        except (ValueError,KeyError):continue
        def walk(n,ancestors):
            if not isinstance(n,dict):return
            if n.get('yes',{}).get('leaf') and n.get('no',{}).get('leaf'):
                counts['sibling_candidates']+=1;spec=numeric_spec(n.get('description',''))
                if spec:
                    counts['primitive_numeric_siblings']+=1
                    for pid,path in enumerate(pathmap[gid]['paths'],1):
                        if path[-1]['index'] not in (n['yes']['index'],n['no']['index']):continue
                        source=casemap.get((gid,pid))
                        if not source:continue
                        observation=observations(source['text'],spec)
                        if not observation:
                            excluded.append({'guideline_id':gid,'path_id':pid,'node_id':n['index'],'reason':'measurement_not_unique_or_qualitative_contradiction_or_unit_mismatch'});continue
                        state='yes' if OPS[spec['operator']](observation['value'],spec['threshold']) else 'no'
                        if n[state]['index']!=path[-1]['index']:
                            excluded.append({'guideline_id':gid,'path_id':pid,'node_id':n['index'],'reason':'original_numeric_evidence_disagrees_with_reference_path'});continue
                        step=max(spec['threshold']*.05,.1)
                        new_value=spec['threshold']-step if state=='yes' and spec['operator'] in ('>','>=','≥') or state=='no' and spec['operator'] in ('<','<=','≤') else spec['threshold']+step
                        if spec['field']=='age':new_value=max(0,round(new_value))
                        new_state='yes' if OPS[spec['operator']](new_value,spec['threshold']) else 'no'
                        if new_state==state:continue
                        # If ancestors mention same physiological variable, each must be a
                        # primitive numeric condition and preserve truth after the edit.
                        ancestor_checks=[];valid=True
                        for parent,outcome in ancestors:
                            if re.search(FIELDS[spec['field']][0],parent['description'],re.I):
                                ps=numeric_spec(parent['description'])
                                if not ps or ps['field']!=spec['field'] or ps['unit'].lower()!=spec['unit'].lower():valid=False;break
                                old=OPS[ps['operator']](observation['value'],ps['threshold']);new=OPS[ps['operator']](new_value,ps['threshold'])
                                ancestor_checks.append({'node_id':parent['index'],'old':old,'new':new})
                                if old!=new or old!=(outcome=='yes'):valid=False;break
                        if not valid:
                            excluded.append({'guideline_id':gid,'path_id':pid,'node_id':n['index'],'reason':'ancestor_condition_cannot_be_verified_unchanged'});continue
                        counts['numeric_edit_candidates']+=1
                        accepted.append({'candidate_id':f'{gid}:{pid}:{n["index"]}','guideline_id':gid,'path_id':pid,'node':n,'spec':spec,'observation':observation,'original_state':state,'new_state':new_state,'new_value':new_value,'ancestor_checks':ancestor_checks,'patient':source['text'],'source_tree_line':ti+1,'filename':rec['filename'],'original_path':path})
            for outcome in ('yes','no'):walk(n.get(outcome),ancestors+[(n,outcome)] if 'description' in n else ancestors)
        walk(tree,[])
    return cfg,counts,accepted,excluded

def screen_original_pairs():
    cfg,trees,path_records,case_records=source_data()
    historical=set(json.loads((ROOT/'configs/acre_pilot.json').read_text())['historical_guidelines'])
    cases={(r['guideline_id'],r['path_id']):r for r in case_records};counts=collections.Counter();candidates=[]
    for rec in path_records:
        gid=rec['guideline_id']
        if gid in historical:continue
        groups=collections.defaultdict(list)
        for pid,path in enumerate(rec['paths'],1):
            if len(path)>1:groups[tuple((x['index'],x.get('decision_outcome')) for x in path[:-2])+(path[-2]['index'],)].append((pid,path))
        for group in groups.values():
            if len(group)!=2 or group[0][1][-2]['decision_outcome']==group[1][1][-2]['decision_outcome']:continue
            counts['sibling_original_pairs']+=1
            texts=[cases.get((gid,pid),{}).get('text','') for pid,path in group]
            if not all(texts):continue
            sentences=lambda t:[x.strip() for x in re.split(r'(?<!\d)[。！？.!?](?!\d)|\n+',t) if x.strip()]
            a,b=map(sentences,texts)
            if len(a)!=len(b):continue
            diff=[(x,y) for x,y in zip(a,b) if x!=y]
            if len(diff)!=1:continue
            counts['single_sentence_change']+=1
            candidates.append({'guideline_id':gid,'source_path_ids':[v[0] for v in group],'condition':group[0][1][-2]['condition'],'changed_sentences':diff[0],'original_patients':texts,'actions':[v[1][-1]['conclusion'] for v in group]})
    return {'counts':counts,'candidates':candidates,'protocol':'Complete original vignette sentences identical except exactly one sentence; lexical screening only, manual validity still required.'}

def finalize_blocked_audit(audit,out):
    original_file=out/'original_pair_screening.json'
    original=json.loads(original_file.read_text()) if original_file.exists() else {'candidates':[]}
    raw=original.get('candidates',[]);original_valid=[c for c in raw if c.get('review',{}).get('valid') is True]
    numeric_valid=[c for c in audit['candidate_records'] if audit['reviewed_candidates'].get(c['candidate_id'],{}).get('incompatible_actions') and audit['reviewed_candidates'][c['candidate_id']].get('clinical_consistency')]
    total=len(original_valid)+len(numeric_valid)
    audit.update({'numeric_reviewed_candidates':len(audit['candidate_records']),'original_reviewed_candidates':len(raw),'candidate_pair_upper_bound_before_review':len(audit['candidate_records'])+len(raw),'valid_candidate_pairs_total':total,'valid_numeric_pairs':len(numeric_valid),'valid_original_pairs':len(original_valid),'excluded_reviewed_candidates':len(audit['candidate_records'])+len(raw)-total,'incomplete_candidates':0,'incomplete_within_excluded_candidates':sum('incomplete' in c.get('review',{}).get('reason','') or 'ambiguous' in c.get('review',{}).get('reason','') or 'unassigned' in c.get('review',{}).get('reason','') or 'not resolved' in c.get('review',{}).get('reason','') for c in raw),'preliminary_numeric_exclusions':len(audit['excluded']),'development_guidelines':[],'validation_guidelines':[],'guideline_overlap':[],'actual_inference_guidelines':[5,9,14,18,20,22],'historical_calibration_guidelines':[3],'guideline_isolation':'No development/validation inference launched; candidates exclude historical guides. This is not an achieved validation split.','reason':f'Two conservative screening protocols yielded {len(raw)+len(audit["candidate_records"])} candidates; only {total} passed source-text and action-incompatibility review, below 40. This is a limitation of the implemented screening protocols, not a proof of corpus-wide insufficiency.','original_pair_review_file':str(original_file.relative_to(ROOT)),'review_scope':'Programmatic tree/numeric checks plus source-text consistency review; no clinical expert certification. No candidate used for evaluation.'})
    audit['valid_pair_ids']=['numeric_'+c['candidate_id'] for c in numeric_valid]+['original_'+str(c['guideline_id'])+'_'+str(c['source_path_ids'][0])+'_'+str(c['source_path_ids'][1]) for c in original_valid]
    metadata=[]
    for c in numeric_valid:
        patient=c['patient'];o=c['observation'];new=patient[:o['start']]+format(c['new_value'],'.8g')+patient[o['end']:];review=audit['reviewed_candidates'][c['candidate_id']];rep=review.get('replacement')
        if rep:
            if new.count(rep['old'])!=1:raise ValueError('ambiguous_review_replacement')
            new=new.replace(rep['old'],rep['new'])
        metadata.append({'pair_id':'numeric_'+c['candidate_id'],'guideline_id':c['guideline_id'],'source_case_id':f'{c["guideline_id"]}:{c["path_id"]}','source_dataset_version':audit['source']['revision'],'condition':c['node']['description'],'original_patient':patient,'counterfactual_patient':new,'original_state':c['original_state'],'new_state':c['new_state'],'reference_actions':{v:c['node'][v]['conclusion'] for v in ['yes','no']},'source_node_id':c['node']['index'],'numeric_change':{'from':o['value'],'to':c['new_value']},'ancestor_checks':c['ancestor_checks'],'validation':review,'not_in_experiment':True,'split':'unassigned_candidate'})
    for c in original_valid:
        metadata.append({'pair_id':'original_'+str(c['guideline_id'])+'_'+str(c['source_path_ids'][0])+'_'+str(c['source_path_ids'][1]),'guideline_id':c['guideline_id'],'source_case_ids':[f'{c["guideline_id"]}:{p}' for p in c['source_path_ids']],'source_dataset_version':audit['source']['revision'],'condition':c['condition'],'original_patient':c['original_patients'][0],'counterfactual_patient':c['original_patients'][1],'reference_actions':c['actions'],'changed_evidence':c['changed_sentences'],'validation':c['review'],'not_in_experiment':True,'split':'unassigned_candidate'})
    (out/'screened_pairs.json').write_text(json.dumps({'pairs':metadata,'not_an_evaluation_set':True},ensure_ascii=False,indent=2)+'\n')
    return audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--inspect',action='store_true');p.add_argument('--review',type=Path);a=p.parse_args()
    cfg,counts,candidates,excluded=inspect();out=ROOT/'results/acre_pilot';out.mkdir(parents=True,exist_ok=True)
    if a.inspect:
        (out/'original_pair_candidates.json').write_text(json.dumps(screen_original_pairs(),ensure_ascii=False,indent=2)+'\n')
        (out/'numeric_candidates.json').write_text(json.dumps({'source':cfg,'counts':counts,'candidates':candidates,'excluded':excluded},ensure_ascii=False,indent=2)+'\n');print(json.dumps(counts));return
    if not a.review:raise ValueError('explicit incompatible-action and patient-validity review required')
    reviews=json.loads(a.review.read_text())
    historical=set(json.loads((ROOT/'configs/acre_pilot.json').read_text())['historical_guidelines'])
    # Generation intentionally fails closed until sufficient reviewed candidates exist.
    approved=[c for c in candidates if reviews.get(c['candidate_id'],{}).get('incompatible_actions') is True and reviews[c['candidate_id']].get('clinical_consistency') is True]
    required=json.loads((ROOT/'configs/acre_pilot.json').read_text())['development_pairs_target']
    audit={'source':cfg,'historical_guidelines':[3,5,9,14,18,20,22],'counts':counts,'numeric_candidates':len(candidates),'valid_reviewed_pairs':len(approved),'development_target':required,'excluded':excluded,'reviewed_candidates':reviews,'candidate_records':candidates,'candidate_guideline_ids':sorted({c['guideline_id'] for c in candidates}),'reference_usage':{'construction':True,'offline_evaluation':True,'annotation':False,'normal_inference':False,'normal_masks':False},'split_by_guideline':True}
    if len(approved)<required:
        audit.update({'status':'blocked_insufficient_verified_pairs','development_cases':0,'validation_cases':0,'validation_guidelines':[],'development_guidelines':[],'reason':'Fewer than 40 reviewed clinically consistent pairs with demonstrably incompatible actions. Numeric threshold changes alone cannot certify clinical validity.'})
        audit=finalize_blocked_audit(audit,out)
        (out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':audit['status'],'reviewed':audit['valid_candidate_pairs_total'],'candidates':audit['candidate_pair_upper_bound_before_review']}));return
    # Select once in source order, maximum one source direction per split node.
    # Both members and every node from one guideline stay in the same split.
    unique=[];seen=set()
    for c in approved:
        nodekey=(c['guideline_id'],c['node']['index'])
        if nodekey not in seen:unique.append(c);seen.add(nodekey)
    split_lists={'development':[],'validation':[]};dev_gids=set()
    for c in unique:
        if len(split_lists['development'])<required:
            split_lists['development'].append(c);dev_gids.add(c['guideline_id'])
    for c in unique:
        if c['guideline_id'] not in dev_gids:split_lists['validation'].append(c)
    split_lists['validation']=split_lists['validation'][:70]
    if len(split_lists['development'])<required:
        audit.update(status='blocked_insufficient_unique_pairs',reason='Fewer than 40 distinct verified split nodes',development_cases=0,validation_cases=0)
        (out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');return
    _,trees,paths,cases=source_data();tree_by_gid={r['guideline_id']:r for r in trees};path_by_gid={r['guideline_id']:r['paths'] for r in paths}
    def rules_for(gid):
        rules=[]
        for path in path_by_gid[gid]:
            conditions=[{'condition_id':f'mega_{gid}_node_{n["index"]}_{n["decision_outcome"]}','text':n['condition'],'negated':n['decision_outcome']=='no'} for n in path[:-1]]
            text='Apply this recommendation only when ALL requirements hold: '+' AND '.join('The answer must be '+('NO' if c['negated'] else 'YES')+': '+c['text'] for c in conditions)+'. Recommendation: '+path[-1]['conclusion']
            rules.append({'rule_id':f'mega_{gid}_leaf_{path[-1]["index"]}','text':text,'action':path[-1]['conclusion'],'conditions':conditions,'source_node_index':path[-1]['index']})
        return rules
    audit['pairs']=[]
    for split,selected in split_lists.items():
        samples=[];references={};oracles={}
        for c in selected:
            gid=c['guideline_id'];pair_id=f'acre_{gid}_{c["node"]["index"]}';rules=rules_for(gid)
            observation=c['observation'];patient=c['patient'];value_text=format(c['new_value'],'.8g')
            changed=patient[:observation['start']]+value_text+patient[observation['end']:]
            # A review may correct a qualifier of the same condition only. This
            # must be an exact audited replacement, not additional state changes.
            review=reviews[c['candidate_id']]
            if review.get('replacement'):
                replacement=review['replacement'];a,b=locate(changed,replacement['old']);changed=changed[:a]+replacement['new']+changed[b:]
            original_leaf=c['node'][c['original_state']]['index'];new_leaf=c['node'][c['new_state']]['index']
            pairmeta={k:c[k] for k in ('candidate_id','guideline_id','path_id','spec','original_state','new_state','new_value','ancestor_checks','filename','source_tree_line')}
            pairmeta.update(pair_id=pair_id,split=split,review=review,original_patient=patient,new_patient=changed,source_reference_path=c['original_path'],changed_condition=c['node']['description'],numeric_flip_verified=True)
            for member,record,leaf,competitor in [('original',patient,original_leaf,new_leaf),('counterfactual',changed,new_leaf,original_leaf)]:
                cid=pair_id+'_'+member;gold=f'mega_{gid}_leaf_{leaf}';comp=f'mega_{gid}_leaf_{competitor}'
                source_obs=observations(record,c['spec'])
                # Review corrections may contain explicit polarity; exact numeric
                # quote is still located using the original unique variable match.
                if source_obs is None:
                    alias,unit=FIELDS[c['spec']['field']]
                    pattern=r'(?:'+alias+r')(?:\s*(?:is|was|of|at|为|是|:|：|=))*\s*\d+(?:\.\d+)?\s*(?:'+unit+r')'
                    matches=list(re.finditer(pattern,record,re.I))
                    if len(matches)!=1:raise ValueError('reviewed_evidence_not_unique')
                    evidence=matches[0][0]
                else:evidence=source_obs['evidence']
                samples.append({'case_id':cid,'pair_id':pair_id,'guideline_id':gid,'guideline':{'rules':rules},'patient':record,'question':'For this patient, what is the main management decision recommended by the provided guideline?','input_source':{'dataset':'MEGA-CDP','split':'train_val','guideline_id':gid,'source_path_id':c['path_id'],'tree_line':c['source_tree_line'],'synthetic_counterfactual':member=='counterfactual'}})
                # All leaves in this complete tree are exclusive signed paths.
                states={r['rule_id']:('applicable' if r['rule_id']==gold else 'inapplicable') for r in rules}
                ref={'case_id':cid,'pair_id':pair_id,'guideline_id':gid,'gold_rule_id':gold,'gold_action':next(r['action'] for r in rules if r['rule_id']==gold),'competitor_action':next(r['action'] for r in rules if r['rule_id']==comp),'target_rule_id':f'mega_{gid}_leaf_{c["node"]["yes"]["index"]}','target_applicability':'applicable' if leaf==c['node']['yes']['index'] else 'inapplicable','rule_states':states,'evidence':evidence,'information_sufficient':True,'incompatible_actions_verified':True,'source_audit_pair':pair_id}
                references[cid]=ref
                oracle_rows=[]
                for r in rules:
                    is_gold=r['rule_id']==gold;is_comp=r['rule_id']==comp
                    oracle_rows.append({'rule_id':r['rule_id'],'relevant':is_gold or is_comp,'applicability':states[r['rule_id']],'action_spans':[r['action']],'supporting_evidence':[evidence] if is_gold else [],'refuting_evidence':[evidence] if is_comp else [],'exception_spans':[]})
                oracles[cid]={'verified':True,'rules':oracle_rows,'scope':'Verified changed sibling rules only; other branch evidence not assumed'}
            audit['pairs'].append(pairmeta)
        (out/(split+'_manifest.json')).write_text(json.dumps({'samples':samples},ensure_ascii=False,indent=2)+'\n')
        (out/(split+'_references.json')).write_text(json.dumps(references,ensure_ascii=False,indent=2)+'\n')
        (out/(split+'_oracle.json')).write_text(json.dumps(oracles,ensure_ascii=False,indent=2)+'\n')
    dev_gids=sorted(dev_gids);val_gids=sorted({c['guideline_id'] for c in split_lists['validation']})
    if set(dev_gids)&set(val_gids) or set(val_gids)&historical:raise AssertionError('guideline_split_leakage')
    audit.update(status='prepared',development_cases=len(split_lists['development'])*2,validation_cases=len(split_lists['validation'])*2,development_guidelines=dev_gids,validation_guidelines=val_gids,guideline_overlap=[],validation_not_run=True)
    (out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
