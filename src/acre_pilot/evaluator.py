"""Offline metrics only: labels cannot affect normal prompts or target selection."""
import collections,math,random
from .annotator import STATES

def fraction(n,d):return {'numerator':n,'denominator':d,'value':n/d if d else None}
def evaluate_cases(cases,references):
    bycase={c['case_id']:c for c in cases};metrics={};details=[]
    methods=sorted({m for c in cases for m in c['variants']})
    for method in methods:
        correct=app_n=app_correct=oar_n=oar=retained_n=retained=enhanced=0;pairs=collections.defaultdict(list);margins=collections.defaultdict(collections.Counter);covered=0
        for cid,c in bycase.items():
            ref=references[cid];options=c['options'];v=c['variants'][method]
            gold=next(o['label'] for o in options if o['action']==ref['gold_action']);ok=v['prediction']==gold
            correct+=ok;covered+=v['prediction'] is not None
            base=c['variants'].get('B0');baseok=base and base['prediction']==gold
            retained_n+=bool(baseok);retained+=bool(baseok and ok)
            if ref.get('target_applicability')=='applicable':app_n+=1;app_correct+=ok
            if ref.get('target_applicability')=='inapplicable' and ref.get('incompatible_actions_verified'):
                target=next(r for r in c['annotations'] if r['rule_id']==ref['target_rule_id'])
                target_action=ref.get('target_action') or ref['competitor_action']
                oar_n+=1;oar+=v.get('action')==target_action
            if ref.get('pair_id'):pairs[ref['pair_id']].append(ok)
            enhanced+=v.get('attention',{}).get('enabled',False)
            row={'case_id':cid,'guideline_id':c['guideline_id'],'pair_id':ref.get('pair_id'),'method':method,'prediction':v['prediction'],'gold_label':gold,'correct':ok,'baseline_correct':bool(baseok),'margin':None,'delta_margin':None}
            if base and 'option_logits' in v:
                gi=next(i for i,o in enumerate(options) if o['label']==gold)
                if ref.get('competitor_action'):
                    ci=next(i for i,o in enumerate(options) if o['action']==ref['competitor_action'])
                else:ci=max((i for i in range(len(options)) if i!=gi),key=lambda i:base['option_logits'][i])
                margin=v['option_logits'][gi]-v['option_logits'][ci];original=base['option_logits'][gi]-base['option_logits'][ci];delta=margin-original
                row.update(margin=margin,delta_margin=delta,fixed_competitor_label=options[ci]['label'],margin_competitor_protocol='verified sibling action' if ref.get('competitor_action') else 'B0 strongest incorrect option; diagnostic only')
                sign='positive' if delta>1e-8 else 'negative' if delta< -1e-8 else 'zero'
                margins[ref.get('target_applicability','historical_unstratified')][sign]+=1
            details.append(row)
        complete=[v for v in pairs.values() if len(v)==2]
        metrics[method]={'accuracy':fraction(correct,len(cases)),'applicable_accuracy':fraction(app_correct,app_n),'oar':fraction(oar,oar_n),'pair_accuracy':fraction(sum(all(v) for v in complete),len(complete)),'retention':fraction(retained,retained_n),'decision_coverage':fraction(covered,len(cases)),'enhancement_coverage':fraction(enhanced,len(cases)),'margin_signs':{k:dict(v) for k,v in margins.items()}}
    # Gold applicability is only reported when the source complete signed tree
    # or a independently verified reference explicitly supports rule-level labels.
    confusion={s:{t:0 for t in STATES} for s in STATES};located=spans=unknown=rules_n=falseapp=neg_n=recall_n=recall=0;parse_failure=0
    for c in cases:
        ref=references[c['case_id']];parse_failure+=bool(c['annotation_errors'])
        for r in c['annotations']:
            rules_n+=1;unknown+=r['applicability']=='unknown'
            for f in r['mapped']:
                located+=len(r['mapped'][f]);spans+=len(r.get(f,[]))
            if r['rule_id'] in ref.get('rule_states',{}):
                gold=ref['rule_states'][r['rule_id']];confusion[gold][r['applicability']]+=1
                neg_n+=gold=='inapplicable';falseapp+=gold=='inapplicable' and r['applicability']=='applicable'
            if ref.get('evidence') and r['rule_id']==ref.get('target_rule_id'):
                recall_n+=1
                field='supporting_evidence' if ref['target_applicability']=='applicable' else 'refuting_evidence'
                recall+=any(ref['evidence'] in s['text'] for s in r['mapped'][field])
    total=sum(sum(v.values()) for v in confusion.values());perclass={}
    for s in STATES:
        tp=confusion[s][s];pred=sum(confusion[t][s] for t in STATES);gold=sum(confusion[s].values())
        perclass[s]={'precision':fraction(tp,pred),'recall':fraction(tp,gold)}
    ann={'confusion_matrix':confusion,'applicability_accuracy':fraction(sum(confusion[s][s] for s in STATES),total),'per_class':perclass,'inapplicable_as_applicable':fraction(falseapp,neg_n),'unknown_fraction':fraction(unknown,rules_n),'exact_span_location':fraction(located,spans),'decisive_evidence_recall':fraction(recall,recall_n),'annotation_failure_cases':parse_failure,'all_cases_included':True,'unknown_gold_class_available':sum(confusion['unknown'].values())>0}
    return metrics,ann,details

def paired_comparisons(details,pairs,seed=20261015,n_boot=2000):
    # Cluster bootstrap samples guidelines (thereby retaining intact pairs).
    # Paired exact binomial McNemar is descriptive; cluster CI handles dependence.
    lookup={(r['case_id'],r['method']):r for r in details};ids=sorted({r['case_id'] for r in details});results=[]
    for a,b in pairs:
        rows=[(lookup[(cid,a)],lookup[(cid,b)]) for cid in ids]
        better=sum(x['correct'] and not y['correct'] for x,y in rows);worse=sum(y['correct'] and not x['correct'] for x,y in rows);n=better+worse
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1.
        clusters=collections.defaultdict(list)
        for x,y in rows:clusters[x['guideline_id']].append(int(x['correct'])-int(y['correct']))
        keys=list(clusters);rng=random.Random(seed);draws=[]
        if keys:
            for _ in range(n_boot):
                sampled=[v for k in rng.choices(keys,k=len(keys)) for v in clusters[k]];draws.append(sum(sampled)/len(sampled))
            draws.sort();ci=[draws[int(.025*n_boot)],draws[int(.975*n_boot)]]
        else:ci=None
        results.append({'method_a':a,'method_b':b,'a_only_correct':better,'b_only_correct':worse,'n':len(rows),'absolute_accuracy_difference':(better-worse)/len(rows) if rows else None,'exact_mcnemar_p':p,'guideline_cluster_bootstrap_95ci':ci,'clusters':len(keys),'inference_scope':'descriptive; insufficient clusters preclude robust claims' if len(keys)<10 else 'paired exploratory'})
    order=sorted(range(len(results)),key=lambda i:results[i]['exact_mcnemar_p']);previous=0
    for rank,i in enumerate(order):
        previous=max(previous,min(1,results[i]['exact_mcnemar_p']*(len(order)-rank)));results[i]['holm_p']=previous
    return results
