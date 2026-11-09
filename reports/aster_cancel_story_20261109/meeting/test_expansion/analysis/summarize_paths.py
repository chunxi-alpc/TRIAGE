import argparse,csv,json
from pathlib import Path
import numpy as np

def load_rows(paths):
    rows={}
    for filename in paths:
        with Path(filename).open() as handle:
            for line in handle:
                if not line.strip():continue
                row=json.loads(line)
                rows[(row['case_id'],row['method'])]=row
    return list(rows.values())

def valid_record(row):
    return not row.get('error') and row.get('status','ok') not in ['error','failed','unavailable']

def summarize(rows,cases,iterations=2000):
    meta={r['case_id']:r for r in cases}
    methods=sorted({r['method'] for r in rows})
    table={(r['case_id'],r['method']):r for r in rows}
    groups={'all':set(meta)}
    for c in cases:
        for field in ['evaluation_group','usage_group','history_group']:
            value=c.get(field)
            if value is not None:groups.setdefault(str(value),set()).add(c['case_id'])
        if 'in_current_fit_dev' in c:
            groups.setdefault('current_fit_dev_sources' if c['in_current_fit_dev'] else 'outside_current_fit_dev_sources',set()).add(c['case_id'])
        if 'in_old_test' in c:
            groups.setdefault('old_test_cases' if c['in_old_test'] else 'additional_cases',set()).add(c['case_id'])
    output={'bootstrap':f'source-level paired resampling; {iterations} samples; seed 11','groups':{}}
    for group,ids in groups.items():
        sources=sorted({meta[c]['source_id'] for c in ids});si={s:i for i,s in enumerate(sources)}
        by_method={}
        for method in methods:
            present=[c for c in ids if (c,method) in table]
            valid=[c for c in present if valid_record(table[c,method])]
            sums=np.zeros((len(sources),3),dtype=np.float64)
            correct=leaf=0
            for cid in valid:
                r=table[cid,method];yes=bool(r.get('correct',r.get('path_correct',False)))
                leaf_yes=bool(r.get('leaf_correct',r.get('leaf',r.get('predicted_leaf'))==meta[cid].get('gold_leaf')))
                j=si[meta[cid]['source_id']];sums[j]+=[yes,leaf_yes,1];correct+=yes;leaf+=leaf_yes
            if len(valid):
                rng=np.random.default_rng(11);boots=[]
                for _ in range(iterations):
                    sampled=sums[rng.integers(0,len(sources),size=len(sources))].sum(axis=0)
                    if sampled[2]:boots.append(sampled[:2]/sampled[2])
                ci=np.quantile(boots,[.025,.975],axis=0).T.tolist() if boots else None
            else:ci=None
            item={'expected':len(ids),'available':len(present),'evaluated':len(valid),'failed':len(present)-len(valid),'missing':len(ids)-len(present),'path_correct':correct,'leaf_correct':leaf,'path_accuracy':correct/len(valid) if valid else None,'leaf_accuracy':leaf/len(valid) if valid else None,'source_bootstrap_95ci':ci}
            if method!='raw':
                common=[c for c in valid if (c,'raw') in table and valid_record(table[c,'raw'])]
                changes=np.zeros((len(sources),2),dtype=np.float64);fixed=broken=0
                for cid in common:
                    ours=bool(table[cid,method]['correct']);base=bool(table[cid,'raw']['correct']);diff=int(ours)-int(base)
                    changes[si[meta[cid]['source_id']]] += [diff,1]
                    fixed+=diff==1;broken+=diff==-1
                rng=np.random.default_rng(11);boot=[]
                for _ in range(iterations):
                    v=changes[rng.integers(0,len(sources),size=len(sources))].sum(axis=0)
                    if v[1]:boot.append(v[0]/v[1])
                item['paired_vs_raw']={'n':len(common),'fixed':fixed,'broken':broken,'accuracy_difference':(fixed-broken)/len(common) if common else None,'source_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist() if boot else None}
            by_method[method]=item
        output['groups'][group]={'case_count':len(ids),'source_count':len(sources),'methods':by_method}
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',nargs='+',required=True);parser.add_argument('--cases',required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
    cases=[json.loads(s) for s in Path(a.cases).read_text().splitlines() if s.strip()]
    result=summarize(load_rows(a.results),cases)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    flat=[]
    for group,data in result['groups'].items():
        for method,m in data['methods'].items():
            flat.append({'group':group,'method':method,**{k:v for k,v in m.items() if isinstance(v,(int,float,type(None)))}})
    if flat:
        with (out/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({'groups':len(result['groups']),'methods':len(result['groups']['all']['methods']),'expected_cases':len(cases)}))
if __name__=='__main__':main()
