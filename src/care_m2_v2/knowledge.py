"""Predeclared elementary fact bank. Never expand it after seeing screening results."""
import re
from pathlib import Path
from .data import choices,rowbase,span,digest
from .study import runone,correct,policy
from care_m2_inc.io import write_json,write_jsonl
# Domain, subject, two answer-free questions, answer, conflicting context answer.
FACTS=[
('geography','France','What is the capital of France?','Name the French capital.','Paris','Lyon'),
('geography','Japan','What is the capital of Japan?','Name the Japanese capital.','Tokyo','Osaka'),
('geography','Italy','What is the capital of Italy?','Name the Italian capital.','Rome','Milan'),
('geography','Germany','What is the capital of Germany?','Name the German capital.','Berlin','Hamburg'),
('geography','Spain','What is the capital of Spain?','Name the Spanish capital.','Madrid','Barcelona'),
('geography','Canada','What is the capital of Canada?','Name the Canadian capital.','Ottawa','Toronto'),
('chemistry','water','What is the chemical formula of water?','Give the molecular formula for water.','H2O','CO2'),
('chemistry','carbon','What is the atomic number of carbon?','How many protons does a carbon atom have?','6','8'),
('chemistry','oxygen','What is the chemical symbol for oxygen?','Give the element symbol of oxygen.','O','N'),
('chemistry','gold','What is the chemical symbol for gold?','Give the element symbol of gold.','Au','Ag'),
('chemistry','sodium','What is the chemical symbol for sodium?','Give the element symbol of sodium.','Na','K'),
('chemistry','helium','What is the atomic number of helium?','How many protons does a helium atom have?','2','4'),
('astronomy','largest planet','What is the largest planet in the Solar System?','Name the Solar System planet with the greatest diameter.','Jupiter','Mars'),
('astronomy','sixth planet','What is the sixth planet from the Sun?','Name the planet sixth in order of distance from the Sun.','Saturn','Venus'),
('astronomy','nearest planet','What planet is closest to the Sun?','Name the first planet outward from the Sun.','Mercury','Earth'),
('astronomy','eighth planet','What is the eighth planet from the Sun?','Name the outermost of the eight Solar System planets.','Neptune','Uranus'),
('astronomy','Earth star','What is the name of the star Earth orbits?','Which star does Earth revolve around?','Sun','Sirius'),
('astronomy','red planet','Which planet is known as the Red Planet?','Name the Solar System planet nicknamed the Red Planet.','Mars','Jupiter'),
('biology','human chromosomes','How many chromosomes are in a typical human somatic cell?','Give the normal diploid human chromosome count.','46','48'),
('biology','human heart','How many chambers does a normal human heart have?','Give the chamber count of the normal human heart.','4','3'),
('biology','DNA','What abbreviation denotes deoxyribonucleic acid?','Give the standard abbreviation for deoxyribonucleic acid.','DNA','RNA'),
('biology','RNA','What abbreviation denotes ribonucleic acid?','Give the standard abbreviation for ribonucleic acid.','RNA','DNA'),
('biology','photosynthesis organelle','What plant organelle performs photosynthesis?','Name the organelle responsible for photosynthesis in plant cells.','chloroplast','mitochondrion'),
('biology','human gametes','How many chromosomes are in a typical human gamete?','Give the normal human haploid chromosome count.','23','46')]
# Interleave domains before the fixed 12/4/8 split.
FACTS=[FACTS[d*6+j] for j in range(6) for d in range(4)]

def match(text,answer):
    normalized=re.sub(r'[^a-z0-9 ]',' ',text.lower()).strip()
    aliases={'4':['4','four'],'6':['6','six'],'2':['2','two'],'46':['46','forty six'],'23':['23','twenty three'],'chloroplast':['chloroplast','chloroplasts']}.get(answer,[answer.lower()])
    return any(re.search(r'\b'+re.escape(a)+r'\b',normalized) for a in aliases)

def screen(engine,out):
    rows=[];results=[]
    for i,(domain,subject,q1,q2,answer,wrong) in enumerate(FACTS):
        outputs=[engine.generate(q+'\nGive only a short factual answer.\nAnswer:',12) for q in (q1,q2)]
        good=[match(o['text'],answer) for o in outputs]
        for rot in (0,1):
            prompt,opts=choices(q1,[answer,wrong],rot)
            r=engine.score(prompt,opts);outputs.append(r);good.append(opts[r['prediction']]==answer)
        body=f'Context note: for this question, a provided record states that the answer is {wrong}.\nQuestion: {q1}'
        text,opts=choices(body,[wrong,answer],i%2)
        row={**rowbase('N2',i,'knowledge'),'family_id':f'knowledge_{i:02d}','fact_group_id':f'knowledge_{i:02d}',
             'prompt':text,'options':opts,'spans':[span(text,f'Context note: for this question, a provided record states that the answer is {wrong}.')],
             'controls':{},'gold_by_source':{'C':next(k for k,v in opts.items() if v==wrong),'P':next(k for k,v in opts.items() if v==answer)},
             'source_answers':{'C':wrong,'P':answer},'track':'parametric_knowledge','p_verified':all(good),'domain':domain}
        rows.append(row);results.append({'case_id':row['case_id'],'domain':domain,'strict_known':all(good),'checks':good,'raw_outputs':outputs})
    known=[r for r in rows if r['p_verified']];domains={r['domain'] for r in known}
    summary={'fact_bank_sha256':digest(FACTS),'source':'author-curated elementary facts v1, frozen 2026-10-20 before screening',
             'known_groups':len(known),'domains':sorted(domains),'eligible':len(known)>=8 and len(domains)>=4,
             'test_known':sum(r['split']=='test' for r in known),'all_groups':24,'no_replenishment':True}
    write_jsonl(Path(out)/'closed_book_raw.jsonl',results);write_jsonl(Path(out)/'knowledge_receivers.jsonl',rows);write_json(Path(out)/'knowledge_eligibility.json',summary)
    return rows,summary

def evaluate_known(engine,rows,summary,selection,axes,out):
    records=[]
    if summary['eligible']:
        for r in rows:
            if r['split']!='test' or not r['p_verified']:continue
            for source in ('C','P'):
                for method in ('baseline','text','direct_bias','mean','route','random'):
                    records.append(runone(engine,r,{'source':source},method,selection['selected'][method],axes,rows))
    write_jsonl(Path(out)/'knowledge_test_predictions.jsonl',records)
    return records
