"""Prepare a pinned DriftMed adaptation. Run on the experiment server only."""
import json, hashlib, re, collections, datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parents[1]
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x): return re.sub(r"[【】〖〗]","",x).strip()
raw=rows(ROOT/'sources/advice_pairs.jsonl')
parent={r['id']:r['id'] for r in raw}
def find(i):
    while parent[i]!=i: i=parent[i]
    return i
def union(a,b): parent[max(find(a),find(b))]=min(find(a),find(b))
# Prespecified medical-topic groups: related statements and near paraphrases stay together.
groups=[ [2,25,26,85,109], [3,4], [5,59,70], list(range(6,15)),
 list(range(15,29))+list(range(52,59))+[75,76,77], [29,30],
 list(range(31,41))+[46,47,48], [41,78,79,80,81], [42,68], [43,44,45],
 [50,51], [60,62,63,65,66], [64,82], [69,112,113], [71,72,73,90],
 [74]+list(range(91,107))+[61], [83,84]+list(range(114,119)), [86,87],
 [88,89,108], list(range(119,125))+[111], [125,126,127],
 list(range(129,141)), [141,142,143,144], [145,181], [148,149,157],
 [150,151,154,155], [152,153], [158,159], [162,163],
 list(range(164,174))+[174], [177,184,185], [178,179,180], [186,187,188,189],
 list(range(190,195))]
for g in groups:
    for i in g[1:]: union(g[0],i)
v=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
x=v.fit_transform([clean(r['correct']) for r in raw]); sim=(x@x.T).toarray()
edges=[]
for a in range(len(raw)):
    for b in range(a):
        if sim[a,b]>=.70:
            union(raw[a]['id'],raw[b]['id']);edges.append([raw[a]['id'],raw[b]['id'],float(sim[a,b])])

# Open questions and exact accepted answer classes are fixed before any model output.
# Two differently worded questions, no candidates or retrieved recommendation in either.
specs=[
(1,"What is the minimum laboratory complexity level for FDA-approved point-of-care A1C diagnosis?","At what laboratory testing complexity must diagnostic point-of-care HbA1c testing be performed?",r"moderate",r"\blow\b"),
(7,"For an immunocompetent adult with diabetes, what is the usual PPSV23 interval after PCV15?","After PCV15, how long should an adult with diabetes and no immunocompromising condition normally wait for PPSV23?",r"1 year|one year|12 months|twelve months",r"6 months|six months"),
(8,"What minimum PCV15-to-PPSV23 interval can be considered in an immunocompromised adult?","In an immunocompromised patient receiving PCV15 then PPSV23, what is the shortest permissible interval in weeks?",r"\b8\b|eight",r"\b4\b|four"),
(9,"After PPSV23 alone, what minimum interval precedes PCV15 or PCV20 in an adult?","How long after a prior PPSV23-only vaccination should an adult wait before PCV15 or PCV20?",r"1 year|one year|12 months|twelve months",r"6 months|six months"),
(16,"With diabetes and a low FIB-4, elevated aminotransferases persisting beyond how many months warrant evaluation for other liver disease?","In a diabetic adult with low FIB-4, what duration of persistent aminotransferase elevation prompts investigating other liver causes?",r"\b6\b|six",r"\b3\b|three"),
(23,"What glucose-lowering therapy is preferred for type 2 diabetes with decompensated cirrhosis?","Which agent is preferred to manage hyperglycemia in an adult with decompensated cirrhosis and type 2 diabetes?",r"insulin",r"sglt"),
(29,"How often is A1C generally monitored in preclinical type 1 diabetes?","What is the approximate A1C monitoring interval in months for preclinical type 1 diabetes?",r"\b6\b|six",r"\b9\b|nine"),
(31,"How often should high-risk older adults with diabetes undergo bone mineral density monitoring?","What is the usual interval in years between DXA assessments in older adults with diabetes at high fracture risk?",r"2\s*[-–to]+\s*3|two to three|2 or 3",r"\b5\b|five"),
(35,"At what T-score threshold are people with diabetes considered at high fracture risk?","What low bone T-score cutoff defines high fracture risk specifically in diabetes?",r"[-−]2(?:\.0)?(?![\d.])",r"[-−]2\.5"),
(36,"Diabetes duration exceeding how many years is associated with increased fracture risk?","Beyond what duration of diabetes should increased fracture risk prompt bone health assessment?",r"\b10\b|ten",r"\b15\b|fifteen"),
(49,"At least how often should a person with diabetes have a dental examination?","What is the minimum routine dental examination frequency for people with diabetes?",r"annual|yearly|once.{0,8}year|every year|12 months",r"two years|2 years|24 months"),
(53,"What FIB-4 threshold prompts additional liver fibrosis assessment in type 2 diabetes or prediabetes?","At what FIB-4 cutoff should diabetic adults undergo elastography or an ELF test for liver fibrosis risk?",r"1\.3",r"1\.1"),
(61,"What daily sodium limit in milligrams is generally advised for people with diabetes?","In diabetes nutrition guidance, sodium intake should remain below how many milligrams per day?",r"2,?300|2\.3\s*g",r"2,?600|2\.6\s*g"),
(67,"How frequently in minutes should prolonged sitting be interrupted in diabetes?","For blood glucose benefit, people with diabetes should break up sitting at least every how many minutes?",r"\b30\b|thirty",r"\b60\b|sixty"),
(82,"Which blood ketone should be measured to assess ketoacidosis risk in patients taking SGLT inhibitors?","What ketone analyte is preferred for accurate blood ketone monitoring in an SGLT inhibitor user?",r"hydroxybutyrate|\bbhb\b",r"urine|dipstick"),
(93,"What daily protein intake in g/kg is advised for non-dialysis stage G3 or worse CKD?","In stage G3+ chronic kidney disease without dialysis, what is the recommended daily grams of protein per kg?",r"0\.8",r"0\.65"),
(94,"What protein intake in g/kg/day should be considered for a person on dialysis?","What daily protein range per kg helps address protein energy wasting during dialysis?",r"1(?:\.0)?\s*[-–to]+\s*1\.2",r"0\.8\s*[-–to]+\s*1(?:\.0)?"),
(95,"In type 1 diabetes with CKD, duration below how many years suggests nondiabetic kidney disease?","What short type 1 diabetes duration cutoff raises concern that CKD has another cause?",r"\b5\b|five",r"\b10\b|ten"),
(128,"For how long should diabetes screening continue after gestational diabetes?","What is the total duration of follow-up diabetes screening after a history of GDM?",r"lifelong|life.long|indefinite",r"9 years|nine years"),
(139,"How long should HBV-active antiretroviral therapy continue after transplant in HIV with active HBV?","In an HIV transplant recipient with active hepatitis B, what is the duration of anti-HBV treatment after transplantation?",r"indefinite|lifelong|life.long",r"24 months|two years|2 years"),
(152,"For how many days after completing 1HP should a suppressed patient continue twice-daily dolutegravir?","After a 1HP tuberculosis-prevention course, how long is the increased twice-daily DTG dose maintained?",r"\b14\b|fourteen|2 weeks|two weeks",r"\b12\b|twelve"),
(153,"After controlling HIV-associated TB meningitis, what minimum anti-TB treatment duration precedes ART?","With TB meningitis under control, how many weeks of anti-TB treatment are required at minimum before ART initiation?",r"2 weeks|two weeks|14 days|fourteen days",r"10 days|ten days"),
(154,"What are the two Heplisav-B dose times in weeks for an unvaccinated adult with HIV?","Give the recommended Heplisav-B schedule for an HIV patient who has never received hepatitis B vaccine.",r"0.{0,12}4|zero.{0,12}four|one month|1 month",r"32 days|thirty.two"),
(163,"Below what CD4 count is primary PCP prophylaxis indicated regardless of HIV RNA level?","What CD4 cutoff alone indicates Pneumocystis primary prophylaxis even with undetectable HIV RNA?",r"\b100\b|one hundred",r"\b110\b"),
(168,"What HIV RNA threshold defines sustained suppression for counseling people with HIV who desire breastfeeding?","For supported breastfeeding in a person on ART, sustained viral load should be below how many copies per mL?",r"\b50\b|fifty",r"\b60\b|sixty"),
(171,"When should additional infant HIV virologic testing occur after presumptive ARV therapy stops?","What is the recommended interval in weeks for additional infant HIV NAT after discontinuing presumptive antiretrovirals?",r"2\s*[-–to]+\s*6|two to six",r"4\s*[-–to]+\s*8|four to eight"),
(172,"At least how often is infant HIV virologic testing performed during breastfeeding?","During ongoing breastfeeding exposure to HIV, what is the maximum routine interval between infant virologic tests?",r"3 months|three months|quarter",r"3 weeks|three weeks"),
(173,"At what age in months can HIV antibody tests be used for standard diagnostic testing in children?","From what age are HIV antibody or antigen/antibody tests recommended for childhood HIV diagnosis?",r"\b18\b|eighteen",r"birth|all ages|0 months"),
(177,"What age cutoff in months separates HCV RNA testing from HCV antibody screening in children?","Below which pediatric age in months should hepatitis C RNA testing be used instead of antibody screening?",r"\b18\b|eighteen",r"\b16\b|sixteen"),
(184,"Which test is recommended at age 2 to 6 months for an infant perinatally exposed to HCV?","For a 3-month-old with perinatal hepatitis C exposure, what diagnostic test should be ordered?",r"rna|nucleic|\bnat\b|\bpcr\b",r"antibod|serolog"),
(189,"From what age in years is pangenotypic DAA treatment recommended for children with HCV?","What is the youngest age in years for pangenotypic direct-acting antiviral treatment of pediatric HCV?",r"\b3\b|three",r"\b4\b|four")]

# Patient-condition counterfactual diagnostics. These are purpose-written, not native dataset cases.
# Only one factual evidence snippet is shown. No old/new rule pair is put in the prompt.
pairs=[
dict(id='pneumococcal',sources=[7,8],field='immune_status',values=['immunocompetent','immunocompromised'],patient='An adult with diabetes received PCV15. Immune status: {value}. There is no cochlear implant or CSF leak.',question='What is the minimum permitted interval before PPSV23?',answers=['1 year','8 weeks'],scope='immune_status == immunocompromised',snippet='For immunocompromised adults, the minimum PCV15-to-PPSV23 interval can be 8 weeks.',scope_answer=1,patterns=[r'1 year|one year|12 months',r'8 weeks|eight weeks']),
dict(id='renal_protein',sources=[93,94],field='dialysis',values=['no','yes'],patient='An adult has diabetes and stage G5 chronic kidney disease. Receiving maintenance dialysis: {value}.',question='What daily protein intake in g/kg should be considered?',answers=['0.8 g/kg/day','1.0-1.2 g/kg/day'],scope='dialysis == yes',snippet='For adults receiving dialysis, consider protein intake of 1.0-1.2 g/kg/day.',scope_answer=1,patterns=[r'0\.8',r'1(?:\.0)?\s*[-–to]+\s*1\.2']),
dict(id='inpatient_glucose',sources=[121,122],field='icu',values=['no','yes'],patient='An adult with diabetes has hyperglycemia requiring inpatient therapy. Critically ill and in the ICU: {value}. There is no indication for a more stringent individualized target.',question='What is the usual blood glucose target range in mg/dL?',answers=['100-180 mg/dL','140-180 mg/dL'],scope='icu == yes',snippet='For most critically ill adults in the ICU, a glucose goal of 140-180 mg/dL is recommended.',scope_answer=1,patterns=[r'100\s*[-–to]+\s*180',r'140\s*[-–to]+\s*180']),
dict(id='hiv_diagnosis',sources=[169,173,174],field='age_months',values=[6,24],patient='A child has perinatal HIV exposure and requires standard diagnostic testing. Age: {value} months.',question='Which type of HIV diagnostic test is appropriate at this age?',answers=['HIV nucleic acid test','HIV antibody or antigen/antibody test'],scope='age_months >= 18',snippet='For children aged 18 months or older, HIV antibody or antigen/antibody tests are recommended for diagnostic testing.',scope_answer=1,patterns=[r'nucleic|\bnat\b|\bpcr\b|rna|dna',r'antibod|antigen'])]
forced={find(i) for p in pairs for i in p['sources']}
for p in pairs:
    for i in p['sources'][1:]: union(p['sources'][0],i)
forced={find(i) for i in forced}
def split(g):
    if g in forced: return 'test'
    u=int(hashlib.sha256(f'20261019:{g}'.encode()).hexdigest()[:8],16)%10
    return 'train' if u<6 else 'validation' if u<8 else 'test'
items=[dict(id=r['id'],correct=clean(r['correct']),wrong=clean(r['wrong']),disease=r['disease'],group=find(r['id']),split=split(find(r['id'])),source_row=r['id'],source_revision='5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e') for r in raw]
index={r['id']:r for r in items}
open_specs=[dict(id=s[0],questions=list(s[1:3]),correct_pattern=s[3],wrong_pattern=s[4],group=index[s[0]]['group'],split=index[s[0]]['split']) for s in specs]
for p in pairs: p.update(group=index[p['sources'][0]]['group'],split='test',authorship='experiment author; grounded in cited advice rows; not clinician-reviewed')
for name,data in [('items',items),('open_probes',open_specs),('patient_pairs',pairs)]:dump(ROOT/f'data/{name}.json',data)
src={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (ROOT/'sources').glob('*') if p.is_file()}
counts={f:len(rows(ROOT/'sources'/f)) for f in ['advice_pairs.jsonl','ConflictMed.jsonl','ConflictMed_v2.jsonl']}
dump(ROOT/'audit/data_audit.json',dict(source='https://huggingface.co/datasets/RDBH/DriftMed',revision=items[0]['source_revision'],files=src,counts=counts,split_counts=dict(collections.Counter(r['split'] for r in items)),group_counts=dict(collections.Counter(split(g) for g in {r['group'] for r in items})),open_counts=dict(collections.Counter(r['split'] for r in open_specs)),near_duplicate_edges=edges,forced_test_groups=sorted(forced),issues=['194 advice pairs versus 195 in paper and 4290/22 scenario groups; no row-to-scenario mapping assumed','Scenario versions/card and paper disagree on generation model; each preserved as published','wrong advice is manually fabricated pseudo-outdated content, not established historical guidance','No per-row guideline URL/year in advice file; row provenance is release provenance, not independent clinical validation','Research Use Only dataset card; raw files kept on server, not redistributed in Git','Topic groups plus lexical connected components reduce leakage but cannot prove full semantic disjointness','No native scenario train/test split supplied; native scenarios acquired and audited but not used for direction fitting or held-out evaluation'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
config=dict(seed=20261019,model='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B',gpu='0',dtype='bfloat16',batch_size=12,max_new_tokens=32,max_input_tokens=2048,alpha_grid=[-2,-1,0,1,2],direction='mean(clean - misleading) final prompt residual on training recognition-known records only; average both option orders',layer='minimum cosine between training clean and conflict mean, excluding final block',selection='highest validation mean accuracy over aligned and misleading contexts, then smallest |alpha|',strict_primary='test records correct on both independent open questions AND both closed-book option orders; no conflict-response filtering',recognition_secondary='test records correct on both closed-book option orders',patient_track='four held-out clinical condition pairs; simplified deterministic text scope gate diagnostic, not full CARE',data_hashes={p.name:sha(p) for p in (ROOT/'data').glob('*.json')})
dump(ROOT/'config.json',config)
print(json.dumps({k:json.loads((ROOT/'audit/data_audit.json').read_text())[k] for k in ['counts','split_counts','group_counts','open_counts']}))
