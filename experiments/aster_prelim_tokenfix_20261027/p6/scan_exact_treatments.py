"""Explore exact treatment-name swaps in development data; no inference."""
import sys,json,re,collections
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import build_candidates as b
TERMS_EN="""aspirin|clopidogrel|ticagrelor|prasugrel|warfarin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran|edoxaban|metformin|insulin|semaglutide|liraglutide|empagliflozin|dapagliflozin|canagliflozin|glipizide|glimepiride|pioglitazone|sitagliptin|surgery|radiotherapy|chemotherapy|immunotherapy|brachytherapy|radiation therapy|surgical resection|amputation|excision|curettage|ablation|embolization|endoscopy|colonoscopy|ultrasound|MRI|CT|PET|biopsy|aspiration|dialysis|transplantation|observation|surveillance|anticoagulation|thrombolysis|thrombectomy|antibiotics|corticosteroids|prednisone|prednisolone|dexamethasone|hydrocortisone|methylprednisolone|budesonide|methotrexate|azathioprine|cyclophosphamide|mycophenolate|rituximab|infliximab|adalimumab|tocilizumab|abatacept|cyclosporine|tacrolimus|hydroxychloroquine|sulfasalazine|leflunomide|amoxicillin|penicillin|ceftriaxone|cefazolin|cefepime|ceftazidime|meropenem|vancomycin|linezolid|daptomycin|azithromycin|clarithromycin|doxycycline|ciprofloxacin|levofloxacin|moxifloxacin|metronidazole|clindamycin|gentamicin|amikacin|rifampicin|isoniazid|ethambutol|pyrazinamide|fluconazole|voriconazole|posaconazole|itraconazole|isavuconazole|caspofungin|micafungin|amphotericin|acyclovir|ganciclovir|foscarnet|oseltamivir|remdesivir|cisplatin|carboplatin|oxaliplatin|paclitaxel|docetaxel|gemcitabine|etoposide|doxorubicin|ifosfamide|vincristine|vinblastine|bleomycin|capecitabine|fluorouracil|irinotecan|topotecan|temozolomide|imatinib|dasatinib|nilotinib|sunitinib|sorafenib|pazopanib|erlotinib|gefitinib|osimertinib|crizotinib|alectinib|nivolumab|pembrolizumab|ipilimumab|atezolizumab|durvalumab|trastuzumab|pertuzumab|bevacizumab|cetuximab|panitumumab|tamoxifen|letrozole|anastrozole|exemestane|fulvestrant|abiraterone|enzalutamide|bicalutamide|leuprolide|goserelin|bisphosphonates|denosumab|zoledronic acid|alendronate|teriparatide|levothyroxine|methimazole|propylthiouracil|radioiodine|lithium|valproate|carbamazepine|lamotrigine|levetiracetam|phenytoin|gabapentin|pregabalin|levodopa|pramipexole|ropinirole|donepezil|memantine|sertraline|fluoxetine|paroxetine|venlafaxine|duloxetine|mirtazapine|amitriptyline|clozapine|olanzapine|risperidone|aripiprazole|quetiapine|haloperidol|morphine|fentanyl|oxycodone|tramadol|ibuprofen|naproxen|diclofenac|celecoxib|paracetamol|acetaminophen|colchicine|allopurinol|febuxostat|probenecid|furosemide|spironolactone|eplerenone|hydrochlorothiazide|chlorthalidone|lisinopril|enalapril|ramipril|losartan|valsartan|candesartan|amlodipine|nifedipine|verapamil|diltiazem|metoprolol|bisoprolol|carvedilol|atenolol|digoxin|amiodarone|sotalol|flecainide|propafenone|atorvastatin|rosuvastatin|simvastatin|ezetimibe|evolocumab|alirocumab|oxygen|ventilation|CPAP|BiPAP|salbutamol|albuterol|formoterol|salmeterol|tiotropium|ipratropium|montelukast|omeprazole|pantoprazole|famotidine|ranitidine|ondansetron|metoclopramide|loperamide|mesalazine""".split('|')
TERMS_ZH="""阿司匹林|氯吡格雷|替格瑞洛|华法林|肝素|利伐沙班|阿哌沙班|达比加群|胰岛素|二甲双胍|放疗|化疗|免疫治疗|手术切除|手术治疗|保守治疗|消融|栓塞|截肢|清创|引流|活检|穿刺|透析|移植|超声|磁共振|地塞米松|泼尼松|甲泼尼龙|氢化可的松|甲氨蝶呤|环磷酰胺|硫唑嘌呤|他克莫司|环孢素|利妥昔单抗|吗替麦考酚酯|羟氯喹|阿莫西林|头孢曲松|万古霉素|美罗培南|左氧氟沙星|环丙沙星|甲硝唑|克林霉素|氟康唑|伏立康唑|泊沙康唑|卡泊芬净|米卡芬净|两性霉素B|阿昔洛韦|更昔洛韦|奥司他韦|顺铂|卡铂|奥沙利铂|紫杉醇|多西他赛|吉西他滨|依托泊苷|多柔比星|异环磷酰胺|长春新碱|卡培他滨|氟尿嘧啶|伊立替康|替莫唑胺|伊马替尼|达沙替尼|舒尼替尼|索拉非尼|培唑帕尼|厄洛替尼|吉非替尼|奥希替尼|克唑替尼|阿来替尼|纳武利尤单抗|帕博利珠单抗|曲妥珠单抗|贝伐珠单抗|西妥昔单抗|他莫昔芬|来曲唑|阿那曲唑|阿比特龙|恩扎卢胺|比卡鲁胺|唑来膦酸|地舒单抗|阿仑膦酸|特立帕肽|左甲状腺素|甲巯咪唑|丙硫氧嘧啶|碘131|丙戊酸|卡马西平|拉莫三嗪|左乙拉西坦|苯妥英|加巴喷丁|普瑞巴林|左旋多巴|普拉克索|多奈哌齐|美金刚|舍曲林|氟西汀|帕罗西汀|文拉法辛|度洛西汀|米氮平|氯氮平|奥氮平|利培酮|阿立哌唑|喹硫平|吗啡|芬太尼|羟考酮|曲马多|布洛芬|萘普生|双氯芬酸|塞来昔布|对乙酰氨基酚|秋水仙碱|别嘌醇|非布司他|呋塞米|螺内酯|依普利酮|氢氯噻嗪|氯沙坦|缬沙坦|氨氯地平|硝苯地平|美托洛尔|比索洛尔|卡维地洛|地高辛|胺碘酮|阿托伐他汀|瑞舒伐他汀|依折麦布|沙丁胺醇|福莫特罗|噻托溴铵|异丙托溴铵|孟鲁司特|奥美拉唑|泮托拉唑|昂丹司琼|甲氧氯普胺|美沙拉嗪|随访|观察""".split('|')
PATTERNS={x:re.compile(r'(?<![A-Za-z0-9])'+re.escape(x)+r'(?![A-Za-z0-9])',re.I) for x in TERMS_EN}
PATTERNS.update({x:re.compile(re.escape(x)) for x in TERMS_ZH})
def terms(s):return {t for t,p in PATTERNS.items() if p.search(s)}
def main():
 lock=json.loads((b.OLD/'mega_dev16_input_lock.json').read_text())
 allowed={f:g['source_cluster'] for g in lock['groups'] if g['partition']=='development' for f in g['filenames']}
 trees={r['filename']:json.loads(r['text']) for r in b.read(b.ROOT/'train_val/decision_trees.jsonl') if r['filename'] in allowed}
 paths={r['filename']:r['paths'] for r in b.read(b.ROOT/'train_val/case_paths.jsonl') if r['filename'] in allowed}
 refs={b.key(r):json.loads(r['output_text']) for r in b.read(b.ROOT/'train_val/gt_paths.jsonl') if r['filename'] in allowed}
 nodes={f:{int(n['index']):(n,p,e) for n,p,e in b.walk(t['decision_tree'])} for f,t in trees.items()}
 counts=collections.Counter();records=[];why=collections.Counter()
 docs={}
 for r in b.read(b.ROOT/'train_val/vignette_cases.jsonl'):
  f=r['filename']
  if f not in allowed:continue
  path=paths[f][int(b.key(r)[1])-1];n,parent,edge=nodes[f][int(path[-1]['index'])]
  children=[v for v in (parent or {}).values() if isinstance(v,dict) and 'index' in v]
  if len(children)!=2 or not all(v.get('leaf') for v in children):why['not_terminal_pair']+=1;continue
  sibling=next(v for v in children if v['index']!=n['index'])
  a=terms(n.get('conclusion',''));bb=terms(sibling.get('conclusion',''));protected=terms(parent.get('description',''))
  aa=a-bb-protected;bs=bb-a-protected
  if not aa or not bs:why['no_distinct_treatment_pair']+=1;continue
  if f not in docs:docs[f]=b.safe_guideline_path(b.ROOT/'train_val/guideline_md',f).read_text()
  ref=refs[b.key(r)]
  aa={x for x in aa if PATTERNS[x].search(docs[f]) and any(PATTERNS[x].search(t) for t in ref)}
  bs={x for x in bs if PATTERNS[x].search(docs[f])}
  if not aa or not bs:why['term_not_in_source_or_reference']+=1;continue
  counts[allowed[f]]+=1
  records.append({'source_case_key':list(b.key(r)),'guideline_id':allowed[f],'applicable_terms':sorted(aa),'sibling_terms':sorted(bs),'condition':parent.get('description'),'native_conclusion':n['conclusion'],'sibling_conclusion':sibling['conclusion'],'decision_nodes':sum('condition' in z for z in path)})
 out=D.parent/'private/p6_lexical_scan';out.mkdir(exist_ok=False)
 (out/'candidates.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
 result={'candidate_count':len(records),'guidelines':len(counts),'guidelines_at_least_five':sum(v>=5 for v in counts.values()),'complex_cases':sum(x['decision_nodes']>2 for x in records),'reasons':dict(why),'counts':dict(counts),'test_files_read':False}
 (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
