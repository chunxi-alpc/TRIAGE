"""Authored research predicates and source-read annotations, fixed before inference.
No dataset prose in this module. Numeric stages use their major stage number.
"""
FIELDS={
'ato_prior':'Arsenic trioxide exposure BEFORE the first relapse/salvage episode, positive if exposed, negative if explicitly never exposed; later salvage ATO does not count',
'relapse_months':'Months from completion of initial ATRA plus anthracycline therapy to the first relapse',
'flt3':'Diagnostic FLT3 mutation result (ITD or TKD), positive/negative',
'pt':'Major pathological T stage of the original urethral surgical specimen, numeric 1/2/3/4, NOT current recurrence stage; pT1/pT2 is interval [1,2]',
'pn':'Major pathological N stage of that same original urethral surgical specimen, numeric 0/1/2',
'ct':'Major clinical breast T stage BEFORE surgery or preoperative therapy, numeric 1/2/3/4',
'cn':'Major clinical breast N stage BEFORE preoperative therapy/surgery, numeric 0/1/2/3; unspecified cN+ is a lower bound >=1',
'preop_chemo':'Whether chemotherapy was received BEFORE breast surgery, positive/negative; adjuvant chemotherapy does not count',
'sln_count':'Positive sentinel lymph node count at breast surgery, NOT number removed or clinical N stage',
'wbrt':'Whole breast radiotherapy explicitly PLANNED for this patient, positive/negative; a generic recommendation alone is unknown',
'ax_nodes':'Positive axillary lymph node count in the breast surgical pathology, NOT total nodes removed',
'tumor_cm':'Largest invasive breast primary tumor diameter in the surgical pathology, cm; NOT preoperative imaging or another lesion',
'margin_mm':'Closest breast surgical resection margin distance, mm',
'ypn':'Major POST-neoadjuvant pathological N stage, numeric 0/1/2/3',
'distant':'Current confirmed distant metastasis, positive/negative; local recurrence alone does not imply distant spread',
'dmmr':'Mismatch repair deficiency / MSI-high in assessed colorectal cancer, positive if dMMR/MSI-H, negative if pMMR/MSS',
'pole':'POLE or POLD1 mutation in assessed colorectal cancer, positive/negative',
'tmb':'Tumor mutational burden of assessed colorectal lesion, mutations/Mb; preserve inequalities',
'csf':'Latest reassessment CSF cytology malignant cells, positive/negative; NOT protein, white blood cells or MRI',
'margin':'Thyroid lobectomy resection margin involvement, positive/negative',
'contra':'Contralateral thyroid lesion on postoperative imaging, positive/negative; planned scan is not an observed result',
'susp_nodes':'Suspicious cervical lymph nodes on postoperative thyroid imaging, positive/negative',
'age':'Patient age in years at the described thyroid surgical decision, NOT relatives age or time since surgery',
'ret634':'Germline RET codon 634 mutation, positive/negative',
'v804m':'Germline RET V804M mutation, positive/negative',
'strong_family':'EXPLICIT clinical assessment of strong family history of medullary thyroid carcinoma, positive/negative; a single affected relative alone does not define strong',
'gross_margin':'GROSS positive margin after thyroid lobectomy, positive/negative; unspecified microscopic vs gross positive is unknown',
'gross_ete':'GROSS extrathyroidal extension in lobectomy specimen, positive/negative; minimal extension alone is unknown',
'thyroid_nodes':'Biopsy/pathology-confirmed cervical nodal thyroid cancer metastases, positive/negative; suspicious imaging alone is unknown',
'confirmed_contra':'Confirmed contralateral thyroid carcinoma, positive/negative; suspicious lesion without biopsy is unknown',
'vascular':'Vascular invasion in thyroid lobectomy pathology, positive/negative',
'resectable':'Current recurrent thyroid cancer surgically resectable, positive/negative',
'iodine_avid':'Current recurrent thyroid tumor radioiodine uptake, positive/negative',
'progressive':'Current recurrent thyroid cancer progression, positive/negative; growth explicitly confirms progression',
'symptoms':'Current cancer-related symptoms anywhere (including locoregional), positive/negative; asymptomatic pulmonary lesions do not exclude symptoms elsewhere',
'indolent':'Current systemic metastatic thyroid disease described as indolent, positive/negative; no explicit assessment is unknown',
'brain_mets':'Confirmed brain metastasis on latest brain imaging, positive/negative; an ordered scan with pending result is unknown'
}
NUMERIC={'relapse_months','pt','pn','ct','cn','sln_count','ax_nodes','tumor_cm','margin_mm','ypn','tmb','age'}
CATEGORY=set(FIELDS)-NUMERIC
NAMES={
'ato_prior':'arsenic trioxide exposure before the first relapse', 'relapse_months':'interval from completion of initial ATRA plus anthracycline therapy to first relapse',
'flt3':'diagnostic FLT3 mutation', 'pt':'original urethral surgical pathological T stage', 'pn':'original urethral surgical pathological N stage',
'ct':'preoperative breast clinical T stage','cn':'preoperative breast clinical N stage','preop_chemo':'chemotherapy received before breast surgery','sln_count':'positive breast sentinel lymph node count','wbrt':'patient-specific plan for whole breast radiotherapy',
'ax_nodes':'positive axillary node count in breast surgical pathology','tumor_cm':'invasive breast tumor diameter in surgical pathology','margin_mm':'closest breast surgical margin distance','ypn':'post-neoadjuvant pathological N stage',
'distant':'distant metastasis','dmmr':'mismatch repair deficiency','pole':'POLE or POLD1 mutation','tmb':'tumor mutational burden','csf':'latest CSF malignant cytology',
'margin':'thyroid lobectomy resection margin involvement','contra':'contralateral thyroid lesion on postoperative imaging','susp_nodes':'suspicious cervical lymph nodes on postoperative thyroid imaging','age':'patient age at the thyroid surgical decision','ret634':'germline RET codon 634 mutation','v804m':'germline RET V804M mutation','strong_family':'strong family history of medullary thyroid carcinoma',
'gross_margin':'gross positive margin after thyroid lobectomy','gross_ete':'gross extrathyroidal extension in thyroid lobectomy pathology','thyroid_nodes':'pathology-confirmed thyroid cancer nodal metastasis','confirmed_contra':'confirmed contralateral thyroid carcinoma','vascular':'vascular invasion in thyroid lobectomy pathology',
'resectable':'resectability of the recurrent thyroid tumor','iodine_avid':'radioiodine uptake in the recurrent thyroid tumor','progressive':'progression of the recurrent thyroid tumor','symptoms':'cancer-related symptoms at any site','indolent':'indolent systemic metastatic thyroid disease','brain_mets':'brain metastasis on completed brain imaging'}
UNITS={k:('months' if k=='relapse_months' else 'cm' if k=='tumor_cm' else 'mm' if k=='margin_mm' else 'mutations/Mb' if k=='tmb' else 'years' if k=='age' else '') for k in FIELDS}
def a(f,o,v):return {'field':f,'op':o,'value':v}
def r(i,row,node,text,atoms,mode='all',scope='complete stated predicate'):
 return {'rule_id':f'V{i:02}','source_row':row,'source_node_index':node,'text':text,'schema_gold':{'mode':mode,'conditions':atoms,'exceptions':[]},'scope':scope}
RULES=[
r(1,5,1,'No arsenic trioxide exposure before first relapse OR relapse less than 6 months after initial ATRA plus anthracycline therapy.',[a('ato_prior','eq','negative'),a('relapse_months','lt',6)],'any'),
r(2,15,1,'A diagnostic FLT3 mutation is present.',[a('flt3','eq','positive')]),
r(3,39,4,'The original urethral surgical specimen is pT1 or pT2 AND pN0.',[a('pt','ge',1),a('pt','le',2),a('pn','eq',0)]),
r(4,62,3,'ALL: preoperative breast clinical T stage 1 to 3 inclusive; clinical N stage 0; no preoperative chemotherapy; 1 to 2 positive sentinel nodes inclusive; whole breast radiotherapy planned.',[a('ct','ge',1),a('ct','le',3),a('cn','eq',0),a('preop_chemo','eq','negative'),a('sln_count','ge',1),a('sln_count','le',2),a('wbrt','eq','positive')]),
r(5,65,2,'Breast surgical pathology has zero positive axillary nodes AND invasive tumor diameter at most 5 cm AND closest resection margin at least 1 mm.',[a('ax_nodes','eq',0),a('tumor_cm','le',5),a('margin_mm','ge',1)]),
r(6,71,3,'Clinical breast nodes were positive before neoadjuvant treatment AND postoperative nodes are ypN0.',[a('cn','ge',1),a('ypn','eq',0)]),
r(7,92,2,'Current distant metastasis is absent.',[a('distant','eq','negative')]),
r(8,144,2,'The colorectal cancer is mismatch-repair proficient / microsatellite stable.',[a('dmmr','eq','negative')]),
r(9,145,2,'POLE or POLD1 mutation is present AND tumor mutational burden exceeds 50 mutations/Mb.',[a('pole','eq','positive'),a('tmb','gt',50)],scope='POLE/POLD1 numeric branch only; excludes alternative dMMR/MSI-H branch'),
r(10,150,0,'Latest CSF cytology is negative for malignant cells.',[a('csf','eq','negative')]),
r(11,211,2,'Thyroid lobectomy margins are negative AND postoperative imaging shows no contralateral thyroid lesion AND no suspicious cervical lymph node.',[a('margin','eq','negative'),a('contra','eq','negative'),a('susp_nodes','eq','negative')],scope='three-condition branch only, excludes alternative NIFTP branch'),
r(12,259,3,'RET codon 634 mutation is present AND patient age at surgical decision is below 5 years.',[a('ret634','eq','positive'),a('age','lt',5)],scope='genotype and timing projection, not complete surgical eligibility'),
r(13,261,3,'RET V804M mutation is present AND strong family history of medullary thyroid carcinoma is explicitly absent.',[a('v804m','eq','positive'),a('strong_family','eq','negative')]),
r(14,208,2,'ANY: gross positive thyroid lobectomy margins; gross extrathyroidal extension; confirmed cervical nodal metastasis; confirmed contralateral thyroid carcinoma; vascular invasion.',[a(f,'eq','positive') for f in ['gross_margin','gross_ete','thyroid_nodes','confirmed_contra','vascular']],'any'),
r(15,218,2,'Recurrent thyroid cancer is unresectable AND non-radioiodine-avid AND progressive.',[a('resectable','eq','negative'),a('iodine_avid','eq','negative'),a('progressive','eq','positive')]),
r(16,225,4,'The patient is asymptomatic from cancer AND systemic metastatic thyroid disease is indolent AND brain metastasis is absent.',[a('symptoms','eq','negative'),a('indolent','eq','positive'),a('brain_mets','eq','negative')])]
# Each evidence field is a list of acceptable alternative sufficient sentence groups.
CASES={
'V01':[(5805,{'ato_prior':'negative','relapse_months':4},{'ato_prior':[[1]],'relapse_months':[[1]]}), (1954,{'ato_prior':'negative','relapse_months':6},{'ato_prior':[[5]],'relapse_months':[[3]]})],
'V02':[(2515,{'flt3':'positive'},{'flt3':[[3]]}),(4721,{'flt3':'positive'},{'flt3':[[4]]})],
'V03':[(1983,{'pt':{'ge':1,'le':2},'pn':0},{'pt':[[3]],'pn':[[3]]}),(6372,{'pt':1,'pn':0},{'pt':[[1]],'pn':[[1]]})],
'V04':[(7526,{'ct':2,'cn':0,'preop_chemo':'negative','sln_count':2,'wbrt':'positive'},{'ct':[[2],[6]],'cn':[[2],[6]],'preop_chemo':[[2],[6]],'sln_count':[[4],[6]],'wbrt':[[6]]}),(4491,{'ct':2,'cn':0,'preop_chemo':'negative','sln_count':2,'wbrt':'positive'},{'ct':[[4]],'cn':[[4]],'preop_chemo':[[7]],'sln_count':[[6]],'wbrt':[[9]]})],
'V05':[(6397,{'ax_nodes':0,'tumor_cm':3.5,'margin_mm':2},{f:[[1]] for f in ['ax_nodes','tumor_cm','margin_mm']}),(5038,{'ax_nodes':0,'tumor_cm':2.5,'margin_mm':2.5},{'ax_nodes':[[5],[6]],'tumor_cm':[[4]],'margin_mm':[[4]]})],
'V06':[(3403,{'cn':None,'ypn':0},{'cn':[],'ypn':[[5]]}),(7536,{'cn':{'ge':1},'ypn':0},{'cn':[[1],[6]],'ypn':[[5],[6]]})],
'V07':[(6153,{'distant':'negative'},{'distant':[[2],[3]]}),(923,{'distant':'negative'},{'distant':[[5],[6]]})],
'V08':[(5391,{'dmmr':'negative'},{'dmmr':[[6],[8]]}),(2922,{'dmmr':'negative'},{'dmmr':[[7]]})],
'V09':[(2087,{'pole':'positive','tmb':{'gt':50}},{'pole':[[5]],'tmb':[[5]]}),(2361,{'pole':'positive','tmb':{'gt':50}},{'pole':[[4]],'tmb':[[4]]})],
'V10':[(5939,{'csf':'negative'},{'csf':[[2]]}),(2092,{'csf':'negative'},{'csf':[[5]]})],
'V11':[(7675,{'margin':'negative','contra':'negative','susp_nodes':'negative'},{'margin':[[1],[3]],'contra':[[2],[3]],'susp_nodes':[[2],[3]]}),(3543,{'margin':'negative','contra':'negative','susp_nodes':'negative'},{f:[[5]] for f in ['margin','contra','susp_nodes']})],
'V12':[(4958,{'age':4,'ret634':'positive'},{'age':[[1]],'ret634':[[1],[7]]}),(1926,{'age':4,'ret634':'positive'},{'age':[[1]],'ret634':[[2]]})],
'V13':[(1090,{'v804m':'positive','strong_family':None},{'v804m':[[2]],'strong_family':[]}),(2478,{'v804m':'positive','strong_family':'negative'},{'v804m':[[2]],'strong_family':[[6]]})],
'V14':[(5180,{'gross_margin':None,'gross_ete':None,'thyroid_nodes':'positive','confirmed_contra':None,'vascular':'positive'},{'gross_margin':[],'gross_ete':[],'thyroid_nodes':[[4],[9]],'confirmed_contra':[],'vascular':[[3],[9]]}),(7107,{'gross_margin':None,'gross_ete':'positive','thyroid_nodes':'positive','confirmed_contra':None,'vascular':None},{'gross_margin':[],'gross_ete':[[3]],'thyroid_nodes':[[3]],'confirmed_contra':[],'vascular':[]})],
'V15':[(1047,{'resectable':'negative','iodine_avid':'negative','progressive':'positive'},{'resectable':[[5]],'iodine_avid':[[3]],'progressive':[[4]]}),(4647,{'resectable':'negative','iodine_avid':'negative','progressive':'positive'},{'resectable':[[5]],'iodine_avid':[[6],[9]],'progressive':[[8],[9]]})],
'V16':[(6283,{'symptoms':'positive','indolent':'positive','brain_mets':'negative'},{'symptoms':[[1],[5]],'indolent':[[5]],'brain_mets':[[3]]}),(3556,{'symptoms':None,'indolent':None,'brain_mets':None},{'symptoms':[],'indolent':[],'brain_mets':[]})]
}
