#!/usr/bin/env python3
"""Build source-grounded clinical rules with simulated, executable patient pairs."""
from __future__ import annotations
import collections, hashlib, json, pathlib, random
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'data/clinical'; OUT.mkdir(exist_ok=True)
SEED=261016
# Paraphrases of public reference rules; no altered clinical thresholds or action swaps.
RULES=[
dict(family='glycemia',rule='a1c',source='https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis',
 section='Test results for diagnosis of prediabetes and diabetes',query='What glycemic category does the laboratory A1C result indicate?',
 context='Nonpregnant adult without a condition that invalidates A1C. Laboratory A1C is used for category interpretation, not a complete diagnosis.',
 field='laboratory A1C',unit='%',kind='a1c',categories=['Normal glycemia','Prediabetes','Diabetes range','Uninterpretable result'],
 guidance='For nonpregnant adults, if A1C is below 5.7%, assign Normal glycemia; from 5.7% to below 6.5%, assign Prediabetes; at 6.5% or higher, assign Diabetes range. A diagnosis usually requires confirmation; this question asks only for the test-result category.',
 target='Prediabetes',app=[5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4],non=[6.5,6.6,6.8,7.0,7.2,7.5,8.0,8.5]),
dict(family='glycemia',rule='fpg',source='https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis',
 section='Test results table, fasting plasma glucose',query='What glycemic category does the fasting laboratory plasma glucose indicate?',
 context='Nonpregnant adult after an eight-hour fast; sample is laboratory plasma, not a home meter. Interpret the result category only.',
 field='fasting plasma glucose',unit='mg/dL',kind='fpg',categories=['Normal glycemia','Prediabetes','Diabetes range','Uninterpretable result'],
 guidance='For fasting laboratory plasma glucose in nonpregnant adults, below 100 mg/dL assign Normal glycemia; from 100 to below 126 mg/dL assign Prediabetes; at 126 mg/dL or above assign Diabetes range. Confirmatory testing is usually needed for diagnosis.',
 target='Prediabetes',app=[100,103,106,110,114,118,122,125],non=[82,85,88,90,93,95,97,99]),
dict(family='adiposity',rule='bmi_over',source='https://fingertips.phe.org.uk/profile/obesity-physical-activity-nutrition/supporting-information/definitions',
 section='Definition of adult BMI categories',query='What standard adult BMI category applies?',
 context='Nonpregnant White European adult, without unusually high muscularity. Use standard adult BMI categories, not a full adiposity diagnosis.',
 field='BMI',unit='kg/m2',kind='bmi',categories=['Underweight','Healthy weight','Overweight','Obesity'],
 guidance='For the standard adult BMI categories, below 18.5 kg/m2 assign Underweight; from 18.5 to below 25 assign Healthy weight; from 25 to below 30 assign Overweight; at 30 or higher assign Obesity.',
 target='Overweight',app=[25,25.5,26,26.5,27,28,29,29.9],non=[30,30.5,31,32,33,34,35,36]),
dict(family='adiposity',rule='bmi_healthy',source='https://fingertips.phe.org.uk/profile/obesity-physical-activity-nutrition/supporting-information/definitions',
 section='Definition of adult BMI categories',query='What standard adult BMI category applies?',
 context='Nonpregnant White European adult, without unusually high muscularity. Use standard adult BMI categories, not a full adiposity diagnosis.',
 field='BMI',unit='kg/m2',kind='bmi',categories=['Underweight','Healthy weight','Overweight','Obesity'],
 guidance='For the standard adult BMI categories, below 18.5 kg/m2 assign Underweight; from 18.5 to below 25 assign Healthy weight; from 25 to below 30 assign Overweight; at 30 or higher assign Obesity.',
 target='Healthy weight',app=[18.5,19,20,21,22,23,24,24.9],non=[16,16.3,16.6,17,17.3,17.6,18,18.4]),
dict(family='renal',rule='gfr_3a',source='https://www.kidney.org/why-classify-ckd',
 section='GFR categories G1-G5',query='What GFR category applies to this established CKD patient?',
 context='Adult with established chronic kidney disease; eGFR has been persistent for over three months, without acute kidney injury.',
 field='eGFR',unit='mL/min/1.73m2',kind='gfr',categories=['G2','G3a','G3b','G4'],
 guidance='For GFR categorization, eGFR at least 90 corresponds to G1; 60 to below 90 to G2; 45 to below 60, assign G3a; 30 to below 45, assign G3b; 15 to below 30, assign G4; below 15 to G5. G1/G2 alone do not establish CKD.',
 target='G3a',app=[45,46,48,50,52,55,57,59],non=[30,32,34,36,38,40,42,44]),
dict(family='renal',rule='gfr_4',source='https://www.kidney.org/why-classify-ckd',
 section='GFR categories G1-G5',query='What GFR category applies to this established CKD patient?',
 context='Adult with established chronic kidney disease; eGFR has been persistent for over three months, without acute kidney injury.',
 field='eGFR',unit='mL/min/1.73m2',kind='gfr',categories=['G2','G3a','G3b','G4'],
 guidance='For GFR categorization, eGFR at least 90 corresponds to G1; 60 to below 90 to G2; 45 to below 60, assign G3a; 30 to below 45, assign G3b; 15 to below 30, assign G4; below 15 to G5. G1/G2 alone do not establish CKD.',
 target='G4',app=[15,17,19,21,23,25,27,29],non=[30,32,34,36,38,40,42,44]),
dict(family='blood_pressure',rule='clinic_target',source='https://www.nice.org.uk/guidance/ng136/chapter/recommendations',
 section='Recommendations 1.4.20-1.4.21',query='Which clinic blood-pressure target is recommended by NICE NG136?',
 context='Adult with treated hypertension, without type 1 diabetes, severe CKD, frailty or multimorbidity requiring an individualized target.',
 field='age',unit='years',kind='bp_clinic',categories=['Below 140/90 mmHg','Below 150/90 mmHg','Below 130/80 mmHg','Below 160/100 mmHg'],
 guidance='For adults with hypertension under age 80, assign Below 140/90 mmHg as the clinic blood-pressure target; at age 80 or older, assign Below 150/90 mmHg. Individualize for frailty or multimorbidity; special disease targets are outside this case.',
 target='Below 140/90 mmHg',app=[65,67,69,71,73,75,77,79],non=[80,81,82,83,84,85,86,87]),
dict(family='blood_pressure',rule='home_target',source='https://www.nice.org.uk/guidance/ng136/chapter/recommendations',
 section='Recommendation 1.4.22',query='Which usual-waking-hours home/ambulatory blood-pressure target is recommended by NICE NG136?',
 context='Adult with treated hypertension, without type 1 diabetes, severe CKD, frailty or multimorbidity requiring an individualized target.',
 field='age',unit='years',kind='bp_home',categories=['Below 135/85 mmHg','Below 145/85 mmHg','Below 130/80 mmHg','Below 150/90 mmHg'],
 guidance='For monitoring hypertension by usual-waking-hours home or ambulatory readings, under age 80 assign Below 135/85 mmHg; at age 80 or older assign Below 145/85 mmHg. Individualize for frailty or multimorbidity.',
 target='Below 135/85 mmHg',app=[65,67,69,71,73,75,77,79],non=[80,81,82,83,84,85,86,87]),
dict(family='venous_thromboembolism',rule='pe_wells',source='https://www.nice.org.uk/guidance/ng158/chapter/recommendations',
 section='Table 2, two-level PE Wells score',query='Which two-level PE Wells probability category applies?',
 context='Adult with suspected pulmonary embolism. The PE Wells score has already been correctly calculated; classify probability, not confirmed diagnosis.',
 field='PE Wells score',unit='points',kind='pe',categories=['PE likely','PE unlikely','DVT likely','DVT unlikely'],
 guidance='In the two-level PE Wells scheme, for a score greater than 4 assign PE likely; for a score of 4 or less assign PE unlikely.',
 target='PE likely',app=[4.5,5,5.5,6,6.5,7,8,9],non=[0,1,1.5,2,2.5,3,3.5,4]),
dict(family='venous_thromboembolism',rule='dvt_wells',source='https://www.nice.org.uk/guidance/ng158/chapter/recommendations',
 section='Table 1, two-level DVT Wells score',query='Which two-level DVT Wells probability category applies?',
 context='Adult with suspected deep-vein thrombosis. The DVT Wells score has already been correctly calculated; classify probability, not confirmed diagnosis.',
 field='DVT Wells score',unit='points',kind='dvt',categories=['PE likely','PE unlikely','DVT likely','DVT unlikely'],
 guidance='In the two-level DVT Wells scheme, for a score of 2 or greater assign DVT likely; for a score of 1 or less assign DVT unlikely.',
 target='DVT likely',app=[2,3,4,5,6,7,8,9],non=[-2,-1,0,1,-2,-1,0,1]),
dict(family='pneumonia',rule='crb65',source='https://www.nice.org.uk/guidance/NG250/chapter/recommendations',
 section='Box 1, CRB65 mortality risk in primary care',query='Which CRB65 mortality-risk band applies?',
 context='Adult age 52 with community-acquired pneumonia assessed in primary care. The CRB65 score is correctly calculated. Ask for the score risk band, not a complete disposition decision.',
 field='CRB65 score',unit='points',kind='crb',categories=['Low risk','Intermediate risk','High risk','Cannot classify from this score'],
 guidance='For the CRB65 mortality-risk bands, at score 0 assign Low risk; at score 1 or 2 assign Intermediate risk; at score 3 or 4 assign High risk. Clinical judgment also informs management.',
 target='Low risk',app=[0]*8,non=[1,2,1,2,1,2,1,2]),
dict(family='pneumonia',rule='curb65',source='https://www.nice.org.uk/guidance/NG250/chapter/recommendations',
 section='Box 2, CURB65 mortality risk in hospital',query='Which CURB65 mortality-risk band applies?',
 context='Adult age 52 with community-acquired pneumonia assessed in hospital. The CURB65 score is correctly calculated. Ask for the score risk band, not a complete disposition decision.',
 field='CURB65 score',unit='points',kind='curb',categories=['Low risk','Intermediate risk','High risk','Cannot classify from this score'],
 guidance='For CURB65 mortality-risk bands, at score 0 or 1 assign Low risk; at score 2 assign Intermediate risk; at score 3 to 5 assign High risk. Clinical judgment also informs management.',
 target='High risk',app=[3,4,3,4,3,4,3,4],non=[2]*8),
dict(family='sore_throat',rule='centor',source='https://www.nice.org.uk/guidance/ng84/chapter/recommendations',
 section='Recommendations 1.1.6 and 1.1.10',query='Which antibiotic-prescribing recommendation wording matches NICE NG84 for this score?',
 context='Adult with acute sore throat, not systemically very unwell, without high-risk comorbidity, red flags or suppurative complications.',
 field='Centor score',unit='points',kind='centor',categories=['Do not offer an antibiotic prescription','Consider no antibiotic or a back-up prescription','Consider an immediate or a back-up prescription','Refer immediately for hospital treatment'],
 guidance='For otherwise uncomplicated acute sore throat, Centor scores 0, 1 or 2: assign Do not offer an antibiotic prescription. Centor scores 3 or 4: assign Consider an immediate or a back-up prescription. Very unwell or high-risk patients need separate assessment.',
 target='Consider an immediate or a back-up prescription',app=[3,4,3,4,3,4,3,4],non=[0,1,2,0,1,2,0,2]),
dict(family='sore_throat',rule='feverpain',source='https://www.nice.org.uk/guidance/ng84/chapter/recommendations',
 section='Recommendations 1.1.6, 1.1.8 and 1.1.10',query='Which antibiotic-prescribing recommendation wording matches NICE NG84 for this score?',
 context='Adult with acute sore throat, not systemically very unwell, without high-risk comorbidity, red flags or suppurative complications.',
 field='FeverPAIN score',unit='points',kind='feverpain',categories=['Do not offer an antibiotic prescription','Consider no antibiotic or a back-up prescription','Consider an immediate or a back-up prescription','Refer immediately for hospital treatment'],
 guidance='For otherwise uncomplicated acute sore throat, FeverPAIN 0 or 1: assign Do not offer an antibiotic prescription; 2 or 3: assign Consider no antibiotic or a back-up prescription; 4 or 5: assign Consider an immediate or a back-up prescription.',
 target='Consider no antibiotic or a back-up prescription',app=[2,3,2,3,2,3,2,3],non=[0,1,0,1,0,1,0,1]),
dict(family='atrial_fibrillation',rule='af_zero',source='https://www.nice.org.uk/guidance/NG196/chapter/recommendations',
 section='Recommendations 1.6.3, 1.6.4 and 1.6.7',query='Which stroke-prevention recommendation wording matches NICE NG196 for this man?',
 context='Man age 55 with nonvalvular atrial fibrillation, correctly calculated CHA2DS2-VASc score, no contraindication to a DOAC and no acute bleeding. Ask for the score-based recommendation wording.',
 field='CHA2DS2-VASc score',unit='points',kind='af',categories=['Do not offer stroke-prevention anticoagulation','Consider a direct-acting oral anticoagulant','Offer a direct-acting oral anticoagulant','Offer aspirin alone for stroke prevention'],
 guidance='In men under 65 with atrial fibrillation, if CHA2DS2-VASc is 0 assign Do not offer stroke-prevention anticoagulation; if it is 1 assign Consider a direct-acting oral anticoagulant, accounting for bleeding risk; if it is 2 or more assign Offer a direct-acting oral anticoagulant, accounting for bleeding risk and suitability.',
 target='Do not offer stroke-prevention anticoagulation',app=[0]*8,non=[2,3,4,5,2,3,4,5]),
dict(family='atrial_fibrillation',rule='af_one',source='https://www.nice.org.uk/guidance/NG196/chapter/recommendations',
 section='Recommendations 1.6.3, 1.6.4 and 1.6.7',query='Which stroke-prevention recommendation wording matches NICE NG196 for this man?',
 context='Man age 55 with nonvalvular atrial fibrillation, correctly calculated CHA2DS2-VASc score, no contraindication to a DOAC and no acute bleeding. Ask for the score-based recommendation wording.',
 field='CHA2DS2-VASc score',unit='points',kind='af',categories=['Do not offer stroke-prevention anticoagulation','Consider a direct-acting oral anticoagulant','Offer a direct-acting oral anticoagulant','Offer aspirin alone for stroke prevention'],
 guidance='In men under 65 with atrial fibrillation, if CHA2DS2-VASc is 0 assign Do not offer stroke-prevention anticoagulation; if it is 1 assign Consider a direct-acting oral anticoagulant, accounting for bleeding risk; if it is 2 or more assign Offer a direct-acting oral anticoagulant, accounting for bleeding risk and suitability.',
 target='Consider a direct-acting oral anticoagulant',app=[1]*8,non=[2,3,4,5,2,3,4,5]),
dict(family='bone_density',rule='osteopenia',source='https://www.niams.nih.gov/health-topics/bone-mineral-density-tests-what-numbers-mean',
 section='What does a T-score mean?',query='Which DXA T-score range category applies, rather than a complete clinical diagnosis?',
 context='Postmenopausal woman age 58. The measurement is a valid DXA T-score, not a Z-score.',
 field='DXA T-score',unit='SD',kind='bone',categories=['Normal bone density','Low bone mass','Osteoporosis range','Below expected range for age'],
 guidance='For a postmenopausal adult DXA T-score, at -1 or higher assign Normal bone density; below -1 but above -2.5 assign Low bone mass; at -2.5 or lower assign Osteoporosis range. These are T-score range categories rather than a complete diagnosis.',
 target='Low bone mass',app=[-1.1,-1.3,-1.5,-1.7,-1.9,-2.1,-2.3,-2.4],non=[-1,-.8,-.6,-.4,-.2,0,.2,.5]),
dict(family='bone_density',rule='osteoporosis',source='https://www.niams.nih.gov/health-topics/bone-mineral-density-tests-what-numbers-mean',
 section='What does a T-score mean?',query='Which DXA T-score range category applies, rather than a complete clinical diagnosis?',
 context='Postmenopausal woman age 58. The measurement is a valid DXA T-score, not a Z-score.',
 field='DXA T-score',unit='SD',kind='bone',categories=['Normal bone density','Low bone mass','Osteoporosis range','Below expected range for age'],
 guidance='For a postmenopausal adult DXA T-score, at -1 or higher assign Normal bone density; below -1 but above -2.5 assign Low bone mass; at -2.5 or lower assign Osteoporosis range. These are T-score range categories rather than a complete diagnosis.',
 target='Osteoporosis range',app=[-2.5,-2.7,-2.9,-3.1,-3.3,-3.5,-3.7,-4],non=[-1.1,-1.3,-1.5,-1.7,-1.9,-2.1,-2.3,-2.4]),
dict(family='anaemia',rule='anaemia_mild',source='https://www.ncbi.nlm.nih.gov/books/NBK602185/table/ch5.tab3/',
 section='WHO 2024 Table 3, adults 15-65 nonpregnant women',query='What WHO 2024 anaemia severity category applies?',
 context='Nonpregnant woman age 40, nonsmoker at sea level. Haemoglobin is measured in g/L and requires no altitude adjustment.',
 field='haemoglobin',unit='g/L',kind='anaemia',categories=['No anaemia','Mild anaemia','Moderate anaemia','Severe anaemia'],
 guidance='For nonpregnant women aged 15 to 65, haemoglobin at least 120 g/L: assign No anaemia; 110 to below 120: assign Mild anaemia; 80 to below 110: assign Moderate anaemia; below 80: assign Severe anaemia.',
 target='Mild anaemia',app=[110,111,112,113,114,115,117,119],non=[80,84,88,92,96,100,105,109]),
dict(family='anaemia',rule='anaemia_moderate',source='https://www.ncbi.nlm.nih.gov/books/NBK602185/table/ch5.tab3/',
 section='WHO 2024 Table 3, adults 15-65 nonpregnant women',query='What WHO 2024 anaemia severity category applies?',
 context='Nonpregnant woman age 40, nonsmoker at sea level. Haemoglobin is measured in g/L and requires no altitude adjustment.',
 field='haemoglobin',unit='g/L',kind='anaemia',categories=['No anaemia','Mild anaemia','Moderate anaemia','Severe anaemia'],
 guidance='For nonpregnant women aged 15 to 65, haemoglobin at least 120 g/L: assign No anaemia; 110 to below 120: assign Mild anaemia; 80 to below 110: assign Moderate anaemia; below 80: assign Severe anaemia.',
 target='Moderate anaemia',app=[80,84,88,92,96,100,105,109],non=[60,63,66,69,72,75,77,79]),
dict(family='copd',rule='copd_moderate',source='https://www.nice.org.uk/guidance/ng115/chapter/Recommendations',
 section='Recommendation 1.1.26, Table 4, NICE CG101 column',query='What NICE CG101 spirometric airflow-obstruction stage applies?',
 context='Adult with confirmed stable COPD and chronic respiratory symptoms. Post-bronchodilator FEV1/FVC is 0.60. There is no respiratory failure.',
 field='post-bronchodilator FEV1 percent predicted',unit='%',kind='copd',categories=['Stage 1 (mild)','Stage 2 (moderate)','Stage 3 (severe)','Stage 4 (very severe)'],
 guidance='Using the NICE CG101 spirometric classification with FEV1/FVC below 0.7 and no respiratory failure: FEV1 at least 80% predicted, assign Stage 1 (mild); 50 to below 80%, assign Stage 2 (moderate); 30 to below 50%, assign Stage 3 (severe); below 30%, assign Stage 4 (very severe).',
 target='Stage 2 (moderate)',app=[50,53,56,60,65,70,75,79],non=[30,33,36,39,42,45,47,49]),
dict(family='copd',rule='copd_mild',source='https://www.nice.org.uk/guidance/ng115/chapter/Recommendations',
 section='Recommendation 1.1.26, Table 4, NICE CG101 column',query='What NICE CG101 spirometric airflow-obstruction stage applies?',
 context='Adult with confirmed stable COPD and chronic respiratory symptoms. Post-bronchodilator FEV1/FVC is 0.60. There is no respiratory failure.',
 field='post-bronchodilator FEV1 percent predicted',unit='%',kind='copd',categories=['Stage 1 (mild)','Stage 2 (moderate)','Stage 3 (severe)','Stage 4 (very severe)'],
 guidance='Using the NICE CG101 spirometric classification with FEV1/FVC below 0.7 and no respiratory failure: FEV1 at least 80% predicted, assign Stage 1 (mild); 50 to below 80%, assign Stage 2 (moderate); 30 to below 50%, assign Stage 3 (severe); below 30%, assign Stage 4 (very severe).',
 target='Stage 1 (mild)',app=[80,82,84,86,88,90,94,98],non=[50,53,56,60,65,70,75,79]),
dict(family='hypoglycemia',rule='hypo_one',source='https://doi.org/10.2337/dc26-s006',
 section='ADA Standards of Care 2026, Table 6.4',query='What ADA biochemical hypoglycemia category applies?',
 context='Adult with diabetes, alert and able to self-treat, without altered function or a need for another person to assist recovery.',
 field='blood glucose',unit='mg/dL',kind='hypo',categories=['No biochemical hypoglycemia','Level 1 hypoglycemia','Level 2 hypoglycemia','Level 3 hypoglycemia'],
 guidance='Glucose below 70 but at least 54 mg/dL: assign Level 1 hypoglycemia; below 54 mg/dL: assign Level 2 hypoglycemia. At least 70 mg/dL is outside these biochemical hypoglycemia ranges: assign No biochemical hypoglycemia. Level 3 instead requires altered functioning and assistance for recovery, irrespective of glucose.',
 target='Level 1 hypoglycemia',app=[54,56,58,60,62,64,67,69],non=[40,42,44,46,48,50,52,53]),
dict(family='hypoglycemia',rule='hypo_two',source='https://doi.org/10.2337/dc26-s006',
 section='ADA Standards of Care 2026, Table 6.4',query='What ADA biochemical hypoglycemia category applies?',
 context='Adult with diabetes, alert and able to self-treat, without altered function or a need for another person to assist recovery.',
 field='blood glucose',unit='mg/dL',kind='hypo',categories=['No biochemical hypoglycemia','Level 1 hypoglycemia','Level 2 hypoglycemia','Level 3 hypoglycemia'],
 guidance='Glucose below 70 but at least 54 mg/dL: assign Level 1 hypoglycemia; below 54 mg/dL: assign Level 2 hypoglycemia. At least 70 mg/dL is outside these biochemical hypoglycemia ranges: assign No biochemical hypoglycemia. Level 3 instead requires altered functioning and assistance for recovery, irrespective of glucose.',
 target='Level 2 hypoglycemia',app=[40,42,44,46,48,50,52,53],non=[54,56,58,60,62,64,67,69]),
]
def classify(kind,x):
    # Independent executable oracle; never infer gold from the model or option position.
    if kind=='a1c': return 'Normal glycemia' if x<5.7 else 'Prediabetes' if x<6.5 else 'Diabetes range'
    if kind=='fpg': return 'Normal glycemia' if x<100 else 'Prediabetes' if x<126 else 'Diabetes range'
    if kind=='bmi': return 'Underweight' if x<18.5 else 'Healthy weight' if x<25 else 'Overweight' if x<30 else 'Obesity'
    if kind=='gfr': return 'G1' if x>=90 else 'G2' if x>=60 else 'G3a' if x>=45 else 'G3b' if x>=30 else 'G4' if x>=15 else 'G5'
    if kind=='bp_clinic': return 'Below 140/90 mmHg' if x<80 else 'Below 150/90 mmHg'
    if kind=='bp_home': return 'Below 135/85 mmHg' if x<80 else 'Below 145/85 mmHg'
    if kind=='pe': return 'PE likely' if x>4 else 'PE unlikely'
    if kind=='dvt': return 'DVT likely' if x>=2 else 'DVT unlikely'
    if kind=='crb': return 'Low risk' if x==0 else 'Intermediate risk' if x<=2 else 'High risk'
    if kind=='curb': return 'Low risk' if x<=1 else 'Intermediate risk' if x==2 else 'High risk'
    if kind=='centor': return 'Do not offer an antibiotic prescription' if x<=2 else 'Consider an immediate or a back-up prescription'
    if kind=='feverpain': return 'Do not offer an antibiotic prescription' if x<=1 else 'Consider no antibiotic or a back-up prescription' if x<=3 else 'Consider an immediate or a back-up prescription'
    if kind=='af': return 'Do not offer stroke-prevention anticoagulation' if x==0 else 'Consider a direct-acting oral anticoagulant' if x==1 else 'Offer a direct-acting oral anticoagulant'
    if kind=='bone': return 'Normal bone density' if x>=-1 else 'Low bone mass' if x> -2.5 else 'Osteoporosis range'
    if kind=='anaemia': return 'No anaemia' if x>=120 else 'Mild anaemia' if x>=110 else 'Moderate anaemia' if x>=80 else 'Severe anaemia'
    if kind=='copd': return 'Stage 1 (mild)' if x>=80 else 'Stage 2 (moderate)' if x>=50 else 'Stage 3 (severe)' if x>=30 else 'Stage 4 (very severe)'
    if kind=='hypo': return 'No biochemical hypoglycemia' if x>=70 else 'Level 1 hypoglycemia' if x>=54 else 'Level 2 hypoglycemia'
    raise ValueError(kind)
def main():
    rng=random.Random(SEED); families=sorted({r['family'] for r in RULES}); shuffled=families.copy(); rng.shuffle(shuffled)
    folds={f:i%3 for i,f in enumerate(shuffled)}
    rows=[]; cards=[]
    for rule in RULES:
        card={k:v for k,v in rule.items() if k not in ['app','non']}
        card.update(source_verified=True,verification='Primary reference inspected; thresholds, scope, exceptions and output wording checked against cited section; executable boundary tests. No independent clinician adjudication.',
            source_type='authentic_clinical_rule_paraphrase',patient_type='simulated',counterfactual_guideline=False)
        cards.append(card)
        for j,(a,b) in enumerate(zip(rule['app'],rule['non'])):
            names=rule['categories'].copy(); rng.shuffle(names)
            options=dict(zip('ABCD',names)); inv={v:k for k,v in options.items()}
            pair=f"{rule['family']}-{rule['rule']}-{j:02d}"
            for state,val in [('app',a),('non',b)]:
                answer=classify(rule['kind'],val)
                assert answer in inv and (answer==rule['target'])==(state=='app')
                # Pair members have identical wording, candidates and rule; only the measured value changes.
                patient=f"Patient: {rule['context']} {rule['field']} = {val} {rule['unit']}."
                option_text='\n'.join(f'{k}. {v}' for k,v in options.items())
                query=rule['query']
                closed=f'Question: {query}\n{patient}\nOptions:\n{option_text}\nAnswer (A/B/C/D):'
                paraphrase=f'Patient vignette:\n{patient}\nTask: {query} Select the matching option.\nOptions:\n{option_text}\nAnswer (A/B/C/D):'
                prefix=f'Question: {query}\n{patient}\nClinical guidance: '
                prompt=prefix+rule['guidance']+f'\nUse the supplied clinical guidance for this question.\nOptions:\n{option_text}\nAnswer (A/B/C/D):'
                action='assign '+rule['target']
                start=prompt.index(action,len(prefix))
                # Some tokenizers merge the closing parenthesis with punctuation.
                # Include the existing adjacent delimiter; never change prompt text.
                following=prompt[start+len(action):start+len(action)+1]
                if following in [';',',','.']: action+=following
                span=[start,start+len(action)]
                assert span[1]<prompt.index('\nOptions:')
                rows.append(dict(case_id=pair+'-'+state,pair_id=pair,group_id=rule['family'],rule_id=rule['rule'],
                    outer_fold=folds[rule['family']],pair_index=j,state=state,subset=state,applicable=state=='app',
                    value=val,kind=rule['kind'],source_url=rule['source'],source_section=rule['section'],source_verified=True,
                    source_type='authentic_clinical_rule_paraphrase',patient_type='simulated',counterfactual_guideline=False,
                    model='Qwen/Qwen3-4B-Base',seed=SEED,query=query,patient=patient,guidance=rule['guidance'],
                    target_action=action,option_map=options,gold_label=answer,gold_answer=inv[answer],
                    inapplicable_action=inv[rule['target']],closedbook_prompt=closed,paraphrase_prompt=paraphrase,
                    prompt=prompt,spans={'action':span}))
    (OUT/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'rule_cards.json').write_text(json.dumps(cards,indent=2,ensure_ascii=False))
    protocol=dict(seed=SEED,cases=len(rows),pairs=len(rows)//2,families=len(families),rules=len(RULES),
        clinical_rule_family_outer_folds=folds,outer_folds=3,inner_dev_rule_families=2,
        strengths={'seka':[.25,.5,1.,2.,4.],'pasta':[.25,.5,1.,2.,4.605170186]},
        selection='development all-case harmonic mean of app/non accuracy; tie smaller strength',
        eligibility='source verified AND same argmax letter for two closedbook formulations AND closedbook argmax differs from rule gold; NO confidence threshold',
        main_unit='eligible cases, separate app/non denominators',secondary_unit='complete pairs with both members eligible',
        screening_order='all candidates screened before intervention results',confidence_interval='10000 clinical-rule-family cluster bootstrap replicates',
        cross_fitting='Each family has one outer-test fold. Projection and strength selection exclude all examples of its family.',
        source_scope='Authentic public clinical reference rules, faithfully paraphrased; simulated patients; no guideline counterfactuals or patient records.',
        no_claim='Stable closedbook disagreement is an operational measure, not proof of a memorized medical belief or clinical safety.',
        model_id='Qwen/Qwen3-4B-Base')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({k:protocol[k] for k in ['cases','pairs','families','rules']}))
if __name__=='__main__': main()

