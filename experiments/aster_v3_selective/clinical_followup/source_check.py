"""Compare public guideline and template metadata with V3 exclusions. Open no cases or labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import unicodedata

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GUIDELINES = {
    'headache': ('Guideline for primary care management of headache in adults',
                 'merged_decision_tree.json', 'generate_vignette.py', 'headache'),
    'low_back_pain': ('Evidence-informed primary care management of low back pain: clinical practice guideline',
                      'low_back_pain/decision_tree_lower_back_pain.json', 'low_back_pain/vignette_generator.py', 'lowbackpain'),
    'prostate': ('Guideline for referral of patients with suspected prostate cancer by family physicians and other primary care providers',
                 'prostate_cancer/decision_trees_output_pc2.json', 'prostate_cancer/generate_vignettes_pc.py','prostate')
}
CPG_REVISION = 'aa6061e2f6b2e810f7c5897232f45ab223a956a2'


def normalized(text):
    return ''.join(ch for ch in unicodedata.normalize('NFKC',text).casefold() if ch.isalnum())


def strings(value):
    if isinstance(value,str):
        yield value
    elif isinstance(value,list):
        for v in value:
            yield from strings(v)
    elif isinstance(value,dict):
        for k,v in value.items():
            yield k
            yield from strings(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path.home()/'aster-v3-selective-20261124'/'reports/aster_v3_selective_20261124')
    parser.add_argument('--clusters',type=Path,default=Path.home()/'aster_manuscript_20261030/clinical01/source_clusters.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[3]/'reports/aster_v3_selective_20261124/followup/sources/source_check.json')
    args = parser.parse_args()
    t0,cpu=time.perf_counter(),time.process_time()
    with (args.archive/'s4/prior_sources.csv').open() as f:
        previous=list(csv.DictReader(f))
    families={r['family'] for r in previous}
    registry=json.loads(args.clusters.read_text())['records']
    used={k:v for k,v in registry.items() if v['cluster'] in families}
    rules=json.loads((args.archive/'s1/s3_protocol.json').read_text())['allowed_rules']
    results=[]
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=3,backoff_factor=0.5,status_forcelist=(429,500,502,503,504))))
        for domain,(title,path,generator,keyword) in GUIDELINES.items():
            base=f'https://raw.githubusercontent.com/bionlplab/CPGPrompt/{CPG_REVISION}/'
            response=session.get(base+path,timeout=25)
            response.raise_for_status()
            tree=response.json()
            # The official assets are tree schemas; patient vignettes, scoring outputs and reference responses stay unopened.
            fields=list(strings(tree))
            exact=sorted({r['cluster'] for r in used.values() if normalized(title)==normalized(r['title'])})
            contained=sorted({r['cluster'] for r in used.values() if normalized(title) in normalized(r['title'])})
            topic=sorted({r['cluster'] for r in used.values() if keyword in normalized(r['title'])})
            results.append(dict(dataset='CPGPrompt',domain=domain,source_title=title,tree_url=base+path,
                                template_implementation=base+generator,
                                exact_title_cluster_ids=exact,contains_full_title_cluster_ids=contained,
                                related_topic_english_title_cluster_ids=topic,
                                frozen_rule_literal_matches=[q for q in rules if any(q in field for field in fields)]))
    prior_cancer={k:sorted({r['source'] for r in previous if r['family']=='MedGUIDE:'+k}) for k in ('nscl','prostate')}
    output=dict(mega_title_records=len(used),mega_distinct_used_clusters=len({r['cluster'] for r in used.values()}),
                medguide_previously_used_trees=prior_cancer,cpgprompt=results,
                eligibility=dict(cpgprompt_qualified_families=0,cpgprompt_family_provenance_pending=3,
                                 cancerGUIDE_family_overlap=1,prospective_evaluations=0),
                data_opened='prior study identifiers, guideline titles and three public tree-only metadata files',
                patient_cases_downloaded=0, test_labels_opened=0, model_calls=0,gpu_hours=0,
                cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-t0)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    main()
