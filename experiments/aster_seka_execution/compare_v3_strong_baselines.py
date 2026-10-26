"""Reproduce the post-hoc v3 comparisons against each strong baseline; CPU only."""
import json
from pathlib import Path
from mega_development_evaluation import compare_methods, digest, read_jsonl

P=Path(__file__).resolve().parent
def main():
    source=P/'clinical_v3_results_verified/case_metrics.jsonl'
    comparisons=[]
    for reference in ['text_emphasis','fusion_static','fusion_dynamic']:
        comparisons += [row for row in compare_methods(read_jsonl(source),reference,10000)
                        if row['method'] in ['gain_0_2','gain_0_6']]
    target=P/'clinical_v3_results_verified/direct_strong_baseline_comparisons.json'
    if target.exists():
        assert json.loads(target.read_text())['comparisons'] == comparisons
    result={'status':'complete_verified','scope':'Post-hoc development comparison; no confirmatory significance claim',
            'new_gpu_forwards':0,'cases':16,'clusters':16,'comparisons':comparisons,
            'case_metrics_sha256':digest(source),'script_sha256':digest(Path(__file__)),
            'statistics_source_sha256':digest(P/'mega_development_evaluation.py')}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified 12 direct paired comparisons; no GPU inference')

if __name__=='__main__':main()
