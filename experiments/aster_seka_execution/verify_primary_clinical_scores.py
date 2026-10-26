"""Server CPU verification/export of finished primary clinical scores; no inference."""
from pathlib import Path
import json, hashlib, math, sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from mega_development_evaluation import load_official,native_metrics,compare_methods,parse_with_official
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    runtime=P/"runtime"; e=runtime/"mega_dev16_primary_gpu016_evaluation"
    out=P/"mega_dev16_primary_scores_verified"
    cfg=read(e/"evaluation_config.json");report=read(e/"evaluation_report.json")
    manifest=read(runtime/"mega_dev16_primary_gpu016_evaluation_job/run_manifest.json")
    assert report["status"]=="complete_verified" and manifest["exit_code"]==0 and manifest["process_state"]=="exited"
    official,source_hashes=load_official(runtime/"vendors/MEGA-CDP-ssh/mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py")
    assert source_hashes==cfg["official_source_files_sha256"]
    frozen=read(P/"mega_dev16_input_lock.json")
    expected={x["sample_id"]:x["source_cluster"] for x in frozen["selected"]}
    assert len(expected)==16 and len(cfg["method_order"])==11
    saved=[json.loads(x) for x in (e/"case_metrics.jsonl").read_text().splitlines()]
    lookup={(x["method"],x["sample_id"]):x for x in saved}
    assert len(lookup)==len(saved)==176
    rows=[];files={};summaries={}
    for method in cfg["method_order"]:
        pred=runtime/"mega_dev16_seed0_gpu01"/(method+".predictions.private.jsonl")
        assert sha(pred)==cfg["prediction_sha256"][method]
        predictions={x["sample_id"]:x for x in map(json.loads,pred.read_text().splitlines())}
        native_file=e/"private"/(method+"_official.jsonl")
        natives=[json.loads(x) for x in native_file.read_text().splitlines()]
        assert len(natives)==16 and {x["sample_id"] for x in natives}==set(expected)
        files[method]=sha(native_file)
        for x in natives:
            sid=x["sample_id"];old=lookup[(method,sid)]
            parsed,_=parse_with_official(official["parser"],predictions[sid])
            assert parsed==x["parsed"]
            values=native_metrics(x["pathway"],x["accuracy"],official["summary"],official["metrics"])
            for metric in ["pcc_dtw_cost","acc_normalized"]:
                assert math.isclose(values[metric],old[metric],rel_tol=0,abs_tol=1e-12)
            assert old["tree_cluster"]==expected[sid]
            rows.append({k:old[k] for k in ["sample_id","tree_cluster","method","method_seed","parsed_success",
                "pcc_dtw_cost","acc_normalized","pathway_valid","accuracy_valid","pathway_judge_results","accuracy_judge_results"]})
        subset=[x for x in rows if x["method"]==method]
        means={"PCC_DTW_cost_lower_better":math.fsum(x["pcc_dtw_cost"] for x in subset)/16,
               "Acc_normalized_1_to_5_higher_better":math.fsum(x["acc_normalized"] for x in subset)/16}
        for k,v in means.items():assert math.isclose(v,report["methods"][method][k],rel_tol=0,abs_tol=1e-12)
        summaries[method]={**report["methods"][method],**means}
    comparisons=[]
    for ref in ["A1B0","A1B1_zero","A0B0"]:
        comparisons+=compare_methods(rows,ref,10000)
    out.mkdir(exist_ok=False)
    (out/"case_metrics.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
    audit={"status":"complete_verified","scope":"All176 native scores independently recomputed from saved official records and matched original predictions; development only",
        "new_gpu_forwards":0,"cases":16,"methods":summaries,"comparisons":comparisons,
        "clinical_accuracy_is_binary":False,"independent_model_claim":False,
        "judge_limit":"Local Qwen judge shares model family/checkpoint with generator; not the paper's judge",
        "raw_text_published":False,"raw_native_file_sha256":files,"evaluation_config_sha256":sha(e/"evaluation_config.json"),
        "run_manifest_sha256":sha(runtime/"mega_dev16_primary_gpu016_evaluation_job/run_manifest.json"),
        "verifier_sha256":sha(__file__),"judge_request_costs":report["judge_request_costs"],
        "judge_process_costs":report["judge_process_costs"],"elapsed_seconds":report["elapsed_seconds"],
        "interpretation":"Current main A1B1 is worse than A1B0 and matched zero; no independent attention benefit demonstrated. Text emphasis has best point estimates among these11 only; stronger fusion comparisons are unfinished."}
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (out/"evaluation_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    print(json.dumps({"status":"complete_verified","rows":176,"main_comparisons":[x for x in comparisons if x["method"]=="A1B1" and x["reference"] in ["A1B0","A1B1_zero"]]}))
if __name__=="__main__":main()
