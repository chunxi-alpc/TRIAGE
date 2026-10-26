"""Export anonymous clinical generation records; never infer clinical scores."""
import argparse, hashlib, json
from pathlib import Path
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def public_row(d):
    g=d.get("generation") or {}
    allowed=["input_tokens","output_tokens","highlighted_tokens","actual_forward_calls","elapsed_seconds","wall_seconds","peak_allocated_gib","peak_reserved_gib","generation_seed","greedy_decoding","no_support_fallback"]
    costs={k:g[k] for k in allowed if k in g and isinstance(g[k],(int,float,bool,type(None)))}
    hashes={k:v for k,v in g.items() if k.endswith("_sha256") and isinstance(v,str) and len(v)==64 and all(c in "0123456789abcdef" for c in v)}
    return {"sample_id_sha256":sha_bytes(str(d["sample_id"]).encode()),"guideline_sha256":d["guideline_sha256"],"method":d["method"],"status":d.get("status"),"prediction_sha256":sha_bytes(str(d.get("llm_output","")).encode()),"error_sha256":sha_bytes(str(d["error"]).encode()) if d.get("error") else None,"condition_rule_count":d.get("condition_rule_count"),"generation_seed":d.get("generation_seed"),"generation_measurements":costs,"input_fingerprints":hashes,"clinical_evaluation_status":"pending","clinical_score":None}
def main():
    a=argparse.ArgumentParser();a.add_argument("--run",type=Path,required=True);a.add_argument("--manifest",type=Path,required=True);a.add_argument("--output",type=Path,required=True);x=a.parse_args()
    manifest=read(x.manifest);protocol=read(x.run/"protocol.json")
    assert manifest["process_state"]=="exited" and manifest["exit_code"]==0
    outputs={};identity=None;source={}
    for method in protocol["arms"]:
        f=x.run/(method+".predictions.private.jsonl");rows=[json.loads(t) for t in f.read_text().splitlines() if t.strip()]
        assert len(rows)==16 and all(z["method"]==method for z in rows)
        ids=sorted(str(z["sample_id"]) for z in rows);assert len(set(ids))==16
        if identity is None:identity=ids
        assert ids==identity
        outputs[method]=[public_row(z) for z in rows];source[method]=sha_bytes(f.read_bytes())
    assert len(outputs)==11
    x.output.mkdir(exist_ok=False)
    for method,rows in outputs.items():(x.output/(method+".jsonl")).write_text("".join(json.dumps(z,ensure_ascii=False)+"\n" for z in rows))
    (x.output/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
    report={"status":"complete_verified","scope":"Generation completeness and anonymous export only; no clinical effectiveness claim","cases":16,"methods":11,"prediction_rows":176,"clinical_evaluation_status":"pending","raw_text_published":False,"raw_predictions_preserved_at":str(x.run),"raw_file_sha256":source,"run_manifest_sha256":sha_bytes(x.manifest.read_bytes()),"exporter_sha256":sha_bytes(Path(__file__).read_bytes()),"costs":{k:manifest.get(k) for k in ["forward_completed","elapsed_seconds","peak_allocated_gib","peak_reserved_gib"]}}
    (x.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"rows":176,"clinical_evaluation":"pending"}))
if __name__=="__main__":main()
