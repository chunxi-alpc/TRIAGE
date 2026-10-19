"""Offline model identity/config check before GPU use; no inference or fallback."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys, time

EXPECTED_BASE = "microsoft/phi-4"
EXPECTED_ADAPTER = "majdabd33/trialmatchai-phi4-reasoning-lora"
EXPECTED_NLI = "cnut1648/biolinkbert-mednli"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base", type=pathlib.Path, required=True)
    p.add_argument("--adapter", type=pathlib.Path, required=True)
    p.add_argument("--evidence", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a=p.parse_args(); start=time.perf_counter()
    result={"status":"checking","repository_ids":{"base":EXPECTED_BASE,"adapter":EXPECTED_ADAPTER,"evidence":EXPECTED_NLI},"models":{}}
    try:
        from peft import PeftConfig
        from transformers import AutoConfig
        adapter=PeftConfig.from_pretrained(str(a.adapter))
        base_name=getattr(adapter,"base_model_name_or_path",None)
        if base_name != EXPECTED_BASE:
            raise RuntimeError(f"adapter base mismatch: expected {EXPECTED_BASE}, got {base_name!r}")
        if not (a.adapter/"adapter_model.safetensors").is_file() and not (a.adapter/"adapter_model.bin").is_file():
            raise FileNotFoundError("specified TrialMatchAI adapter weights not found")
        base=AutoConfig.from_pretrained(str(a.base))
        if base.model_type != "phi3" or "Phi3ForCausalLM" not in (getattr(base, "architectures", None) or []):
            raise RuntimeError(f"unexpected microsoft/phi-4 architecture: model_type={base.model_type!r}, architectures={getattr(base, 'architectures', None)!r}")
        nli=AutoConfig.from_pretrained(str(a.evidence))
        id2label={int(k):str(v).casefold() for k,v in nli.id2label.items()}
        if set(id2label.values()) != {"entailment","neutral","contradiction"}:
            raise RuntimeError(f"MedNLI config labels not as expected: {id2label}")
        if not any((a.base/name).is_file() for name in ("model.safetensors.index.json","model.safetensors","pytorch_model.bin")):
            raise FileNotFoundError("Phi-4 base weights are missing")
        index_path=a.base/"model.safetensors.index.json"
        if index_path.is_file():
            index=json.loads(index_path.read_text())
            missing=[name for name in set(index["weight_map"].values()) if not (a.base/name).is_file()]
            if missing:
                raise FileNotFoundError(f"Phi-4 weight shards are missing: {missing}")
        if not any((a.evidence/name).is_file() for name in ("model.safetensors","pytorch_model.bin")):
            raise FileNotFoundError("BioLinkBERT-MedNLI weights are missing")
        result["models"]={"base":{"path":str(a.base),"model_type":base.model_type,
                "architectures":getattr(base,"architectures",None),"parameter_layers":getattr(base,"num_hidden_layers",None)},
            "adapter":{"path":str(a.adapter),"base_model_name_or_path":base_name,
                "peft_type":str(getattr(adapter,"peft_type",None)),"task_type":str(getattr(adapter,"task_type",None))},
            "evidence":{"path":str(a.evidence),"model_type":nli.model_type,
                "architectures":getattr(nli,"architectures",None),"id2label":{str(k):v for k,v in id2label.items()},
                "max_position_embeddings":getattr(nli,"max_position_embeddings",None)}}
        try:
            from huggingface_hub import HfApi
            api=HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
            result["revisions"]={"base":api.model_info(EXPECTED_BASE).sha,
                "adapter":api.model_info(EXPECTED_ADAPTER).sha,"evidence":api.model_info(EXPECTED_NLI).sha}
        except Exception as exc:
            result["revision_lookup_error"]=f"{type(exc).__name__}: {str(exc)[:300]}"
        result["weight_files"]={}
        for name,root in (("base",a.base),("adapter",a.adapter),("evidence",a.evidence)):
            files=[]
            for f in sorted(root.iterdir()):
                if f.is_file() and (f.suffix in {".safetensors",".bin"} or f.name.endswith(".index.json")):
                    files.append({"name":f.name,"bytes":f.stat().st_size,"sha256":sha256(f)})
            result["weight_files"][name]=files
        result["status"]="passed"
    except Exception as exc:
        result["status"]="failed"; result["error_type"]=type(exc).__name__; result["error"]=str(exc)
    result["elapsed_seconds"]=round(time.perf_counter()-start,3)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":result["status"],"id2label":result.get("models",{}).get("evidence",{}).get("id2label"),"elapsed_seconds":result["elapsed_seconds"],"error":result.get("error")},ensure_ascii=False))
    if result["status"] != "passed": raise SystemExit(2)
if __name__=="__main__": main()
