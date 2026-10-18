"""Unscored model identity/load/interface check; no benchmark labels or tuning."""
from __future__ import annotations
import argparse,json,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.module1.care_v5 import TrialMatchAIAdapter,BioLinkBERTMedNLI

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:6');p.add_argument('--output',required=True);a=p.parse_args()
 result={'status':'starting','unscored':True,'device':a.device,'patients_used':'8 identical invented age facts; no benchmark cases or labels'}
 start=time.perf_counter()
 try:
  import torch
  torch.cuda.set_device(a.device);torch.cuda.reset_peak_memory_stats(a.device)
  tm=TrialMatchAIAdapter.from_pretrained(a.device,base_model='/home/gcx/models/care_module1_v5/phi-4',adapter_model='/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora')
  nli=BioLinkBERTMedNLI.from_pretrained(a.device,model_name='/home/gcx/models/care_module1_v5/biolinkbert-mednli')
  result['load_seconds']=time.perf_counter()-start
  patient='The patient is a 45-year-old woman. Her age is explicitly documented as 45 years.'
  condition={'text':'The patient is at least 18 years old.'}
  rows=tm.judge_many([{'patient':patient,'condition':condition,'criterion_kind':'inclusion'} for _ in range(8)],batch_size=8)
  result['trialmatch']=[{k:r.get(k) for k in ('state','raw_label','parse_error','truncated','input_tokens','output_tokens','inference_seconds','batch_elapsed_seconds')} for r in rows]
  result['mednli']=nli.predict([(patient,condition['text'])])
  result['peak_allocated_gib']=round(torch.cuda.max_memory_allocated(a.device)/1024**3,3)
  result['peak_reserved_gib']=round(torch.cuda.max_memory_reserved(a.device)/1024**3,3)
  result['status']='completed'
 except Exception as exc:
  result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
 result['seconds']=round(time.perf_counter()-start,3)
 pathlib.Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))
 if result['status']!='completed': raise SystemExit(2)
if __name__=='__main__':main()
