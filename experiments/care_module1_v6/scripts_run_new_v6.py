from pathlib import Path
import argparse,json,sys,time,traceback
ROOT=Path('/home/gcx/TRIAGE'); sys.path.insert(0,str(ROOT))
from src.module1.care_v6 import CAREModule1V6, TrialMatchAIAdapter, BioLinkBERTMedNLI

def load(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--inputs',required=True); p.add_argument('--rules',required=True); p.add_argument('--output',required=True); p.add_argument('--device',required=True); p.add_argument('--start',type=int,required=True); p.add_argument('--end',type=int,required=True); a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); inputs=load(a.inputs); rules={x['rule_id']:x for x in load(a.rules)}; rows=inputs[a.start:a.end]
 status={'device':a.device,'start':a.start,'end':a.end,'n':len(rows),'status':'starting','started':time.time()}; (out/f'status_{a.start}_{a.end}.json').write_text(json.dumps(status))
 try:
  import torch
  torch.cuda.set_device(int(a.device.split(':')[-1]))
  tm=TrialMatchAIAdapter.from_pretrained(a.device,cache_dir='/home/gcx/.cache/huggingface',base_model='/home/gcx/models/care_module1_v5/phi-4',adapter_model='/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora')
  nli=BioLinkBERTMedNLI.from_pretrained(a.device,cache_dir='/home/gcx/.cache/huggingface',model_name='/home/gcx/models/care_module1_v5/biolinkbert-mednli')
  engine=CAREModule1V6(tm,nli,nli_batch_size=32)
  with (out/f'pred_{a.start}_{a.end}.jsonl').open('w') as f:
   for i,sample in enumerate(rows,a.start):
    t=time.perf_counter(); graph=engine.run([rules[sample['rule_id']]],sample['sentences'],question=rules[sample['rule_id']]['text']); r=graph['rules'][0]
    public={'sample_id':sample['sample_id'],'rule_id':sample['rule_id'],'applicability':r['applicability'],'trialmatch_applicability':r['trialmatch_applicability'],'trialmatch_model_only_applicability':r['trialmatch_model_only_applicability'],'evidence_status':r['evidence_status'],'allow_attention_intervention':r['allow_attention_intervention'],'elapsed_seconds':time.perf_counter()-t,'conditions':[]}
    for c in r['conditions']+r['exceptions']:
     public['conditions'].append({'condition_id':c['condition_id'],'trialmatch_state':c['trialmatch']['state'],'trialmatch_raw_label':c['trialmatch'].get('raw_label'),'trialmatch_parse_status':c.get('generation_status'),'trialmatch_parse_error':c.get('technical_error') or c['trialmatch'].get('parse_error'),'numeric_status':(c.get('numeric') or {}).get('status','U'),'applicability':c['applicability'],'applicability_source':c['applicability_source'],'evidence_status':c['evidence_status'],'conflict':c['conflict'],'allow_attention_intervention':c['allow_attention_intervention'],'support_spans':[{'sentence_id':x['sentence_id'],'start':x['start'],'end':x['end'],'nli_relation':x['nli_relation']} for x in c['support_evidence']],'refute_spans':[{'sentence_id':x['sentence_id'],'start':x['start'],'end':x['end'],'nli_relation':x['nli_relation']} for x in c['refute_evidence']],'numeric_evidence_spans':[{'sentence_id':x['sentence_id'],'start':x['start'],'end':x['end']} for x in (c.get('numeric') or {}).get('evidence',[])], 'trialmatch_seconds':c['trialmatch_seconds'],'numeric_seconds':c['numeric_seconds'],'mednli_seconds':c['mednli_seconds']})
    f.write(json.dumps(public,ensure_ascii=False)+'\n'); f.flush()
  status.update(status='completed',elapsed_seconds=time.time()-status['started'],peak_gib=round(torch.cuda.max_memory_allocated()/1024**3,2))
 except Exception as e:
  status.update(status='failed',error=repr(e),traceback=traceback.format_exc())
  raise
 finally: (out/f'status_{a.start}_{a.end}.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
