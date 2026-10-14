#!/usr/bin/env python3
"""One original JUICE inference utility call using an existing Qwen model.
This does not run JUICE attention intervention. No dataset is generated.
"""
import argparse,json,sys,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);args=parser.parse_args()
sys.path.insert(0,str(ROOT/'third_party/JUICE/src'))
from model_utils import get_first_few_word
path=ROOT/'third_party/JUICE/dataset/factual/world_capital_gemma/head_size_4/unfiltered_test.json'
sample=json.loads(path.read_text())[0]
tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float16,attn_implementation='eager').to('cuda:0').eval()
model.generation_config.temperature=None;model.generation_config.top_p=None
prediction=get_first_few_word(sample['Clean Prompt'],tok,model,max_new_tokens=8,num_keep=5)
result={'test':'original JUICE model_utils inference utility; no attention intervention','model':args.model,'question':sample['Clean Prompt'],'reference_answer':sample['Answer'],'prediction':prediction,'gpu_peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
(ROOT/'results/juice_inference_smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
