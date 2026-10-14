#!/usr/bin/env python3
"""Non-test, limited real Qwen3 head-localization run; no training or downloads."""
import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module2 import build_cached_calibration_pairs, locate_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(ROOT/'models/modelscope/Qwen3-8B'))
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--output', default=str(ROOT/'results/module2/components.json'))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = build_cached_calibration_pairs(limit=args.pairs)
    (output.parent/'calibration_pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:0').eval()
    torch.cuda.reset_peak_memory_stats()
    result = locate_heads(model, tokenizer, pairs)
    result['metadata']['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    result['metadata']['gpu_peak_reserved_gib'] = torch.cuda.max_memory_reserved()/1024**3
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'selected': len(result['selected_heads']),
        'candidate_count': len(result['candidate_heads']), 'metadata': result['metadata']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
