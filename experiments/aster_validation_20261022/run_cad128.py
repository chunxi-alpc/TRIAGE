"""Same main protocol, with CAD128 from the outset to avoid a redundant CAD8 pass."""
from delegate_pending import wait_if_delegated
if __name__=='__main__':wait_if_delegated()
from pathlib import Path
p=Path(__file__).with_name('run.py');s=p.read_text()
assert s.count('for step in range(8):')==1
s=s.replace('for step in range(8):','for step in range(128):')
s=s.replace('if token==tok.eos_token_id:break',"if token in (model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list) else [model.generation_config.eos_token_id or tok.eos_token_id]):break")
s=s.replace('result.update(completion=text,parsed_prediction=parse_answer(text,options))\n  return result',"result.update(completion=text,parsed_prediction=parse_answer(text,options),generated_tokens=len(generated),max_new_tokens=128,stop_reason='length' if len(generated)==128 else 'eos')\n  return result")
s=s.replace('greedy free generation for all main conditions, 8 tokens; separate first-position option scores','greedy free generation; CAD/prune_cad 128 tokens, others 8; separate first-position option scores')
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
