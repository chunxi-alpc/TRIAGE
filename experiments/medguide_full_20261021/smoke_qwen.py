import json
from pathlib import Path
from qwen_run import Engine,P
c=json.loads((P/'private/input_0.jsonl').read_text().splitlines()[0])
e=Engine();none=e.pair(c,{})
assert none['arms']['baseline']['candidate_logits']==none['arms']['a1_attention']['candidate_logits']
# Interface check only: artificial signs are not benchmark predictions.
test=e.pair(c,{c['rules'][0]['rule_id']:'T',c['rules'][1]['rule_id']:'F'})
assert any(d['attention_absolute_change_mean']>0 for d in test['arms']['a1_attention']['attention_diagnostics'])
(P/'qwen_validation.json').write_text(json.dumps({'status':'passed','zero_signal_exact_logits':True,'signed_attention_changed':True,'options':len(c['options']),'input_tokens':test['prompt_tokens'],'provenance':e.provenance,'purpose':'technical interface only, artificial signs not used for benchmark'},indent=2))
print('qwen validation passed')
