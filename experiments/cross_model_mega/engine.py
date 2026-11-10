"""Five manuscript backbones under the existing native-cache head intervention API.
No GPU is allocated on import. Each Engine loads one BF16 backbone on its device.
"""
import json,sys,time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,Gemma3ForConditionalGeneration
REPO=str(Path(__file__).resolve().parents[2])
if REPO not in sys.path:sys.path.insert(0,REPO)
from experiments.aster_cr.engine import Engine as CanonicalEngine,SYSTEM
ROOT=Path(__file__).resolve().parent
MODELS=json.loads((ROOT/'models.json').read_text())

class TextBackbone(nn.Module):
 """Expose Gemma3's text decoder with the interface used by the existing engine."""
 def __init__(self,backbone):
  super().__init__()
  self.model=backbone.get_decoder()
  self.lm_head=backbone.lm_head
  self.config=self.model.config
 def get_input_embeddings(self):return self.model.get_input_embeddings()

class Engine(CanonicalEngine):
 def __init__(self,model_name,device='cuda:0',model=None,tokenizer=None,metadata_path=None):
  if model_name not in MODELS:raise ValueError('Unknown manuscript model: '+model_name)
  self.model_key=model_name;self.spec=MODELS[model_name]
  path=self.spec['path'];cfg=AutoConfig.from_pretrained(path,local_files_only=True)
  tokenizer=tokenizer or AutoTokenizer.from_pretrained(path,local_files_only=True)
  if model is None:
   loader=Gemma3ForConditionalGeneration if cfg.model_type=='gemma3' else AutoModelForCausalLM
   model=loader.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
       device_map={'':device},attn_implementation='eager' if cfg.model_type=='gemma2' else 'sdpa',low_cpu_mem_usage=True)
  self.original_model=model
  self.attention_implementation='eager' if cfg.model_type=='gemma2' else 'sdpa'
  text=TextBackbone(model) if cfg.model_type=='gemma3' else model
  self.text_config=getattr(cfg,'text_config',cfg)
  self.softcap=getattr(self.text_config,'final_logit_softcapping',None)
  self.chat_style='user_with_system_instruction' if model_name in ('gemma2','gemma3') else 'system_user'
  super().__init__(model_name,device=device,model=text,tokenizer=tokenizer)
  self.metadata=self.describe()
  if metadata_path:
   target=Path(metadata_path);target.parent.mkdir(parents=True,exist_ok=True)
   target.write_text(json.dumps(self.metadata,ensure_ascii=False,indent=2)+'\n')
 def render(self,prompt,task='binary'):
  if self.chat_style=='user_with_system_instruction':messages=[dict(role='user',content=SYSTEM+'\n\n'+prompt)]
  else:messages=[dict(role='system',content=SYSTEM),dict(role='user',content=prompt)]
  return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def _score_hidden(self,hidden):
  logits=torch.nn.functional.linear(hidden.float(),self._label_weights)
  if self.softcap is not None:logits=torch.tanh(logits/float(self.softcap))*float(self.softcap)
  return logits[...,0]-logits[...,1]
 def score(self,prompt,labels=None,capture=True,replacements=None,grad=False):
  result=super().score(prompt,labels=labels,capture=capture,replacements=replacements,grad=grad)
  if labels and self.softcap is not None:
   logits=torch.as_tensor(result['label_logits'],dtype=torch.float32)
   logits=torch.tanh(logits/float(self.softcap))*float(self.softcap)
   result['label_logits']=logits.tolist();result['ell']=float(logits[0]-logits[1]);result['prediction']=int(result['ell']>0)
  return result
 def describe(self):
  return dict(model=self.model_key,name=self.spec['name'],path=self.spec['path'],
      model_type=self.text_config.model_type,num_layers=self.n_layers,num_attention_heads=self.n_heads,
      total_heads=self.num_heads,head_dim=self.head_dim,max_position_embeddings=self.limit,
      dtype='bfloat16 backbone; float32 label projection',chat_style=self.chat_style,
      official_chat_template=self.tokenizer.chat_template,
      answer_labels=['\u662f','\u5426'],answer_token_ids=self.label_ids,
      score='official final logits(YES) - official final logits(NO)',
      final_logit_softcapping=self.softcap,attention_implementation=self.attention_implementation,cache='native immutable prefix cache replay',
      activation='last prompt token, self_attn.o_proj input, before output projection')
