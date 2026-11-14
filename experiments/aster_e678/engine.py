"""Model-specific E1 extraction, attribution and immutable-prefix patching.

Imports use this checkout. No weights are loaded and no CUDA work is performed
until Engine is constructed. Every checkpoint has its own features and fits.
"""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration
from experiments.aster_cr.engine import Engine as CanonicalEngine, SYSTEM

MODELS = json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))


def prompt_policy(spec, tokenizer):
    """Use the checkpoint template when provided; explicitly handle base models."""
    if tokenizer.chat_template:
        return spec['chat_style']
    if spec.get('checkpoint_type') == 'base':
        return 'plain_completion'
    raise ValueError('Instruction checkpoint has no chat template: ' + spec['name'])


def render_prompt(spec, tokenizer, prompt, policy=None):
    policy = policy or prompt_policy(spec, tokenizer)
    if policy == 'plain_completion':
        return (tokenizer.bos_token or '') + SYSTEM + '\n\n' + prompt + '\n\nAnswer:'
    if policy == 'user_with_system_instruction':
        messages = [dict(role='user', content=SYSTEM + '\n\n' + prompt)]
    else:
        messages = [dict(role='system', content=SYSTEM), dict(role='user', content=prompt)]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


class TextBackbone(nn.Module):
    """Expose Gemma3's text decoder through the canonical head-intervention API."""
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone.get_decoder()
        self.lm_head = backbone.lm_head
        self.config = self.model.config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


class Engine(CanonicalEngine):
    def __init__(self, model_name, device='cuda:0', model=None, tokenizer=None, metadata_path=None):
        if model_name not in MODELS:
            raise ValueError('Unknown checkpoint: ' + model_name)
        self.model_key = model_name
        self.spec = MODELS[model_name]
        path = self.spec['path']
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.attention_implementation = 'eager' if cfg.model_type == 'gemma2' else 'sdpa'
        if model is None:
            loader = Gemma3ForConditionalGeneration if cfg.model_type == 'gemma3' else AutoModelForCausalLM
            model = loader.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                           device_map={'': device},
                                           attn_implementation=self.attention_implementation,
                                           low_cpu_mem_usage=True)
        self.original_model = model
        text = TextBackbone(model) if cfg.model_type == 'gemma3' else model
        self.text_config = getattr(cfg, 'text_config', cfg)
        self.softcap = getattr(self.text_config, 'final_logit_softcapping', None)
        self.chat_style = prompt_policy(self.spec, tokenizer)
        super().__init__(model_name, device=device, model=text, tokenizer=tokenizer)
        self.metadata = self.describe()
        if metadata_path:
            target = Path(metadata_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def render(self, prompt, task='binary'):
        return render_prompt(self.spec, self.tokenizer, prompt, self.chat_style)

    def _score_hidden(self, hidden):
        logits = torch.nn.functional.linear(hidden.float(), self._label_weights)
        if self.softcap is not None:
            logits = torch.tanh(logits / float(self.softcap)) * float(self.softcap)
        return logits[..., 0] - logits[..., 1]

    def run(self, context, *args, **kwargs):
        result = super().run(context, *args, **kwargs)
        # Canonical score() uses these optional label logits. Apply the official
        # softcap here once, just as _score_hidden does for the ordinary E1 score.
        if 'label_logits' in result and self.softcap is not None:
            logits = torch.as_tensor(result['label_logits'], dtype=torch.float32)
            result['label_logits'] = (torch.tanh(logits / float(self.softcap)) * float(self.softcap)).tolist()
        return result

    def describe(self):
        return dict(model=self.model_key, name=self.spec['name'], path=self.spec['path'],
                    checkpoint_type=self.spec['checkpoint_type'],
                    model_type=self.text_config.model_type,
                    num_layers=self.n_layers, num_attention_heads=self.n_heads,
                    total_heads=self.num_heads, head_dim=self.head_dim,
                    max_position_embeddings=self.limit,
                    dtype='bfloat16 backbone; float32 label projection',
                    prompt_policy=self.chat_style, chat_style=self.chat_style,
                    official_chat_template=self.tokenizer.chat_template,
                    plain_completion_suffix='Answer:' if self.chat_style == 'plain_completion' else None,
                    answer_labels=['\u662f', '\u5426'], answer_token_ids=self.label_ids,
                    score='official final logits(YES) - official final logits(NO)',
                    final_logit_softcapping=self.softcap,
                    attention_implementation=self.attention_implementation,
                    cache='native immutable prefix cache replay',
                    activation='last prompt token, self_attn.o_proj input, before output projection')
