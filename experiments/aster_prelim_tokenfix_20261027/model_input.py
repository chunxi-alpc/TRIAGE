"""Chat templates already serialize special tokens; never add a second BOS."""
def encode_prompts(tokenizer,texts,**kwargs):
 assert 'add_special_tokens' not in kwargs
 return tokenizer(texts,add_special_tokens=False,**kwargs)
