"""Original repair-v2 action_guides, copied verbatim for standalone reproduction."""
import copy,re
from experiments.aster_closed_form.data import canonical_guide
def action_guides(g):
 leaves_=canonical_guide(g['guideline'])['leaves'];conditions='\n'.join(n['condition'] for n in canonical_guide(g['guideline'])['nodes'])
 protected={k for k,v in leaves_.items() if re.search(r'定义|指的是|是指|defined as|definition|means that|criteria (?:are|include)',v,re.I) or re.search(r'\b'+k+r'\b',conditions)}
 editable=[k for k in leaves_ if k not in protected]
 maps={'presence':{k:'[处置已删除]' for k in editable},'neutral':{k:('记录内容。' if g['language']=='zh' else 'Administrative information.') for k in editable}}
 if len({leaves_[k] for k in editable})>1:maps['swap']={k:leaves_[editable[(i+1)%len(editable)]] for i,k in enumerate(editable)}
 for mode,mapping in maps.items():
  guide=copy.deepcopy(g['guideline'])
  for node in guide['groups']:
   for rule in node['rules']:
    m=re.fullmatch(r'(L\d+)([：:])(.*)',rule['action'],re.S)
    if m and m[1] in mapping:rule['action']=m[1]+m[2]+mapping[m[1]]
  if guide!=g['guideline']:yield mode,guide,sorted(protected)
