from collections import defaultdict
from common import *
from fusion import fit,evaluate
def main():
 inputs={r['score_id']:r for r in rows(ROOT/'data/medpair_inputs.jsonl')};scores={}
 for path in sorted((ROOT/'results').glob('medpair_scores_*.jsonl')):
  for r in rows(path):
   if r['status']!='complete':continue
   if r['score_id'] in scores:assert scores[r['score_id']]['raw_logits']==r['raw_logits']
   scores[r['score_id']]=r
 assert set(inputs)==set(scores),(len(inputs),len(scores))
 views=rows(ROOT/'data/medpair_views.jsonl');joined=[]
 for v in views:
  s=scores[v['score_id']];assert s['labels']==v['labels'] and s['gold']==v['gold']
  assert abs(sum(np.exp(s['log_probabilities']))-1)<1e-5
  joined.append(dict(v,raw_logits=s['raw_logits'],log_probabilities=s['log_probabilities'],vocabulary_log_probabilities=s['vocabulary_log_probabilities'],prediction=s['prediction'],correct=s['correct'],
   input_tokens=inputs[v['score_id']]['input_tokens'],prompt=inputs[v['score_id']]['prompt']))
 write_rows(ROOT/'results/medpair_complete_views.jsonl',joined)
 by={(r['id'],r['condition']):r for r in joined};counts=[];comp=[];per_pair=[]
 for split in ('dev','test'):
  for view in ('Full','High','C','C+h','C+u'):
   rr=[r for r in joined if r['split']==split and r['condition']==view]
   counts.append(dict(split=split,view=view,n=len(rr),accuracy=rate([r['correct'] for r in rr]),gold_logp=mean_ci([r['log_probabilities'][r['labels'].index(r['gold'])] for r in rr])))
  for first,second in [('Full','High'),('C+h','C'),('C+h','C+u'),('C+u','C')]:
   rr=[]
   for a in joined:
    if a['split']!=split or a['condition']!=first or (a['id'],second) not in by:continue
    b=by[a['id'],second];ix=a['labels'].index(a['gold'])
    row=dict(id=a['id'],split=split,first=first,second=second,correct_first=a['correct'],correct_second=b['correct'],accuracy_difference=int(a['correct'])-int(b['correct']),
     gold_logp_difference=a['log_probabilities'][ix]-b['log_probabilities'][ix],first_correct_second_wrong=a['correct'] and not b['correct'],
     token_difference=a['input_tokens']-b['input_tokens']);rr.append(row);per_pair.append(row)
   comp.append(dict(split=split,first=first,second=second,n=len(rr),accuracy_difference=mean_ci([r['accuracy_difference'] for r in rr]),gold_logp_difference=mean_ci([r['gold_logp_difference'] for r in rr]),
    first_correct_second_wrong=rate([r['first_correct_second_wrong'] for r in rr]),token_difference=mean_ci([r['token_difference'] for r in rr])))
 data=[]
 for r in joined:
  if r['condition']!='Full':continue
  m=by[r['id'],'High'];data.append(dict(id=r['id'],split=r['split'],variant='Full/High',labels=r['labels'],gold_idx=r['labels'].index(r['gold']),f=r['log_probabilities'],m=m['log_probabilities']))
 assert len([r for r in data if r['split']=='dev'])==91
 assert len([r for r in data if r['split']=='test'])==89
 cal=fit([r for r in data if r['split']=='dev'])
 fusion=[];fused=[]
 for method,param in dict(Full=None,Masked=None,**cal['parameters']).items():
  result,rr=evaluate([r for r in data if r['split']=='test'],method,param)
  fusion.append(dict(dataset='MedPAIR',variant='Full/High',method=method,parameter=param,**result));fused.extend(rr)
 pairs=rows(ROOT/'data/medpair_addback_pairs.jsonl');lengths=[]
 for split in ('dev','test'):
  rr=[r for r in pairs if r['split']==split];g=[r['absolute_token_gap'] for r in rr]
  lengths.append(dict(split=split,n=len(rr),absolute_token_gap_min=min(g),absolute_token_gap_median=float(np.median(g)),absolute_token_gap_mean=float(np.mean(g)),absolute_token_gap_max=max(g),h_token_mean=float(np.mean([r['h_tokens'] for r in rr])),u_token_mean=float(np.mean([r['u_tokens'] for r in rr]))))
 write_json(ROOT/'results/medpair_new_accuracy.json',counts);write_json(ROOT/'results/medpair_new_comparisons.json',comp)
 write_json(ROOT/'results/medpair_addback_lengths.json',lengths);write_rows(ROOT/'results/medpair_new_pairs.jsonl',per_pair)
 write_json(ROOT/'results/medpair_fusion_parameters.json',cal);write_json(ROOT/'results/medpair_fusion.json',fusion);write_rows(ROOT/'results/medpair_fusion_predictions.jsonl',fused)
 write_json(ROOT/'results/medpair_completion.json',dict(status='complete',views=len(views),unique_scores=len(scores),reused_identical=len(views)-len(scores),pilot_included=True,dev_cases=91,test_cases=89,addback_dev=31,addback_test=29))
if __name__=='__main__':main()
