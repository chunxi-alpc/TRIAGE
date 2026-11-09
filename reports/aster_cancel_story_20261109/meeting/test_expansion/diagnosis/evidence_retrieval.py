
"""Generic lexical retrieval of patient sentences, without labels or scores."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
STOP={'the','a','an','is','are','was','were','be','been','does','do','did','has','have','had','of','to','in','at','for','with','and','or','there','it','its','that','this','such','as','by','on','from','than','patient','individual','person','whether','currently'}
def sentences(text):
 parts=[s.strip() for s in re.split(r'(?<=[。！？!?；;])|\n+|(?<=[.])\s+(?=[A-Z])',text) if s.strip()]
 return parts or [text]
def terms(text):
 text=text.lower().replace('\\u202f',' ').replace('\u202f',' ')
 words=re.findall(r'[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)?',text)
 words=[s for s in words if s not in STOP]
 result=words+['bi:'+words[j]+'_'+words[j+1] for j in range(len(words)-1)]
 for run in re.findall(r'[\u4e00-\u9fff]+',text):result.extend('zh:'+run[j:j+2] for j in range(max(1,len(run)-1)))
 return result
def retrieve(condition,patient,topk=1):
 """Copy highest lexical-match patient sentences; never emit a judgment."""
 candidates=sentences(patient)
 vectorizer=TfidfVectorizer(analyzer=terms,norm='l2',sublinear_tf=True)
 try:
  matrix=vectorizer.fit_transform(candidates+[condition])
  scores=np.asarray((matrix[:-1]@matrix[-1].T).toarray()).reshape(-1)
 except ValueError:
  scores=np.zeros(len(candidates))
 order=sorted(range(len(candidates)),key=lambda j:(-scores[j],j))[:topk]
 return [dict(rank=k+1,sentence_index=j,quote=candidates[j],matching_score=float(scores[j])) for k,j in enumerate(order)]
def parse_original_prompt(original):
 """Return exact current-condition and patient portions of the canonical prompt."""
 patient_marker='病历：\n';node_marker='\n\n当前节点：';condition_marker='\n当前条件：'
 if patient_marker not in original or node_marker not in original or condition_marker not in original:return None,None
 patient=original.split(patient_marker,1)[1].split(node_marker,1)[0]
 condition=original.split(condition_marker,1)[1].split('\n请判断',1)[0]
 return condition,patient

