import sys,tempfile,json,time,argparse,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from batch_evaluate import evaluate_batch
from batch_judge import BatchJudge
class FakeEngine:
 def __init__(self):self.calls=0;self.active=None;self.batch_stats={};self.tok=self
 def applicability_many(self,cases,batch_size):return [({'r':1},'{}',None) for _ in cases]
 def task(self,c):return 'same prompt',{'r':[True]}
 def config(self,*x):return None
 def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
 def generate_many(self,texts,max_tokens,configs=None,sample=False,batch_size=4,on_result=None):
  result=[]
  for i,text in enumerate(texts):
   self.calls+=1
   if sample:raw='R1-C1: 5\nR1-C2: 1\nR2-C1: 1\nR2-C2: 5' if 'R1:' in text and 'C1:' in text else '5'
   else:raw='A'
   result.append(raw)
   if on_result:on_result(i,raw)
  return result
class BatchRecovery(unittest.TestCase):
 def test_old_prediction_cells_survive_resharding_and_batching(self):
  with tempfile.TemporaryDirectory(prefix='triage_batch_cells_') as d:
   root=Path(d);phase=root/'phase';phase.mkdir()
   cases=[{'case_id':str(i),'split':'heldout','group':'g','invalid_source':None,'patient':'test','rules':[{'id':'r'}],'gold':'A','gold_route':{'r':1},'dataset':'med'} for i in range(7)]
   (root/'med.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
   def row(i,m):return {'case_id':str(i),'method':m,'correct':True,'invalid_output':False,'technical_failure':False,'raw_output':'A','raw_routing':'{}','mcqa_prediction':'A','guideline_id':'g','dataset':'med','split':'heldout','intervention_applied':False}
   for worker,old in [(0,[row(0,'baseline'),row(0,'zero'),row(2,'baseline')]),(1,[row(1,'baseline'),row(1,'zero')])]:
    out=phase/('worker_'+str(worker));out.mkdir();(out/'med_heldout.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old))
   e=FakeEngine();artifact={'selection':{'strength':.1},'directions':{}}
   for worker in range(5):
    out=phase/('worker_'+str(worker));out.mkdir(exist_ok=True);a=argparse.Namespace(run=d,stage='full',worker=worker,world=5,started=time.time(),generation_batch_size=4,judge_batch_size=8)
    evaluate_batch(e,a,out,artifact,'med',['heldout'],methods=['baseline','zero'],tag='med_heldout')
    evaluate_batch(e,a,out,artifact,'med',['heldout'],methods=['baseline','zero'],tag='med_heldout')
   rs=[json.loads(x) for p in phase.glob('worker_*/*.jsonl') for x in p.read_text().splitlines()]
   self.assertEqual(len(rs),14);self.assertEqual(len({(r['case_id'],r['method']) for r in rs}),14);self.assertEqual(e.calls,9)
 def test_all_ten_judge_draws_cached_without_resampling(self):
  with tempfile.TemporaryDirectory(prefix='triage_batch_judge_') as d:
   e=FakeEngine();j=BatchJudge.__new__(BatchJudge);j.e=e
   j.refs={'train_val':{(2,1,'file'):{'filename':'file','language':'en','output_text':'["A", "B"]'}}}
   case={'case_id':'mega:train_val:2:1'};raw='### Clinical Decision Process:\n1. A\n2. B\n### Clinical Decision Outcome: B'
   first=j.score_many([(case,raw)],batch_size=8,cache_dirs=[Path(d)])
   second=j.score_many([(case,raw)],batch_size=8,cache_dirs=[Path(d)])
   self.assertEqual(e.calls,10);self.assertEqual(first,second);self.assertTrue(first[0][0]);self.assertEqual(first[0][2]['official_pcc'],0.)
   self.assertEqual(len(first[0][2]['raw_path_judge_responses']),5);self.assertEqual(len(first[0][2]['raw_accuracy_judge_responses']),5)
if __name__=='__main__':unittest.main()
