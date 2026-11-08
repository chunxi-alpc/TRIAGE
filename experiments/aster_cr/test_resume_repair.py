import numpy as np
from .common import TASK_KINDS,lines,write,read

def test_repaired_f_endpoints_are_in_training_and_evaluation():
 from .readout import TRAIN_KINDS
 assert {"F","F0"}<=TRAIN_KINDS==TASK_KINDS

def test_original_denominator_and_nested_grid(tmp_path,monkeypatch):
 from . import diagnostics
 views=[dict(view_id="a:full",group_id="a",source_id="s",split="dev",variant="full",original=True,label=1),
        dict(view_id="b:full",group_id="b",source_id="s",split="dev",variant="full",original=False,label=0)]
 for i in range(3):views.append(dict(view_id=f"a:F{i}",group_id="a",source_id="s",split="dev",variant=f"F{i}",original=False,label=int(i>=1),field="age",value=64+i,grid=dict(threshold=65,step=1)))
 class Fake:
  def __init__(self,model,splits):self.views=views;self.pairs=[]
  def ell(self,vid):return {"a:full":1.,"b:full":1.,"a:F0":-1.,"a:F1":.1,"a:F2":1.}[vid]
 monkeypatch.setattr(diagnostics,"ROOT",tmp_path);monkeypatch.setattr(diagnostics,"Features",Fake)
 diagnostics.report("qwen",("dev",))
 result=read(tmp_path/"qwen/diagnostics/development/p0.json")
 assert result["dev"]["original_nodes"]==1 and result["additional_candidates"]["dev"]["nodes"]==1
 from .common import rows
 assert len(rows(tmp_path/"qwen/diagnostics/development/thresholds.jsonl"))==1

def test_two_gpu_job_coverage(monkeypatch):
 from . import pipeline
 monkeypatch.setattr(pipeline,"GPUS",[0,1]);calls=[]
 jobs=pipeline.shard_jobs("experiments.aster_cr.score","test")
 pipeline.run_jobs(lambda name,jobs:calls.extend(jobs),"test",jobs)
 assert [gpu for gpu,args in calls]==[0,1]
 assert {args[args.index("--shard")+1] for gpu,args in calls}=={"0","1"}
 assert all(args[args.index("--shards")+1]=="2" for gpu,args in calls)

def test_external_native_label_parsing():
 from .external_score import parse_answer
 assert parse_answer("nli4ct","Entailment")=="Entailment"
 assert parse_answer("nli4ct","Entailment or Contradiction") is None
 assert parse_answer("medpic",'["A", "C"]')==["A","C"]
 assert parse_answer("medpic","A") is None

def test_test_cache_waits_for_new_freeze(tmp_path,monkeypatch):
 import pytest
 from . import resume_repair
 monkeypatch.setattr(resume_repair,"ROOT",tmp_path)
 lines(tmp_path/"qwen/pending_test_scores/test_0.jsonl",[dict(view_id="x",ell=1)])
 with pytest.raises(ValueError):resume_repair.release_test_cache("qwen")
 assert not (tmp_path/"qwen/scores/test_0.jsonl").exists()
 write(tmp_path/"qwen/frozen.json",dict(frozen=True));resume_repair.release_test_cache("qwen")
 assert (tmp_path/"qwen/scores/test_0.jsonl").exists()


def test_training_queue_refills_idle_gpu_and_adopts_running(tmp_path,monkeypatch):
 from . import controller as c
 monkeypatch.setattr(c,"ROOT",tmp_path);monkeypatch.setattr(c,"GPUS",[0,1])
 jobs=[["experiments.aster_cr.baselines","--kind","loreft","--lr",str(x)] for x in (.0001,.001,.01,.1)]
 write(c.baseline_result(jobs[0]),{"done":True})
 monkeypatch.setattr(c,"find_existing",lambda gpu,args:777 if gpu==1 and args==jobs[1] else None)
 events=[];clock=[0]
 def sleep(seconds):clock[0]+=1
 monkeypatch.setattr(c.time,"sleep",sleep)
 def alive(pid):
  if clock[0]<3:return True
  write(c.baseline_result(jobs[1]),{"done":True});return False
 monkeypatch.setattr(c,"process_alive",alive)
 monkeypatch.setattr(c.subprocess,"check_output",lambda *a,**k:"80000")
 class Proc:
  def __init__(self,cmd,**kw):
   self.pid=800+len(events);self.args=cmd[3:];self.start=clock[0]
   events.append((int(kw["env"]["CUDA_VISIBLE_DEVICES"]),self.args,clock[0]))
  def poll(self):
   if clock[0]<=self.start:return None
   write(c.baseline_result(self.args),{"done":True});return 0
 monkeypatch.setattr(c.subprocess,"Popen",Proc)
 monkeypatch.setattr(c,"status",lambda *a,**kw:None)
 c.baseline_queue("train",jobs)
 assert [(g,args) for g,args,t in events]==[(0,jobs[2]),(0,jobs[3])]
 assert events[1][2]<3
 assert all(read(tmp_path/"jobs"/f"train_job{i}.json")["state"]=="complete" for i in range(4))
 assert read(tmp_path/"jobs/train_job1.json")["adopted"]

def test_training_queue_failure_does_not_advance(tmp_path,monkeypatch):
 import pytest
 from . import controller as c
 monkeypatch.setattr(c,"ROOT",tmp_path);monkeypatch.setattr(c,"GPUS",[0])
 jobs=[["experiments.aster_cr.baselines","--kind","loreft","--lr",str(x)] for x in (.0001,.001)]
 monkeypatch.setattr(c,"find_existing",lambda gpu,args:777 if args==jobs[0] else None)
 monkeypatch.setattr(c,"process_alive",lambda pid:False)
 monkeypatch.setattr(c,"status",lambda *a,**kw:None)
 with pytest.raises(RuntimeError,match="failed jobs"):
  c.baseline_queue("train",jobs)
 assert read(tmp_path/"jobs/train_job0.json")["state"]=="failed"
 assert not (tmp_path/"jobs/train_job1.json").exists()


def test_dev_comparator_queue_result_and_test_rejection(tmp_path,monkeypatch):
 import pytest
 from . import controller as c
 monkeypatch.setattr(c,"ROOT",tmp_path)
 args=["experiments.aster_cr.comparators","--model","qwen","--split","dev","--shard","2","--shards","4"]
 assert c.queue_result(args)==tmp_path/"qwen/comparators/dev_2_metrics.json"
 args[4]="test"
 with pytest.raises(ValueError,match="Only development"):
  c.queue_result(args)

def test_development_comparator_runs_while_training_is_active(tmp_path,monkeypatch):
 from . import controller as c
 monkeypatch.setattr(c,"ROOT",tmp_path);monkeypatch.setattr(c,"GPUS",[0,6])
 train=["experiments.aster_cr.baselines","--kind","lofit","--lr","0.01"]
 dev=["experiments.aster_cr.comparators","--model","qwen","--split","dev","--shard","0","--shards","4"]
 monkeypatch.setattr(c,"find_existing",lambda gpu,args:777 if gpu==0 and args==train else None)
 ticks=[0];events=[]
 def sleep(seconds):ticks[0]+=1
 monkeypatch.setattr(c.time,"sleep",sleep)
 def alive(pid):
  if ticks[0]<2:return True
  write(c.baseline_result(train),{"done":True});return False
 monkeypatch.setattr(c,"process_alive",alive)
 monkeypatch.setattr(c.subprocess,"check_output",lambda *a,**kw:"40000")
 class Proc:
  pid=778
  def __init__(self,cmd,**kw):events.append((kw["env"]["CUDA_VISIBLE_DEVICES"],cmd[3:],ticks[0]))
  def poll(self):
   write(c.queue_result(dev),{"done":True});return 0
 monkeypatch.setattr(c.subprocess,"Popen",Proc)
 monkeypatch.setattr(c,"status",lambda *a,**kw:None)
 c.baseline_queue("train",[train,dev])
 assert events==[("6",dev,0)]
 assert ticks[0]>=2
 assert read(tmp_path/"jobs/train_job1.json")["state"]=="complete"


def test_template_selection_saves_json_booleans(tmp_path,monkeypatch):
 from . import pipeline
 monkeypatch.setattr(pipeline,"ROOT",tmp_path)
 root=tmp_path/"qwen/comparators"
 write(root/"settings.json",{"fitted":True})
 forms=("full","order_no_yes","map_ab","map_ba","equivalent")
 lines(root/"dev_0.jsonl",[{"view_id":"a","scores":{"template_"+f:1. for f in forms}}])
 lines(root/"paths_dev_0.jsonl",[{"methods":{"full":{"correct":True},**{"template_"+f:{"correct":True} for f in forms}}}])
 assert pipeline.choose_template("qwen")=="full"
 saved=read(root/"settings.json")
 assert all(r["eligible"] is True for r in saved["template_dev_candidates"])
 assert saved["best_template"]=="full"
