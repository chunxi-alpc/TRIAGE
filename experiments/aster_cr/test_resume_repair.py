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
