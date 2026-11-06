"""CPU integration of data-independent fit/selection/report artifact flow."""
import json,tempfile,unittest
from pathlib import Path
import torch
from .common import write_json,write_rows,append
from . import pipeline
class PipelineTest(unittest.TestCase):
    def test_selection_and_reports_with_synthetic_features(self):
        with tempfile.TemporaryDirectory(prefix="aster_closed_form_cpu_") as folder:
            root=Path(folder);views=[];pairs=[];features={}
            for split in ("fit","dev","test"):
                source="synthetic_"+split
                variants=[("base0",0),("base1",1),("swap0",0),("swap1",1),("condition",1),("pc0",0),("pc1",0),("pp0",1),("pp1",1)]
                ids={}
                for i,(variant,label) in enumerate(variants):
                    vid=source+":"+variant;ids[variant]=vid
                    v=dict(view_id=vid,group_id=source,source_id=source,split=split,cohort="main",variant=variant,label=label,
                           original=variant in ("base0","base1"),prompt="synthetic "+vid,prompts={})
                    views.append(v)
                    z=torch.zeros(1,32,4)
                    z[:,:,0]=label*2-1;z[:,:,1]=i*.1;z[:,:,2]=.2*(i%2);z[:,:,3]=.1*(i%3)
                    f=dict(ell=float(label*2-1)+i*.02,heads=z,**{k:v[k] for k in ("split","original","label","source_id","group_id","cohort","view_id")})
                    file=pipeline.feature_file(root,vid);file.parent.mkdir(exist_ok=True,parents=True);torch.save(f,file)
                    for method in ("full","direct","semantic","ignore","boundary_guideline","boundary_facts"):
                        append(root/"scores/baseline_0.jsonl",dict(view_id=vid,method=method,ell=f["ell"],status="complete"))
                for k,l,r,sub in (("A","base0","swap0",""),("A","base1","swap1",""),("C","base0","condition",""),("F","base0","base1",""),
                                  ("P","pc0","pc1","clinical"),("P","pp0","pp1","preference")):
                    pairs.append(dict(pair_id=source+k+sub+l,group_id=source,source_id=source,split=split,cohort="main",kind=k,subtype=sub,left=ids[l],right=ids[r]))
            write_rows(root/"data/views.jsonl",views);write_rows(root/"data/pairs.jsonl",pairs)
            for k in ("A","P"):write_json(root/"m1"/("m1_"+k+".json"),dict(H=[0],target_met=False,curve=[]))
            pipeline.fit_grid(root)
            configs=json.loads((root/"operators/grid.json").read_text());self.assertEqual(len(configs),18)
            for v in views:
                if v["split"]=="dev":
                    for c in configs:append(root/"scores/grid_0.jsonl",dict(view_id=v["view_id"],method=c["method"],ell=v["label"]*2-1))
            pipeline.select_and_ablations(root)
            final=json.loads((root/"operators/final.json").read_text());self.assertEqual(len(final),14)
            # Only dev amplification candidates are emitted before the frozen choice.
            for v in views:
                if v["split"]=="dev":
                    for m in ("aster","amplify_1.25","amplify_1.5","amplify_2.0"):
                        append(root/"scores/final_amp_dev_0.jsonl",dict(view_id=v["view_id"],method=m,ell=v["label"]*2-1))
            pipeline.select_amplification(root)
            amp=json.loads((root/"results/amplification_selection.json").read_text())
            self.assertEqual(amp["selected"],"aster")
            for v in views:
                for c in final:
                    if c["method"].startswith("amplify_"):continue
                    append(root/"scores/final_0.jsonl",dict(view_id=v["view_id"],method=c["method"],ell=v["label"]*2-1))
            for sp in ("dev","test"):
                write_json(root/"mechanisms"/("mechanisms_"+sp+".json"),dict(h1=[],h2=[],h2_node_count=0,h2_pair_count=0,local_theory={}))
            write_json(root/"data/manifest.json",dict(automatic_pilot=True,synthetic_test=True))
            write_json(root/"data/valid_counts.json",dict(views=len(views),pairs=len(pairs)))
            pipeline.report(root)
            self.assertTrue((root/"REPORT.zh.md").exists())
            results=json.loads((root/"results/evaluations.json").read_text())
            self.assertEqual(results["main_test"]["methods"]["aster"]["macro_PA"],1)
if __name__=="__main__":unittest.main()
