"""Synthetic plot checks; these figures are never reported as model experiments."""
import csv,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import report
from .audit import read
from .prepare import CODE,RUN
from .statistics import scenarios,sets

class ReportTests(unittest.TestCase):
    def check(self,nonempty):
        cfg=read(CODE/"configuration.json")
        shape=(4,40,1152)
        rng=np.random.default_rng(1)
        actual=rng.normal(size=shape)
        effect=dict(true=actual,null=actual*.5,contrast=actual*.5)
        stats=dict(mean=actual.mean(1),source_direction_consistency=np.ones((4,1152))*.7)
        full={}
        for sc in scenarios(cfg):
            mask=np.zeros((4,1152),bool)
            if nonempty:mask[:,10]=True
            ss=sets(mask)
            full[sc["name"]]=dict(sets=ss,AP_frequency=mask[:2].any(0).astype(float),
                CF_frequency=mask[2:].any(0).astype(float),union_frequency=mask.any(0).astype(float),
                summary=dict(counts={k:len(v) for k,v in ss.items()},core_heads=[10] if nonempty else [],by_union_count=int(nonempty)))
        result=dict(recommendation="A" if nonempty else "C",stability=[],stable_scan_values={})
        plan=dict(matched_pairs_by_kind={k:1 for k in "APCF"},eligible_sources_by_kind={k:40 for k in "APCF"},
                  reused_unique_scores=0,new_unique_patch_forwards=0,new_baseline_forwards=0)
        cores=[dict(head=10,stable_core=True)] if nonempty else []
        with tempfile.TemporaryDirectory(prefix="synthetic_plot_",dir=RUN) as tmp:
            early=Path(tmp)/"early"
            # These figures must work before subset frequencies or a core exist.
            report.render_selection(cfg,{n:dict(sets=x["sets"]) for n,x in full.items()},early)
            self.assertEqual(len(list((early/"figures").glob("*.png"))),2)
            with (early/"threshold_counts.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),sum(len(v) for v in cfg["grid"].values()))
            self.assertTrue(all(int(r["union"])==int(nonempty) for r in rows))
            with (early/"jaccard.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(full)**2)
            self.assertTrue(all(r["union_jaccard"]==("1.0" if nonempty else "") for r in rows))
            with patch.object(report,"OUT",Path(tmp)):
                report.render(cfg,plan,effect,stats,full,result,cores)
            self.assertEqual(len(list((Path(tmp)/"figures").glob("*.png"))),4)
            self.assertTrue((Path(tmp)/"REPORT.zh.md").exists())
    def test_empty_selection(self):self.check(False)
    def test_selected_core(self):self.check(True)

if __name__=="__main__":unittest.main()
