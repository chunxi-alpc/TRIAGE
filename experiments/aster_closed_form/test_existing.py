"""Necessary regressions for reused inputs and independent diagnostic denominators."""
import json,tempfile,unittest
from pathlib import Path
from .common import rows,write_rows
from .existing import ROOT,historical_template,HISTORICAL
from .existing_report import history_report,path_report,threshold_value

class ExistingDataChecks(unittest.TestCase):
    def test_fixed_labels_splits_and_removed_C_only(self):
        from .existing import ORIGINAL
        old={v["view_id"]:v for v in rows(ORIGINAL/"data/views.jsonl")}
        new=rows(ROOT/"data/views.jsonl")
        self.assertEqual(len(new),1290)
        self.assertTrue(all(v==old[v["view_id"]] for v in new))
        excluded=set(old)-{v["view_id"] for v in new}
        self.assertEqual(len(excluded),9)
        self.assertTrue(all(old[k]["variant"]=="condition" for k in excluded))
        sources={}
        for v in new:
            self.assertEqual(sources.setdefault(v["source_id"],v["split"]),v["split"])
        system,prompt=historical_template()
        raw=rows(HISTORICAL/"results/scores_full_0.jsonl")+rows(HISTORICAL/"results/scores_full_1.jsonl")
        saved={r["score_id"]:r for r in raw if r.get("status")=="complete"}
        for r in rows(ROOT/"diagnostics/fact_preference_inputs.jsonl")+rows(ROOT/"diagnostics/deletion_restore_inputs.jsonl"):
            old_score=saved.get(r["score_id"])
            if old_score:
                self.assertEqual(prompt(r),old_score["user_prompt"])
                self.assertEqual(system,old_score["model_settings"]["system"])
                self.assertEqual(r["gold"],old_score["gold"])

    def test_historical_failures_stay_in_pair_denominators(self):
        with tempfile.TemporaryDirectory(prefix="aster-existing-test-") as td:
            root=Path(td)
            (root/"results").mkdir()
            for file in ("fact_preference_inputs.jsonl","fact_preference_pairs.jsonl","deletion_restore_inputs.jsonl"):
                write_rows(root/"diagnostics"/file,rows(ROOT/"diagnostics"/file))
            inputs=rows(root/"diagnostics/fact_preference_inputs.jsonl")+rows(root/"diagnostics/deletion_restore_inputs.jsonl")
            target=next(r["score_id"] for r in inputs if r.get("metric_level")=="below" and r.get("background")=="neutral" and r["view"]=="full")
            scores=[]
            for r in inputs:
                for method in ("natural","aster"):
                    failed=method=="aster" and r["score_id"]==target
                    scores.append(dict(score_id=r["score_id"],method=method,ell=None if failed else 2*r["gold"]-1,
                                       status="failed" if failed else "complete"))
            write_rows(root/"diagnostics/scores_0.jsonl",scores)
            fact,deletion=history_report(root)
            self.assertEqual(fact["full:aster"]["PA_F"]["denominator"],56)
            self.assertEqual(fact["full:aster"]["PA_F"]["numerator"],55)
            self.assertEqual(fact["full:aster"]["PA_P"]["denominator"],84)
            self.assertEqual(fact["full:aster"]["PA_P"]["numerator"],83)
            self.assertEqual(deletion["necessary_restore:aster"]["denominator"],11)
            self.assertEqual(deletion["irrelevant_restore:aster"]["denominator"],10)

    def test_paths_keep_432_and_nested_100_with_missing_material(self):
        from .paths import METHODS
        with tempfile.TemporaryDirectory(prefix="aster-path-test-") as td:
            root=Path(td);(root/"results").mkdir()
            cases=rows(ROOT/"data/path_cases.jsonl")
            write_rows(root/"data/path_cases.jsonl",cases)
            records=[]
            for i,c in enumerate(cases):
                for method in METHODS:
                    missing=i==0 and method=="semantic"
                    records.append(dict(case_id=c["case_id"],source_id=c["source_id"],method=method,
                                        correct=not missing,status="unavailable_input" if missing else "complete"))
            write_rows(root/"paths/predictions_0.jsonl",records)
            report=path_report(root)
            self.assertEqual(report["all_432"]["cases"],432)
            self.assertEqual(report["original_100"]["cases"],100)
            semantic=report["all_432"]["methods"]["semantic"]
            self.assertEqual(semantic["denominator"],432)
            self.assertEqual(semantic["unavailable_existing_inputs"],1)
            self.assertEqual(semantic["technical_failures"],0)
if __name__=="__main__":unittest.main()
