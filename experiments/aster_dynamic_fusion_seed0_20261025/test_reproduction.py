import copy, importlib, json, unittest
from pathlib import Path
import numpy as np
import reproduce as r
import p3_hard_anchor_exploratory as p
from balanced_analysis import paired, prediction, configs, fit

class ArithmeticTests(unittest.TestCase):
    def test_probability_arithmetic(self):
        np.testing.assert_allclose(r.temp([.2,.8],.5),[1/17,16/17])
        np.testing.assert_allclose(r.softmax([1000,1000]),[.5,.5])
    def test_hard_anchor(self):
        self.assertEqual(p.hard_anchor({"q":[.6,.4],"none":1}),0)
        self.assertEqual(p.hard_anchor({"q":[.5,.5],"none":1}),1)
    def test_seed_regression_counts(self):
        rows=[dict(model="qwen",tree="a",variant="NC",seed=0,q=[.9,.1],pred=0,none=1,gold=1),
              dict(model="qwen",tree="b",variant="AC",seed=0,q=[.1,.9],pred=1,none=0,gold=1)]
        x=p.summarize(rows,"qwen")
        self.assertEqual(x["by_seed"]["0"]["fusion_harm"],.5)
        self.assertEqual(x["by_seed"]["0"]["fusion_correction"],.5)
    def test_metric_lists_and_identity(self):
        rows=[dict(gold_option=0,none=0,variant=v) for v in ("NC","AC","IC","SW")]
        x=r.metric(rows,[1,0,0,0])
        self.assertEqual(x["accuracy"],.75)
        self.assertEqual(x["accuracy"]-1,x["correction_rate"]-x["harm_rate"])
    def test_matched_zero_required(self):
        row=r.prepare(dict(id="a",scores={"original":[1.,0.],"mass_zero_control":[1.,0.],"action_bias_2.0_last_half":[1.,0.]},option_S=[.4,.6]))
        cfg=dict(kind="residual",beta=.75,method="action_bias_2.0_last_half",gamma=1.,scale=2)
        with self.assertRaises(ValueError): r.predict(row,cfg)
        row["scores"]["zero_last_half"]=[1.,0.]
        self.assertEqual(r.predict(row,cfg),r.predict(row,dict(kind="static",beta=.75)))
    def test_zero_gamma(self):
        row=r.prepare(dict(id="a",scores={"original":[1.,0.],"mass_zero_control":[.2,.8],"action_bias_1.0_all":[-5.,5.]},option_S=[.4,.6]))
        for b in (0,.5,1):
            self.assertEqual(r.predict(row,dict(kind="residual",method="action_bias_1.0_all",beta=b,gamma=0,scale=2)),r.predict(row,dict(kind="static",beta=b)))
    def test_paired_arithmetic(self):
        rows=[dict(tree=t,gold_option=0,none=0,variant="NC") for t in ("a","a","b","b")]
        x=paired(rows,[0,1,0,0],[1,0,1,0])
        self.assertEqual(x["accuracy_delta"],.25)
        self.assertEqual(x["corrected_vs_comparator"]-x["regressed_vs_comparator"],1)
    def test_prediction_does_not_read_gold(self):
        row=r.prepare(dict(id="a",none_q=[.6,.4],option_S=[.2,.8],gold_option=0))
        for k in ("static","dynamic","hard_anchor"):
            for c in configs(k):
                a=prediction(row,c); row["gold_option"]=1
                self.assertEqual(a,prediction(row,c))
    def test_equal_unique_candidate_counts(self):
        for kind in ("static","dynamic","hard_anchor","residual"):
            grid=list(configs(kind))
            self.assertEqual(len(grid),27)
            self.assertEqual(len({json.dumps(c,sort_keys=True) for c in grid}),27)
    def test_held_tree_labels_not_used_for_its_parameters(self):
        rows=[r.prepare(dict(id=str(i),tree=t,variant=v,none=0,none_q=[.7,.3],option_S=[.2,.8],gold_option=i%2)) for i,(t,v) in enumerate((t,v) for t in ("a","b","c","d") for v in ("NC","AC"))]
        before=fit(rows,"static")["folds"]["a"]["parameters"]
        changed=copy.deepcopy(rows)
        for x in changed:
            if x["tree"]=="a": x["gold_option"]=1-x["gold_option"]
        self.assertEqual(before,fit(changed,"static")["folds"]["a"]["parameters"])
    def test_published_p3_identity(self):
        data=json.loads((r.ROOT/"p3_hard_anchor_exploratory.json").read_text())
        for m,x in data["models"].items():
            z=x["overall"]
            self.assertAlmostEqual(z["fusion_accuracy"]-z["original_accuracy"],z["fusion_correction"]-z["fusion_harm"])
            for field in ("fusion_harm","fusion_correction"):
                self.assertAlmostEqual(sum(s[field]*s["n"] for s in x["by_seed"].values())/x["n"],z[field])
if __name__=="__main__": unittest.main()
