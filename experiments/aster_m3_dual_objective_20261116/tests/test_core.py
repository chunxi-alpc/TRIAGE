import ast
import inspect
import unittest
import numpy as np
from experiments.aster_m3_dual_objective_20261116 import solver, evaluation as ev
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as old

class SolverTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.X=rng.normal(size=(30,16));self.E=rng.normal(size=(20,16))
        self.y=rng.normal(size=30)
        self.a=np.ones(30);self.c=np.ones(20)
    def fit(self,eta=1,reg=.01):
        return solver.solve(self.X,self.y,self.E,self.a,self.c,eta,reg)
    def test_sign_direction(self):
        w,_=solver.solve([[1.]],[2.],[[1.]],[1.],[1.],0,1.)
        self.assertAlmostEqual(w[0],-1.)
        self.assertLess(abs(2+w[0]),2)
    def test_eta_zero_analytic_ridge(self):
        w,_=self.fit(eta=0)
        target=-np.linalg.solve(self.X.T@self.X+.01*np.eye(16),self.X.T@self.y)
        np.testing.assert_allclose(w,target,rtol=1e-12,atol=1e-12)
    def test_regularization_positive_definite(self):
        X=np.ones((4,16));E=np.ones((3,16))
        _,d=solver.solve(X,np.ones(4),E,np.ones(4),np.ones(3),1,.01)
        self.assertGreater(d["min_eigenvalue"],0)
        self.assertLess(d["relative_stationary_residual"],1e-12)
        with self.assertRaises(ValueError): self.fit(reg=0)
    def test_cf_penalizes_correction(self):
        w0,_=self.fit(eta=0);w1,d=self.fit(eta=10)
        self.assertLess(np.linalg.norm(self.E@w1),np.linalg.norm(self.E@w0))
        self.assertAlmostEqual(d["L_CF"],float(np.sum((self.E@w1)**2)))
        self.assertAlmostEqual(d["objective"],d["L_AP"]+10*d["L_CF"]+d["regularization_term"])
    def test_centering_preserves_pairs(self):
        w,_=self.fit()
        rho=np.arange(1,31,dtype=float);rho/=rho.sum()
        mu=solver.center(w,self.X,rho)
        np.testing.assert_allclose(np.diff(self.X@w-mu),np.diff(self.X@w),atol=1e-12)
        self.assertAlmostEqual(float(rho@(self.X@w-mu)),0)
    def test_same_m4_after_centering(self):
        w,_=self.fit();labels=np.array([0,1]*15);raw=self.y
        b0=solver.calibrate(raw,self.X,labels,w,0)
        b1=solver.calibrate(raw,self.X,labels,w,3.2)
        self.assertAlmostEqual(b1-b0,3.2,places=10)
    def test_balanced_weights(self):
        ps=[dict(kind=k,source_id=s) for k,s,n in
            [("A","x",1),("A","y",3),("P","x",7),("P","y",2)] for _ in range(n)]
        ww=solver.pair_weights(ps,"E1",solver.WEIGHT_TOTAL)
        self.assertAlmostEqual(ww.sum(),4887.)
        for k in ("A","P"):
            self.assertAlmostEqual(sum(w for p,w in zip(ps,ww) if p["kind"]==k),2443.5)
            for s in ("x","y"):
                self.assertAlmostEqual(sum(w for p,w in zip(ps,ww)
                    if p["kind"]==k and p["source_id"]==s),1221.75)
    def test_no_inequality_or_budget_fields(self):
        source=inspect.getsource(solver)
        tree=ast.parse(source)
        calls={getattr(n.func,"id",getattr(n.func,"attr","")) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {"minimize","LinearConstraint","clip","OSQP"})
        _,d=self.fit()
        self.assertFalse(any("budget" in k or "delta" in k or "constraint" in k for k in d))
    def test_shared_moments_match_original_e1(self):
        rng=np.random.default_rng(5)
        ps=[];raw={};scores={}
        for i in range(50):
            a,b="l"+str(i),"r"+str(i)
            yl,yr=(0,0) if i<25 else (0,1)
            ps.append(dict(pair_id=str(i),left=a,right=b,left_label=yl,right_label=yr,
                source_id=str(i%5),kind="A" if i<25 else "C"))
            raw[a],raw[b]=rng.normal(size=2)
            scores[a]=raw[a]+rng.normal();scores[b]=raw[b]+rng.normal()
        rr=ev.pair_records(ps,scores,raw,"x")
        boot=ev.SourceBootstrap([p["source_id"] for p in ps],draws=20)
        mm,_=ev.aggregate(rr,[],[],boot)
        oldrr=[dict(r,over_budget=False) for r in rr]
        ref,_,_=old.pair_statistics(oldrr,boot)
        for k in mm.keys()&ref.keys(): np.testing.assert_allclose(mm[k],ref[k],atol=1e-12)
    def test_selection_and_pareto(self):
        a=dict(J=.1,cf_er=1.,cf_direction_degradation_count=0,cf_paired_accuracy=.8,
            natural_path_accuracy=.9,regularization=1.,ap_vr=.2,cf_normalized_distortion=.1)
        b=dict(a,regularization=10.)
        self.assertIs(max([a,b],key=ev.selection_key),b)
        c=dict(a,ap_vr=.1,cf_normalized_distortion=.2)
        self.assertEqual(ev.pareto_mask([a,c]),[True,False])
if __name__=="__main__": unittest.main()
