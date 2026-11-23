import unittest
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from .solver import residual, solve, calibrate, disjoint, DELTA
from .run import split_inner, selection_key


class SolverTests(unittest.TestCase):
    def test_projection_and_reorthogonalization(self):
        rng=np.random.default_rng(7);q=np.linalg.qr(rng.normal(size=(17,3)))[0]
        z=rng.normal(size=(11,17));s=q@np.array([[2.,1,0],[0,3,1],[0,0,4]])
        r=residual(z,s)
        np.testing.assert_allclose(r@q,0,atol=1e-13)
        np.testing.assert_allclose(r,residual(z,q),atol=1e-13)

    def test_zero_rank(self):
        z=np.arange(21).reshape(3,7)
        np.testing.assert_array_equal(residual(z,np.empty((7,0))),z)

    def test_pair_difference_and_center(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(10,13));s=rng.normal(size=(13,2));w=rng.normal(size=13)
        r=residual(x,s)
        np.testing.assert_allclose(r[1::2]-r[::2],residual(x[1::2]-x[::2],s),atol=1e-13)
        c=r@w
        np.testing.assert_allclose(np.diff(c-23),np.diff(c),atol=1e-13)
        center,b=calibrate(np.arange(10)-5,c,np.arange(10)%2)
        np.testing.assert_allclose(np.diff(np.arange(10)-5+c-center+b),np.diff(np.arange(10)-5+c),atol=1e-13)

    def test_low_dimension_reference_qp_and_every_budget(self):
        rng=np.random.default_rng(11)
        for d,n,m in [(7,12,8),(23,12,8),(5,13,9)]:
            X=rng.normal(size=(n,d));E=rng.normal(size=(m,d));y=rng.normal(size=n)*3;weights=rng.uniform(.5,2,n)
            w,diag,ec,nu=solve(X,y,weights,E,.1)
            Q=X.T@(weights[:,None]*X)+.1*np.eye(d);q=X.T@(weights*y)
            ref=minimize(lambda v:.5*v@Q@v+q@v,np.zeros(d),jac=lambda v:Q@v+q,method='SLSQP',
                constraints=[LinearConstraint(E,-DELTA,DELTA)],options=dict(ftol=1e-12,maxiter=2000))
            self.assertTrue(ref.success)
            np.testing.assert_allclose(w,ref.x,atol=2e-6)
            self.assertLessEqual(np.max(np.abs(ec)),DELTA+1e-7)
            self.assertLess(diag['relative_duality_gap'],1e-7)

    def test_duplicate_rows_without_feature_truncation(self):
        rng=np.random.default_rng(6);X=rng.normal(size=(8,25));X[3]=X[1];E=rng.normal(size=(5,25));E[3]=E[1]
        w,diag,ec,nu=solve(X,np.arange(8),np.ones(8),E,1.)
        self.assertEqual(diag['feature_dimension'],25)
        self.assertLessEqual(np.max(np.abs(ec)),DELTA+1e-7)

    def test_source_isolation_and_selection(self):
        tr,dev=split_inner([str(i) for i in range(64)],10)
        self.assertEqual((len(tr),len(dev)),(48,16));disjoint(tr,dev,['heldout'])
        with self.assertRaises(ValueError):disjoint(tr,dev,tr[:1])
        a=dict(over_budget_rate=.1,direction_degradation_rate=0,ER=1,CF_correction_RMS=.1,AP_RMS=3,**{'lambda':1})
        b=dict(a,over_budget_rate=.2,AP_RMS=0)
        self.assertLess(selection_key(a),selection_key(b))


if __name__=='__main__':unittest.main()
