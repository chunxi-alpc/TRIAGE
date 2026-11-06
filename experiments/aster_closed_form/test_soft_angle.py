import math
import unittest
import torch
from .operators import oblique_factors, algebra_diagnostics, soft_angle_from_saved

class SoftAngleTests(unittest.TestCase):
    def test_angles_and_signal(self):
        signal=torch.tensor([[1.],[0.]],dtype=torch.float64)
        for angle in (0,15,30,45,60,90):
            rad=math.radians(angle)
            u=torch.tensor([[math.cos(rad)],[math.sin(rad)]],dtype=torch.float64)
            n,b,g=oblique_factors(u,signal,45,variant="soft_angle")
            d=n@b
            expected=min(1.,2*math.sin(rad)**2)
            self.assertTrue(torch.allclose(d@u,expected*u,atol=1e-12,rtol=1e-12))
            self.assertLess(float(torch.linalg.vector_norm(d@signal)),1e-12)
            self.assertLessEqual(float(torch.linalg.matrix_norm(d,ord=2)),math.sqrt(2)+1e-12)
            if angle>=45:
                hn,hb,_=oblique_factors(u,signal,45)
                self.assertTrue(torch.allclose(d,hn@hb,atol=1e-12,rtol=1e-12))
            if angle==0:self.assertEqual(float(d.abs().max()),0.)
            if angle==30:self.assertAlmostEqual(expected,.5)
            if angle==15:self.assertAlmostEqual(expected,1-math.sqrt(3)/2)

    def test_saved_basis_conversion_and_residuals(self):
        torch.manual_seed(7)
        u=torch.linalg.qr(torch.randn(16,4,dtype=torch.float64)).Q.float()
        s=torch.linalg.qr(torch.randn(16,5,dtype=torch.float64)).Q.float()
        n,b,_=oblique_factors(u,s,45)
        mu=torch.randn(16)
        hard={2:dict(N=n.float(),B=b.float(),U_N=u,S=s,mu=mu,type_bases={})}
        soft,diag=soft_angle_from_saved(hard)
        self.assertTrue(torch.equal(soft[2]["mu"],mu))
        self.assertTrue(torch.equal(soft[2]["S"],s))
        self.assertTrue(torch.equal(soft[2]["U_N"],u))
        self.assertLess(diag["2"]["algebra"]["signal_preservation_error"],1e-5)
        self.assertTrue(diag["2"]["algebra"]["angle_bound_satisfied"])
        self.assertFalse(diag["2"]["algebra"]["exact_erasure_expected"])
        self.assertFalse(diag["2"]["algebra"]["idempotence_expected"])
        self.assertLess(diag["2"]["high_angle_vs_stored_hard_error"],1e-5)

if __name__=="__main__":unittest.main()
