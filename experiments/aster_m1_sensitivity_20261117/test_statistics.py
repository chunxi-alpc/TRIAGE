import unittest
import numpy as np
from .statistics import bh,flip_p,robust_z,candidates,select,source_statistics

class StatisticalTests(unittest.TestCase):
    def test_bh_known_values_and_order(self):
        np.testing.assert_allclose(bh([.01,.04,.03,.002]),[.02,.04,.04,.008])
        np.testing.assert_allclose(bh([1,1]),[1,1])
    def test_exact_one_and_two_sided(self):
        x=np.array([[1,1],[2,2],[3,3.]])
        np.testing.assert_allclose(flip_p(x,[False,True],100,0),[1/8,1/4])
        np.testing.assert_allclose(flip_p(-x,[False,True],100,0),[1,1/4])
    def test_missing_sources_not_zero_effects(self):
        x=np.array([[1.],[2.],[3.],[np.nan]])
        np.testing.assert_allclose(flip_p(x,[False],100,0),[1/8])
    def test_degenerate_robust_scale(self):
        z=robust_z([0,0,0,10])
        self.assertTrue(np.isposinf(z[-1]))
        np.testing.assert_array_equal(z[:3],[0,0,0])
    def test_source_unit_not_pair_count(self):
        x=np.array([[1.],[2.],[3.]])
        cfg=dict(sign_flip_draws=100,exact_sign_flip_max_sources=16,bootstrap_draws=1000)
        s=source_statistics(x,[False],cfg,2)
        self.assertEqual(s["valid_sources"][0],3)
        self.assertEqual(s["mean"][0],2)
    def test_min_sources_and_cf_negative_preserved(self):
        shape=(4,2);mask=np.ones(shape,bool)
        stats=dict(p=np.full(shape,.001),valid_sources=np.full(shape,10),
                   mean=np.full(shape,-1.),direction_consistency=np.ones(shape))
        stats["valid_sources"][2,0]=4
        selected,q=select(mask,stats,dict(fdr=.05,direction_consistency=.8,min_valid_sources=5))
        self.assertFalse(selected[:2].any())
        self.assertFalse(selected[2,0])
        self.assertTrue(selected[2,1])
    def test_recompute_family_adjustment(self):
        shape=(4,2)
        stats=dict(p=np.array([[.02,1],[1,1],[1,1],[1,1]]),valid_sources=np.full(shape,10),
                   mean=np.ones(shape),direction_consistency=np.ones(shape))
        settings=dict(fdr=.05,direction_consistency=.8,min_valid_sources=5)
        mask=np.zeros(shape,bool);mask[0,0]=True
        self.assertTrue(select(mask,stats,settings)[0][0,0])
        self.assertFalse(select(np.ones(shape,bool),stats,settings)[0][0,0])
    def test_five_fold_source_screening(self):
        x=np.zeros((4,10,4));x[:,:,3]=10
        mask,f=candidates(x,np.arange(10)%5,np.arange(10),dict(gradient_threshold=2,candidate_frequency=.8))
        self.assertEqual(int(mask.sum()),4)
        np.testing.assert_array_equal(f[:,3],1)

if __name__=="__main__":unittest.main()
