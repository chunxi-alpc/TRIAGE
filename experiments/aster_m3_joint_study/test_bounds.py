"""Closed-form numerical checks for the post-fit attainable-correction bound."""
import unittest
import numpy as np
from .diagnose import correction_bound


class BoundTests(unittest.TestCase):
    def test_box_has_known_support_function_with_negative_weights(self):
        direction=np.zeros(16);direction[:2]=[-2,1]
        result=correction_bound(direction,np.eye(16),delta=.1)
        self.assertEqual(result["status"],"optimal")
        self.assertAlmostEqual(result["maximum"],.3,places=10)
        self.assertLessEqual(result["max_constraint_violation"],1e-9)

    def test_unconstrained_direction_is_reported_as_unbounded(self):
        direction=np.zeros(16);direction[0]=1
        result=correction_bound(direction,np.zeros((2,16)),delta=.1)
        self.assertEqual(result["status"],"unbounded")
        self.assertIsNone(result["maximum"])


if __name__=="__main__":unittest.main()
