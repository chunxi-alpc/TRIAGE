import unittest
from reuse import key
class ReuseTests(unittest.TestCase):
 def test_equivalent_bias_layout(self):
  a={'spans':[[10,20],[30,40]],'positive':[0],'other':[1],'other_gain':-.5}
  b={'spans':[[0,5],[30,40],[10,20]],'positive':[2],'other':[1],'other_gain':-.5}
  self.assertEqual(key('prompt',['A','B'],a),key('prompt',['A','B'],b))
 def test_threshold_and_decoding_must_match(self):
  self.assertNotEqual(key('age < 6',['A','B']),key('age < 3',['A','B']))
  self.assertNotEqual(key('p',['A'],closed='q',cap=8),key('p',['A'],closed='q',cap=128))
  self.assertNotEqual(key('p',['A'],closed='q',cap=128),key('p',['A'],closed='q',cap=128,beta=.1))
 def test_signed_is_not_unsigned(self):
  a={'spans':[[10,20],[30,40]],'positive':[0],'other':[1],'other_gain':-.5}
  self.assertNotEqual(key('p',['A'],a),key('p',['A'],dict(a,other_gain=.5)))
if __name__=='__main__':unittest.main()
