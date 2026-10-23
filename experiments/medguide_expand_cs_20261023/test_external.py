import unittest
from external import calc,weight_rule
class CalculatorTest(unittest.TestCase):
 def test_bmi_boundaries(self):
  self.assertEqual(weight_rule(18.49,60,55),('underweight',60));self.assertEqual(weight_rule(18.5,60,55),('normal',55));self.assertEqual(weight_rule(25,60,55),('adjusted',57))
 def test_published_example(self):
  e={'sex':'Male','age':[87,'years'],'weight':[48,'kg'],'height':[163,'cm'],'creatinine':[1.4,'mg/dL']}
  self.assertAlmostEqual(calc(e)['crcl'],25.2381,places=4)
 def test_units_and_adjustment(self):
  e={'sex':'Female','age':[62,'years'],'weight':[72,'kg'],'height':[1.6,'m'],'creatinine':[85,'µmol/L']}
  self.assertAlmostEqual(calc(e)['crcl'],57.68058628,places=6);self.assertLess(abs(calc(e)['crcl']-57.65181),0.05) # source rounds converted creatinine to 0.962
  self.assertEqual(calc(e,29)['branch'],'normal');self.assertNotEqual(calc(e)['crcl'],calc(e,29)['crcl'])
 def test_point_bound_exclusion(self):
  from prepare import build
  r={'disease':'t','path':['root','≥6 mm','B'],'profile':'A lung nodule with size greater than or equal to 6 mm.','answer':'B','answer_text':'B','options':['A','B']}
  _,_,cs,audit=build([r],{'t':[['root','<6 mm','A'],['root','≥6 mm','B']]})
  self.assertFalse(cs[0]['patient_point_value_verified']);self.assertEqual(audit[0]['reason'],'patient evidence gives a bound rather than a point value')
if __name__=='__main__':unittest.main()
