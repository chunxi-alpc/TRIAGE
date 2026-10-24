import unittest
from trial import parse_check
class ParseCheckTests(unittest.TestCase):
 def test_inline_and_formatted_marker(self):
  self.assertEqual(parse_check('The value is 5 mm. Branch: B.'),'B')
  self.assertEqual(parse_check('Comparison completed.\n**Branch: A**'),'A')
 def test_last_explicit_marker(self):
  self.assertEqual(parse_check('Branch: A is a possibility. Final selection: Branch: B'),'B')
 def test_no_invented_choice(self):
  self.assertIsNone(parse_check('The patient might satisfy condition A.'))
  self.assertIsNone(parse_check('Branch: uncertain'))
if __name__=='__main__':unittest.main()
