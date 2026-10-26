import unittest
from clinical_v3_identity import index_predictions

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id=11, case_path_id=2, path_index=0, filename='guide', llm_output='answer')
        self.frozen = {'selected': [{'identity': {k:self.row[k] for k in ('id','case_path_id','path_index','filename')}, 'sample_id':'frozen_hash'}]}
    def test_legacy_row_resolves_without_mutation(self):
        result = index_predictions([self.row], self.frozen)
        self.assertIs(result['frozen_hash'], self.row)
        self.assertNotIn('sample_id', self.row)
    def test_explicit_id_must_match(self):
        with self.assertRaises(ValueError): index_predictions([dict(self.row,sample_id='wrong')], self.frozen)
        self.assertIn('frozen_hash', index_predictions([dict(self.row,sample_id='frozen_hash')], self.frozen))
    def test_duplicate_is_rejected(self):
        with self.assertRaises(ValueError): index_predictions([self.row,self.row], self.frozen)
    def test_unfrozen_identity_is_rejected(self):
        with self.assertRaises(ValueError): index_predictions([dict(self.row,id=99)], self.frozen)

if __name__ == '__main__': unittest.main()
