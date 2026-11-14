"""CPU configuration/version rejection properties; no Torch/model execution."""
import copy
import tempfile
import unittest
from pathlib import Path

from compatibility import CRITICAL_CONFIG_FIELDS, load_frozen_manifest, config_audit, environment_audit
from run import HERE, ROOT, read


class FrozenModelGuardProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read(HERE / 'protocol.json')
        cls.original_path = ROOT / 'experiments/care_qwen3_4b_base_20261016/results/manifest.json'
        cls.original = load_frozen_manifest(cls.original_path, cls.protocol)

    def test_each_critical_geometry_field_and_missing_null_are_rejected(self):
        self.assertTrue(config_audit(self.original['config'], self.original, self.protocol)['compatible'])
        for key in CRITICAL_CONFIG_FIELDS:
            with self.subTest(field=key):
                changed = copy.deepcopy(self.original['config'])
                changed[key] = 'unit-test-drift'
                audit = config_audit(changed, self.original, self.protocol)
                self.assertFalse(audit['compatible'])
                self.assertEqual(set(audit['differences']), {key})
        changed = copy.deepcopy(self.original['config'])
        del changed['rope_scaling']
        self.assertTrue(config_audit(changed, self.original, self.protocol)['differences']['rope_scaling']['missing_from_observed'])

    def test_unvalidated_core_versions_block_while_other_differences_are_recorded(self):
        versions = self.original['versions'].copy()
        self.assertTrue(environment_audit(versions, self.original, self.protocol)['compatible'])
        for name, value in [('torch', '2.9.0'), ('torch', '2.8.0+cu128'), ('transformers', '4.58.0'), ('torch', None)]:
            with self.subTest(package=name, value=value):
                changed = dict(versions, **{name: value})
                audit = environment_audit(changed, self.original, self.protocol)
                self.assertFalse(audit['compatible'])
                self.assertEqual(set(audit['blocking_differences']), {name})
        audit = environment_audit(dict(versions, numpy='unit-test-version'), self.original, self.protocol)
        self.assertTrue(audit['compatible'])
        self.assertEqual(audit['differences']['numpy']['observed'], 'unit-test-version')

    def test_changed_reference_manifest_cannot_relax_the_checks(self):
        with tempfile.TemporaryDirectory(prefix='care-model-guard-test-') as directory:
            path = Path(directory) / 'manifest.json'
            path.write_bytes(self.original_path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                load_frozen_manifest(path, self.protocol)


if __name__ == '__main__':
    unittest.main(verbosity=2)
