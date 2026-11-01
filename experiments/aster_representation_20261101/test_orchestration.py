import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import orchestrate


class ConfirmationQueueTests(unittest.TestCase):
    def test_ready_methods_run_once_without_waiting_for_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = {'layer': 18, 'epoch': 3, 'checkpoint': 'full.pt'}
            (root/'selected_full.json').write_text(json.dumps(selected))
            complete = {'train_full_18'} | {'dev_'+k for k in ['zero', 'static', 'fusion', 'blind_lookup']}
            for kind in ['ordinary', 'no_js', 'blind_readout', 'context']:
                folder = root/f'{kind}_layer18'
                folder.mkdir()
                (folder/'selected.json').write_text(json.dumps({**selected, 'checkpoint': kind+'.pt'}))
                if kind != 'context':
                    complete.add('train_'+kind)
            with patch.object(orchestrate, 'OUT', root):
                jobs = orchestrate.ready_confirmation_jobs(complete, set())
                names = {job[0] for job in jobs}
                self.assertEqual(len(names), 8)
                self.assertEqual(jobs[0][0], 'confirmation_full')
                self.assertNotIn('confirmation_context', names)
                self.assertEqual(orchestrate.ready_confirmation_jobs(complete, names), [])
                # A completed/active/pending job is never queued again after restart.
                complete |= names | {'train_context'}
                extra = orchestrate.ready_confirmation_jobs(complete, names)
                self.assertEqual([job[0] for job in extra], ['confirmation_context'])
                frozen = json.loads((root/'freeze_before_confirmation.json').read_text())
                self.assertEqual(frozen['status'], 'frozen')
                self.assertEqual(frozen['pending_models'], [])
                self.assertEqual(frozen['models']['full'], selected)
                names.add('confirmation_context')
                self.assertEqual(orchestrate.ready_confirmation_jobs(complete, names), [])


if __name__ == '__main__':
    unittest.main()
