"""Three server-only isolation tests; synthetic files, no model/data/GPU access.

Set ASTER_RUNNER_PATH to the coordinator's current run_official_job.py.
Only input_fingerprints is imported/called; the runner entrypoint is not run.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


RUNNER_PATH = Path(os.environ.get('ASTER_RUNNER_PATH', Path(__file__).resolve().parents[1]/'run_official_job.py'))
SPEC = importlib.util.spec_from_file_location('aster_runner_under_test', RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FingerprintIsolation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aster_fingerprint_isolation_')
        self.root = Path(self.temporary.name)
        self.source = self.root/'experiment'
        self.source.mkdir()
        self.output = self.source/'job_output'
        self.output.mkdir()
        self.model = self.root/'model'
        self.model.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def file(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def hf_link(self, snapshot, relative, text):
        # Match HF's file links into a sibling blobs directory. Contents are
        # invented labels and sentences; no actual clinical data is accessed.
        blob = self.file(snapshot.parent.parent/'blobs'/hashlib.sha256(text.encode()).hexdigest(), text)
        link = snapshot/relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link, blob.resolve()

    def state(self):
        return dict(input_files_sha256={}, input_file_origins={}, unresolved_input_references=[])

    def fingerprint(self, arguments, forbidden=()):
        state = self.state()
        original_sha = RUNNER.sha
        forbidden = {path.resolve() for path in forbidden}
        def guarded_hash(path):
            self.assertNotIn(Path(path).resolve(), forbidden, 'Excluded data bytes were read by fingerprinting')
            return original_sha(path)
        with patch.object(RUNNER, 'sha', side_effect=guarded_hash):
            RUNNER.input_fingerprints(arguments, self.source, self.output, self.model, state)
        return state

    def test_source_scope_and_rejected_symlink_targets(self):
        code = self.file(self.source/'reader.py', '# synthetic reader\n')
        template = self.file(self.source/'prompt.jinja', 'Synthetic prompt {{ item }}\n')
        runtime = self.file(self.source/'runtime'/'hidden_reference.txt', 'invented hidden answer')
        git_file = self.file(self.source/'.git'/'hook.py', '# excluded metadata\n')
        output_file = self.file(self.output/'prediction.txt', 'invented prediction')
        state = self.fingerprint(['--adapter-dir',str(self.source),'--reader-dir',str(self.source)],
                                 [runtime,git_file,output_file])
        self.assertEqual(set(state['input_files_sha256']),{str(code.resolve()),str(template.resolve())})
        outside = self.file(self.root/'hf'/'blobs'/'synthetic_blob', 'invented held-out bytes')
        for target in [runtime,outside,runtime.parent,outside.parent]:
            with self.subTest(target=target.name):
                alias = self.source/'linked_template.txt'
                alias.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ValueError,'Source/template symlink'):
                        self.fingerprint(['--reader-dir',str(self.source)],[target])
                finally:
                    alias.unlink()

    def test_hf_snapshot_only_hashes_development_inputs_and_guidelines(self):
        snapshot = self.root/'hf'/'snapshots'/'synthetic_revision'
        snapshot.mkdir(parents=True)
        _, vignette = self.hf_link(snapshot,'train_val/vignette_cases.jsonl','{"case_id":"fixture","text":"synthetic"}\n')
        _, guideline = self.hf_link(snapshot,'train_val/guideline_md/example.md','# Invented guideline\n')
        _, reference = self.hf_link(snapshot,'train_val/reference_paths.jsonl','{"fixture":"invented reference path"}\n')
        _, answers = self.hf_link(snapshot,'train_val/reference_answers.jsonl','{"fixture":"invented answer"}\n')
        _, test_case = self.hf_link(snapshot,'test/vignette_cases.jsonl','{"fixture":"invented hidden test"}\n')
        _, test_guideline = self.hf_link(snapshot,'test/guideline_md/example.md','# Invented hidden guideline\n')
        state = self.fingerprint(['--dataset-root',str(snapshot)],[reference,answers,test_case,test_guideline])
        self.assertEqual(set(state['input_files_sha256']),{str(vignette),str(guideline)})
        self.assertEqual(state['unresolved_input_references'],[])

    def test_missing_train_val_never_falls_back_to_whole_snapshot(self):
        for missing in ['train_val','vignette_cases.jsonl','guideline_md']:
            with self.subTest(missing=missing):
                snapshot = self.root/missing.replace('.','_')/'snapshots'/'synthetic_revision'
                snapshot.mkdir(parents=True)
                _, hidden = self.hf_link(snapshot,'test/reference_answers.jsonl','{"fixture":"never read '+missing+'"}\n')
                if missing != 'train_val':
                    (snapshot/'train_val').mkdir()
                    if missing != 'vignette_cases.jsonl':
                        self.hf_link(snapshot,'train_val/vignette_cases.jsonl','{"fixture":"allowed synthetic input"}\n')
                    if missing != 'guideline_md':
                        self.hf_link(snapshot,'train_val/guideline_md/example.md','# Allowed synthetic guideline\n')
                with self.assertRaisesRegex(FileNotFoundError,'MEGA dataset root requires'):
                    self.fingerprint(['--dataset-root',str(snapshot)],[hidden])


if __name__ == '__main__':
    unittest.main()
