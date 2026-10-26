from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from confirmation_contract import DOMAINS, REVISIONS, digest_file, payload_digest, verify_frozen, freeze
from mega_confirmation_reader import load_confirmation_inputs


class ConfirmationInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        data = cls.root / 'test'; data.mkdir()
        documents = {}
        for i in range(440):
            name = 'symbolic_' + str(i)
            f = data / 'guideline_md' / name / (name+'.md')
            f.parent.mkdir(parents=True)
            f.write_text('Invented symbolic instruction group '+str(i))
            documents[str(f.relative_to(cls.root))] = digest_file(f)
        source = data / 'vignette_cases.jsonl'
        with source.open('w') as stream:
            for i in range(7458):
                stream.write(json.dumps({'id':i,'case_path_id':i % 440,'path_index':i // 440,
                    'filename':'symbolic_'+str(i % 440),'text':'Invented observation '+str(i),
                    'language':'en','answer':'HIDDEN_SENTINEL','gold_path':['HIDDEN_SENTINEL']})+'\n')
        code = cls.root / 'fake_code.py'; code.write_text('# synthetic code artifact\n')
        projection = cls.root / 'fake_projection.pt'; projection.write_bytes(b'synthetic artifact, not weights')
        cls.lock = {'schema_version':'aster.confirmation.v1','frozen':True,'frozen_at_utc':'synthetic-time',
            'trial_seeds':[0,1,2], 'seed_domains':{k:{'values_by_trial':{'0':0,'1':1,'2':2},
                'meaning':'Synthetic contract fixture, no model'} for k in DOMAINS},
            'methods':['method','baseline'],'primary_comparisons':[{'method':'method','reference':'baseline'}],
            'thresholds':{'effect':.02,'noninferiority':.02},'model_fingerprint':'synthetic-checkpoint',
            'model_files_sha256':{'synthetic_model':'a'*64},'method_config':{'templates':{}},
            'source_files_sha256':{str(code):digest_file(code)},
            'projection_files_sha256':{str(projection):digest_file(projection)},
            'seed_interpretation':'Report all trials; greedy duplicates are not independent replications',
            'datasets':{'MEGA-CDP':{'revision':REVISIONS['MEGA-CDP'],'rows':7458,
                    'input_sha256':digest_file(source),'guideline_files_sha256':documents},
                'MedPIC-Bench':{'revision':REVISIONS['MedPIC-Bench'],'rows':467,'input_sha256':'b'*64}}}
        cls.lock['lock_sha256'] = payload_digest(cls.lock)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_full_native_count_identity_groups_and_label_isolation(self):
        rows, manifest = load_confirmation_inputs(self.root, self.lock, 0)
        self.assertEqual(len(rows), 7458)
        self.assertEqual(len({x['source_cluster'] for x in manifest['selected']}), 440)
        self.assertEqual(len({i.sample_id for i,t in rows}), 7458)
        self.assertTrue(all(set(asdict(t)) == {'guideline','vignette','language'} for i,t in rows))
        self.assertNotIn('HIDDEN_SENTINEL', json.dumps([asdict(t) for i,t in rows]))
        self.assertNotIn('HIDDEN_SENTINEL', json.dumps(manifest))
        self.assertFalse(manifest['reference_files_read'])

    def test_unfrozen_or_modified_contract_rejects_before_any_file_open(self):
        for mutation in [{'frozen':False}, {'methods':['changed']}]:
            lock = {**self.lock, **mutation}
            with patch.object(Path,'open',side_effect=AssertionError('Dataset opened too early')):
                with self.assertRaises(ValueError):
                    load_confirmation_inputs('/nonexistent',lock,0)

    def test_all_seeds_are_required_and_unknown_seed_rejected(self):
        for seed in [0,1,2]: verify_frozen(self.lock,seed,'MEGA-CDP')
        with self.assertRaises(ValueError):verify_frozen(self.lock,3,'MEGA-CDP')
        bad=deepcopy(self.lock);bad['seed_domains']['projection']['values_by_trial'].pop('2')
        bad['lock_sha256']=payload_digest(bad)
        with self.assertRaises(ValueError):verify_frozen(bad,0,'MEGA-CDP')

    def test_freeze_cannot_use_metadata_audit_as_clinical_development_results(self):
        spec=deepcopy(self.lock);spec['frozen']=False;spec.pop('lock_sha256')
        model=self.root/'model_manifest.json'
        model.write_text(json.dumps({'process_state':'exited','exit_code':0,
                                    'model_files_sha256':spec['model_files_sha256']}))
        spec['model_manifest_evidence']={'path':str(model),'sha256':digest_file(model)}
        audit=self.root/'input_audit.json';audit.write_text(json.dumps({'status':'complete_verified'}))
        spec['development_evidence']={k:{'path':str(audit),'sha256':digest_file(audit)}
                                      for k in ['summary','power_analysis']}
        with self.assertRaises(ValueError):freeze(spec,self.root/'must_not_freeze.json')
        self.assertFalse((self.root/'must_not_freeze.json').exists())

    def test_changed_source_and_missing_seed_are_not_silently_accepted(self):
        bad=deepcopy(self.lock)
        bad['datasets']['MEGA-CDP']['input_sha256']='0'*64
        bad['lock_sha256']=payload_digest(bad)
        with self.assertRaises(ValueError):load_confirmation_inputs(self.root,bad,1)


if __name__=='__main__':unittest.main()
