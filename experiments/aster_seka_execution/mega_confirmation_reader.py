"""Read full native MEGA test inputs only after the method/seed freeze.

Identity remains outside the method input. No test reference/path/tree is opened.
"""
from dataclasses import asdict
from pathlib import Path
import json
import os
import re

from confirmation_contract import verify_frozen, verify_executable_files, digest_file
from mega_trainval_reader import MethodInput, TaskIdentity, TEMPLATES, value_digest


def snapshot_file(root, relative):
    root = root.resolve()
    path = Path(os.path.abspath(root / relative))
    if not path.is_relative_to(root / 'test'):
        raise ValueError('Only declared native test input paths are allowed')
    if (root / 'test').is_symlink() or not path.parent.resolve(strict=True).is_relative_to(root / 'test'):
        raise ValueError('Test input parent escaped the snapshot')
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(root / 'test') and not path.is_symlink() and resolved.is_file():
        return path
    # The actual HF snapshot legitimately links files into its own object store.
    blobs = root.parent.parent / 'blobs'
    if (root.parent.name != 'snapshots' or not path.is_symlink() or blobs.is_symlink()
            or resolved.parent != blobs or not resolved.is_file()
            or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', resolved.name)):
        raise ValueError('Unexpected snapshot file link')
    return path


def load_confirmation_inputs(root, lock, seed):
    """Materialize the complete 7458-row input contract, without exposing labels.

    No partial prefix is returned if the complete count/uniqueness check fails.
    This is data preparation, never a claim that a model evaluated all cases.
    """
    verify_frozen(lock, seed, 'MEGA-CDP')
    verify_executable_files(lock)
    root = Path(root)
    data = lock['datasets']['MEGA-CDP']
    source = snapshot_file(root, 'test/vignette_cases.jsonl')
    if digest_file(source) != data['input_sha256']:
        raise ValueError('Native vignette file differs from the frozen fingerprint')
    rows, identities, documents, files = [], set(), {}, {}
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            filename, vignette = raw.get('filename'), raw.get('text')
            case = raw.get('case_path_id', raw.get('guideline_id'))
            index = raw.get('path_index', raw.get('path_id'))
            language = raw.get('language', 'en')
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or filename in {'', '.', '..'} or '\\' in filename or '\0' in filename
                    or not isinstance(vignette, str) or not isinstance(language, str)
                    or case is None or index is None):
                raise ValueError('Invalid official model-input schema')
            identity = TaskIdentity(raw.get('id', len(rows)+1), case, index, filename)
            if identity.sample_id in identities:
                raise ValueError('Duplicate native task; no silent dropping')
            identities.add(identity.sample_id)
            if filename not in documents:
                relative = 'test/guideline_md/' + filename + '/' + filename + '.md'
                f = snapshot_file(root, relative)
                fingerprint = digest_file(f)
                expected = data.get('guideline_files_sha256', {}).get(relative)
                if expected is None or fingerprint != expected:
                    raise ValueError('Guideline file is missing from the frozen content inventory')
                documents[filename] = f.read_text(encoding='utf-8')
                files[relative] = fingerprint
            task = MethodInput(documents[filename], vignette, language)
            rows.append((identity, task))
    if len(rows) != 7458:
        raise ValueError('Native confirmation requires every one of the 7458 released cases')
    selected = []
    for identity, task in rows:
        relative = 'test/guideline_md/' + identity.filename + '/' + identity.filename + '.md'
        document = files[relative]
        selected.append({'identity':asdict(identity), 'sample_id':identity.sample_id,
                         'source_cluster':document,
                         'input_sha256':value_digest([document,task.vignette,task.language])})
    manifest = {'schema_version':'aster.mega_confirmation_inputs.v1','status':'complete_verified',
        'scope':'Full input identity and fingerprints only; no model inference', 'split':'test',
        'expected_dataset_revision':data['revision'],'source_file_sha256':data['input_sha256'],
        'lock_sha256':lock['lock_sha256'],'trial_seed':seed,
        'seed_domains':{k:v['values_by_trial'][str(seed)] for k,v in lock['seed_domains'].items()},
        'expected_cases':7458,'selected':selected,'snapshot_guideline_files':files,
        'templates':lock['method_config']['templates'],
        'source_group_definition':'identical original guideline SHA256; external editions/translation overlap unresolved',
        'reference_files_read':False,'test_inputs_read_after_method_freeze':True,
        'native_input_order_rng_used':False,'model_seed_application':'Not performed by this data reader',
        'method_input_fields':['guideline','vignette','language'],
        'raw_input_text_published':False,'clinical_effect_claim':False}
    return rows, manifest
