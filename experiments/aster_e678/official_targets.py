"""Official target evaluation inputs and metadata; no fitting or calibration.

Input prompts are reused from existing prepared data. Target labels occur only
in the separate view/pair metadata; extra_inputs.jsonl has no answer labels.
"""
import json
import os
from collections import Counter
from pathlib import Path

NLI_INVENTORY = Path('/home/gcx/aster_next_20261109/inventory')
NLI_STAGE = 'nli4ct_official5500'
MEDPIC_STAGE = 'medpic_official467'


def _rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _lines(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary.replace(path)


def build(source_inputs, source_data_dir, output_shared_dir):
    """Return stage UIDs and file paths for full official target evaluation.

    source_inputs is a read-only mapping from uid to input record, such as
    load_inputs(SOURCE). source_data_dir is SOURCE/data. Only output_shared_dir
    is written. Existing u IDs are retained when their prompts match exactly.
    """
    data = Path(source_data_dir)
    output = Path(output_shared_dir) / 'official_targets'
    current_nli = _rows(data / 'nli4ct_views.jsonl')
    known = {}
    for view in current_nli:
        uid = view.get('uid', view['view_id'])
        for key in (view.get('case_id'), view.get('original_view_id'), view.get('id')):
            if key:
                known[key] = uid
    cases = [case for case in _rows(NLI_INVENTORY / 'nli4ct_all_cases.jsonl')
             if case['dataset_split'] == 'test']
    native_pairs = [pair for pair in _rows(NLI_INVENTORY / 'nli4ct_native_pairs.jsonl')
                    if pair['dataset_split'] == 'test']
    if len(cases) != 5500 or len(native_pairs) != 5000:
        raise ValueError('Expected the existing official 5500 NLI cases and 5000 native pairs')
    mapping, nli_views, extra_inputs = {}, [], []
    reused = 0
    for ordinal, case in enumerate(cases):
        key = case['case_id']
        uid = known.get(key)
        if uid is not None and source_inputs[uid]['prompt'] == case['prompt_text']:
            reused += 1
        else:
            uid = 'nli_official_' + format(ordinal, '05d')
            if uid in source_inputs and source_inputs[uid]['prompt'] != case['prompt_text']:
                raise ValueError('Conflicting source UID: ' + uid)
            extra_inputs.append(dict(uid=uid, view_id=uid, prompt=case['prompt_text'],
                                     stages=[NLI_STAGE], split='test', required_grads=False,
                                     dataset='nli4ct', original_case_id=key))
        mapping[key] = uid
        original = case['native_id'] == case['family']
        nli_views.append(dict(uid=uid, view_id=uid, original_view_id=key, id=key,
                              case_id=key, source_id=case['family_cluster_id'],
                              family=case['family'], native_id=case['native_id'],
                              split='test', evaluation_group=NLI_STAGE, task='nli4ct',
                              dataset='nli4ct', label=case['label'], original=original,
                              variant='full' if original else 'native_variant',
                              kind='original' if original else 'native_relation'))
    nli_pairs = []
    for pair in native_pairs:
        value = dict(pair)
        value.update(original_left=pair['left'], original_right=pair['right'],
                     left=mapping[pair['left']], right=mapping[pair['right']],
                     source_id='NLI4CT-2024:test:' + pair['family'], split='test',
                     kind='native_relation', subtype=pair.get('intervention'),
                     evaluation_group=NLI_STAGE)
        nli_pairs.append(value)
    medpic_views = []
    for original in _rows(data / 'medpic_views.jsonl'):
        if not original.get('original'):
            continue
        uid = original.get('uid', original['view_id'])
        if uid not in source_inputs:
            raise KeyError('Missing prepared MedPIC input: ' + uid)
        value = dict(original)
        value.update(uid=uid, view_id=uid, original_split=original['split'],
                     evaluation_group=MEDPIC_STAGE,
                     evaluation_population='official_all_includes_historical_development')
        medpic_views.append(value)
    medpic_ids = {view['uid'] for view in medpic_views}
    item_ids = {view['item_id'] for view in medpic_views}
    if len(medpic_views) != 2335 or len(item_ids) != 467:
        raise ValueError('Expected 467 prepared MedPIC items with 2335 original options')
    medpic_pairs = []
    for original in _rows(data / 'medpic_pairs.jsonl'):
        if original['left'] in medpic_ids and original['right'] in medpic_ids:
            value = dict(original)
            value.update(original_kind=original.get('kind'), kind='native_relation',
                         evaluation_group=MEDPIC_STAGE)
            medpic_pairs.append(value)
    stage_ids = {NLI_STAGE: [view['uid'] for view in nli_views],
                 MEDPIC_STAGE: [view['uid'] for view in medpic_views]}
    metadata = {}
    for stage, views, pairs in [(NLI_STAGE, nli_views, nli_pairs),
                                (MEDPIC_STAGE, medpic_views, medpic_pairs)]:
        paths = dict(views=str(output / (stage + '_views.jsonl')),
                     pairs=str(output / (stage + '_pairs.jsonl')))
        _lines(paths['views'], views)
        _lines(paths['pairs'], pairs)
        metadata[stage] = paths
    extra_path = output / 'extra_inputs.jsonl'
    mapping_path = output / 'nli_case_uid_mapping.json'
    _lines(extra_path, extra_inputs)
    _write(mapping_path, mapping)
    summary = dict(nli4ct=dict(total=len(nli_views), originals=sum(v['original'] for v in nli_views),
                               reused_inputs=reused, extra_inputs=len(extra_inputs),
                               native_relations=dict(Counter(p['relation'] for p in nli_pairs))),
                   medpic=dict(items=len(item_ids), options=len(medpic_views),
                               historical_option_splits=dict(Counter(v['original_split'] for v in medpic_views)),
                               native_relations=dict(Counter(p['relation'] for p in medpic_pairs)),
                               interpretation='Descriptive official population including historical development items'),
                   target_labels='metadata only; never used for training, calibration or parameter selection')
    result = dict(stage_ids=stage_ids, extra_inputs=str(extra_path), metadata=metadata,
                  mapping=str(mapping_path), summary=summary)
    _write(output / 'manifest.json', result)
    return result
