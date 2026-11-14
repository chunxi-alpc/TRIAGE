"""Frozen Qwen geometry/environment guards; stdlib only and no model imports.

The caller persists returned audits before refusing incompatible execution.
Config inputs must be resolved AutoConfig/model.config dictionaries, since a
checkpoint's raw config.json may omit defaults recorded in the old manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CRITICAL_CONFIG_FIELDS = (
    'model_type', 'architectures', 'vocab_size', 'hidden_size',
    'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'head_dim', 'hidden_act', 'rms_norm_eps',
    'max_position_embeddings', 'rope_theta', 'rope_scaling',
    'attention_bias', 'attention_dropout', 'use_sliding_window',
    'sliding_window', 'max_window_layers', 'layer_types', 'tie_word_embeddings',
)


def load_frozen_manifest(path, protocol):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol['model_manifest_sha256']:
        raise ValueError('Original model manifest bytes differ from the frozen protocol')
    original = json.loads(raw)
    if original['model_revision'] != protocol['model_revision']:
        raise ValueError('Original model revision differs from the frozen protocol')
    required = protocol['model_compatibility']['required_dependency_versions']
    if required != {name: original['versions'][name] for name in ('torch', 'transformers')}:
        raise ValueError('Core dependency requirements differ from the original manifest')
    # This operator implements full causal Qwen3 GQA only. The frozen source
    # manifest supplies the values; these assertions state its supported scope.
    config = original['config']
    if (config['use_sliding_window'] is not False or config['sliding_window'] is not None
            or config['layer_types'] != ['full_attention'] * config['num_hidden_layers']
            or config['attention_dropout'] != 0.0 or config['rope_scaling'] is not None):
        raise ValueError('Reference configuration exceeds the validated full-causal geometry scope')
    return original


def config_audit(actual_config, original_manifest, protocol):
    required = list(CRITICAL_CONFIG_FIELDS)
    if protocol['model_compatibility']['critical_config_fields'] != required:
        raise ValueError('Frozen configuration check fields changed')
    expected = {key: original_manifest['config'][key] for key in required}
    observed = {key: actual_config.get(key) for key in required}
    differences = {
        key: dict(expected=expected[key], observed=observed[key],
                  missing_from_observed=key not in actual_config)
        for key in required if key not in actual_config or observed[key] != expected[key]
    }
    return dict(compatible=not differences, differences=differences,
                expected=expected, observed=observed, required_fields=required,
                comparison='resolved config values, exact equality')


def environment_audit(observed_versions, original_manifest, protocol):
    required = protocol['model_compatibility']['required_dependency_versions']
    expected = original_manifest['versions']
    if required != {name: expected[name] for name in ('torch', 'transformers')}:
        raise ValueError('Core dependency requirements differ from the original manifest')
    differences = {name: dict(expected=value, observed=observed_versions.get(name))
                   for name, value in expected.items() if observed_versions.get(name) != value}
    blocking = {name: value for name, value in differences.items() if name in required}
    return dict(compatible=not blocking, required_versions=required,
                original_versions=expected, observed_versions=observed_versions,
                differences=differences, blocking_differences=blocking,
                policy='Exact installed-distribution Torch/Transformers versions; no local-suffix normalization or unvalidated-version override',
                remaining_scope='CUDA build/driver versions were not in the original manifest; real setup records the current Torch CUDA build and GPU model. Driver equality is not claimed. Passing this gate is not a real-model numerical validation.')
