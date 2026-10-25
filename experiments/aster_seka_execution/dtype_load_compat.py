"""Override only AutoModelForCausalLM's load dtype for a matched control.

No upstream source or weight file is edited. The original loader is restored
after the entrypoint exits. Actual weights/data/source hashes remain in the
outer run_official_job manifest; this report records the load adaptation and
the actual model/generation configuration. Run on the research server only.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--torch-dtype', choices=['float32','bfloat16','float16'], required=True)
    parser.add_argument('--upstream-module', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('upstream_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.upstream_args[:1] == ['--']:
        args.upstream_args = args.upstream_args[1:]
    args.report = args.report.expanduser().resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', adaptation_label='Explicit model-load dtype override for matched numerical control',
                  requested_torch_dtype=args.torch_dtype, upstream_module=args.upstream_module,
                  upstream_args=args.upstream_args, wrapper_sha256=sha(__file__), loaded_models=[],
                  upstream_source_modified=False, checkpoint_files_modified=False,
                  attention_or_generation_operators_modified=False,
                  extra_model_forwards_from_wrapper=0,
                  weight_fingerprint_location='Outer run_official_job.py run_manifest.json model_files_sha256; not recomputed inside this wrapper',
                  benchmark_or_numerical_equivalence_verified=False)
    with args.report.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    started = time.monotonic()
    code = 0
    auto_class = None
    previous_descriptor = None
    originally_own_attribute = False
    patched = False
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        auto_class = AutoModelForCausalLM
        original_loader = auto_class.from_pretrained
        originally_own_attribute = 'from_pretrained' in auto_class.__dict__
        previous_descriptor = auto_class.__dict__.get('from_pretrained')
        wanted = getattr(torch, args.torch_dtype)
        report.update(torch=torch.__version__, transformers=transformers.__version__,
                      torch_default_dtype_before_load=str(torch.get_default_dtype()),
                      loader_source_path=inspect.getfile(original_loader),
                      loader_source_sha256=sha(inspect.getfile(original_loader)))
        module_spec = importlib.util.find_spec(args.upstream_module)
        if module_spec is None or module_spec.origin is None:
            raise ValueError('Upstream module source was not found')
        report['upstream_module_path'] = str(Path(module_spec.origin).resolve())
        report['upstream_module_sha256'] = sha(module_spec.origin)

        def load_with_dtype(cls, *model_args, **kwargs):
            original_dtype = kwargs.get('torch_dtype')
            record = dict(original_torch_dtype_keyword=str(original_dtype) if original_dtype is not None else None,
                          original_torch_dtype_keyword_present='torch_dtype' in kwargs,
                          overridden_torch_dtype=str(wanted),
                          device_map=str(kwargs.get('device_map')),
                          requested_attn_implementation=kwargs.get('attn_implementation'),
                          load_status='started')
            report['loaded_models'].append(record)
            # Do not modify torch's global default dtype, tokenizer settings,
            # generation config, attention implementation, model operators, or
            # checkpoint files. This replaces only the model loader argument.
            kwargs['torch_dtype'] = wanted
            model = original_loader(*model_args, **kwargs)
            location = model_args[0] if model_args else kwargs.get('pretrained_model_name_or_path')
            record['requested_checkpoint'] = str(location)
            checkpoint = Path(str(location)).expanduser().resolve()
            record['checkpoint_path'] = str(checkpoint)
            record['checkpoint_config_files_sha256'] = {
                name:sha(checkpoint/name) for name in ['config.json','generation_config.json']
                if (checkpoint/name).is_file()}
            dtypes = {}
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    key = str(parameter.dtype)
                    dtypes[key] = dtypes.get(key,0)+parameter.numel()
            record.update(floating_parameter_dtype_element_counts=dtypes,
                          observed_model_config_name_or_path=str(getattr(model.config,'_name_or_path','')),
                          observed_attn_implementation=str(getattr(model.config,'_attn_implementation',None)),
                          model_config=model.config.to_dict(),
                          generation_config=model.generation_config.to_dict() if model.generation_config is not None else None)
            if set(dtypes) != {str(wanted)}:
                raise ValueError('Actual floating parameter dtypes differ from requested load override: '+str(dtypes))
            record['load_status'] = 'complete_verified_dtype_only'
            return model

        auto_class.from_pretrained = classmethod(load_with_dtype)
        patched = True
        sys.argv = [args.upstream_module]+args.upstream_args
        runpy.run_module(args.upstream_module,run_name='__main__')
        report['status'] = 'executed_pending_independent_metric_verification'
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code in [None,0]:
            report['status'] = 'executed_pending_independent_metric_verification'
        else:
            code = error.code if isinstance(error,SystemExit) and isinstance(error.code,int) and error.code else 1
            report.update(status='failed',error=type(error).__name__+': '+str(error))
            traceback.print_exc()
    finally:
        if patched:
            if originally_own_attribute:
                auto_class.from_pretrained = previous_descriptor
            else:
                delattr(auto_class,'from_pretrained')
        report['loader_restored'] = patched
        report['elapsed_seconds'] = time.monotonic()-started
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
