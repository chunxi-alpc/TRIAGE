"""Storage-only adapter for the official adaptive SEKA projection builder.

The official attention kernel, FP32 dtype, prompts, examples, two forwards,
span indexing, layer order, sums, and acceptance threshold are unchanged.
Full attention weights are discarded immediately after the required scalar is
read. Hidden states/logits/cache outputs are otherwise preserved.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
import traceback


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def argument(arguments,name):
    for index,value in enumerate(arguments):
        if value==name:
            return arguments[index+1]
        if value.startswith(name+'='):
            return value.split('=',1)[1]
    return None


def guard_model(builder,torch):
    require(builder.model.__class__.__name__=='Qwen3ForCausalLM','This storage adapter is validated for Qwen3 only')
    require(not builder.model.training,'The official builder must use eval mode')
    dtypes={str(p.dtype) for p in builder.model.parameters() if p.is_floating_point()}
    require(dtypes=={'torch.float32'},'Do not change the official single-GPU FP32 calibration dtype')


def sums_from_original(builder,outputs,inputs,indices):
    query_idx=inputs.input_ids.shape[1]-1
    return {layer:outputs.attentions[layer][0,:,query_idx,indices].sum().item() for layer in builder.layers}


def decision(builder,h,hp):
    total_h=0.0
    total_hp=0.0
    for layer in builder.layers:
        total_h+=h[layer]
        total_hp+=hp[layer]
    included=(total_hp>0) if total_h==0 else total_hp/total_h>=builder.attention_threshold
    return included,total_h,total_hp


def storage_check(builder,text_h,text_hp,indices_h,indices_hp,torch,trace):
    guard_model(builder,torch)
    inputs_h=builder.tokenizer(text_h,return_tensors='pt',add_special_tokens=False)
    inputs_hp=builder.tokenizer(text_hp,return_tensors='pt',add_special_tokens=False)
    device=builder.get_model_device()
    inputs_h=inputs_h.to(device)
    inputs_hp=inputs_hp.to(device)
    selected=set(builder.layers)

    def forward(inputs,indices):
        query_idx=inputs.input_ids.shape[1]-1
        sums={}
        handles=[]
        def hook_for(layer):
            def collect(module,args,output):
                require(isinstance(output,tuple) and len(output)==2,
                        'Qwen3Attention must return (attn_output, attn_weights)')
                activation,weights=output
                if layer in selected:
                    require(isinstance(weights,torch.Tensor) and weights.ndim==4,
                            'Selected attention weights unavailable; do not silently change attention implementation')
                    # Exactly the same GPU dtype, indexing and sum as upstream.
                    sums[layer]=weights[0,:,query_idx,indices].sum().item()
                # attn_output already includes attention @ V and o_proj.
                # No later calculation consumes the returned weights except
                # the upstream scalar check replaced by this collector.
                return activation,None
            return collect
        try:
            for layer,module in enumerate(builder.model.model.layers):
                handles.append(module.self_attn.register_forward_hook(hook_for(layer)))
            output=builder.model(**inputs,output_attentions=True,output_hidden_states=True)
        finally:
            for handle in handles:
                handle.remove()
        require(set(sums)==selected,'Incomplete selected-layer attention sums')
        require(output.hidden_states is not None,'Hidden states required by official extract_keys are absent')
        require(output.attentions is not None and all(value is None for value in output.attentions),
                'Attention weights were unexpectedly retained')
        return output,sums

    outputs_h,h=forward(inputs_h,indices_h)
    outputs_hp,hp=forward(inputs_hp,indices_hp)
    included,total_h,total_hp=decision(builder,h,hp)
    record=dict(text_h_sha256=text_sha(text_h),text_hp_sha256=text_sha(text_hp),
                tokens_h=int(inputs_h.input_ids.shape[1]),tokens_hp=int(inputs_hp.input_ids.shape[1]),
                answer_indices_h=list(indices_h),answer_indices_hp=list(indices_hp),
                selected_layers=list(builder.layers),attention_h=h,attention_hp=hp,
                total_attention_h=total_h,total_attention_hp=total_hp,
                ratio=None if total_h==0 else total_hp/total_h,
                attention_threshold=builder.attention_threshold,included=bool(included),
                discarded_full_attention_weights=True)
    if trace is not None:
        trace.write(json.dumps(record,sort_keys=True)+'\n')
        trace.flush()
    builder._storage_last_record=record
    return included,outputs_h,outputs_hp


def tensor_digest(tensor,torch):
    value=tensor.detach().contiguous().cpu()
    header=json.dumps({'shape':list(value.shape),'dtype':str(value.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header+b'\x00'+value.view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot(builder,text_h,text_hp,indices_h,indices_hp,outputs_h,outputs_hp,torch):
    # CPU snapshots permit releasing the original full attention tensors before
    # the optimized pair, so validation does not hold four full GPU outputs.
    result={}
    for label,text,indices,outputs in [('h',text_h,indices_h,outputs_h),('hp',text_hp,indices_hp,outputs_hp)]:
        keys=builder.extract_keys(builder.model,builder.tokenizer,text,indices,builder.layers,builder.feature,outputs=outputs)
        result[label]={'keys':[key.detach().cpu().clone() for key in keys],
                       'logits':outputs.logits.detach().cpu().clone()}
    return result


def tensor_comparison(left,right,torch):
    require(left.shape==right.shape and left.dtype==right.dtype,'Validation tensor shape/dtype differs')
    return dict(shape=list(left.shape),dtype=str(left.dtype),exact_equal=bool(torch.equal(left,right)),
                max_abs_error=float((left-right).abs().max()),
                original_sha256=tensor_digest(left,torch),optimized_sha256=tensor_digest(right,torch))


def validate_one(builder,original_check,report,torch,trace):
    guard_model(builder,torch)
    for example_index,example in enumerate(builder.iter_examples()):
        for triplet_index,(context,question,answer) in enumerate(builder.get_triplets(example)):
            text_h,text_hp=builder.assemble_texts(context,question)
            indices_h=builder.span_token_indices(builder.tokenizer,text_h,answer)
            indices_hp=builder.span_token_indices(builder.tokenizer,text_hp,answer)
            if not (indices_h and indices_hp):
                continue
            # First eligible source triplet, regardless of acceptance outcome.
            valid,original_h,original_hp=original_check(builder,text_h,text_hp,indices_h,indices_hp)
            inputs_h=builder.tokenizer(text_h,return_tensors='pt',add_special_tokens=False)
            inputs_hp=builder.tokenizer(text_hp,return_tensors='pt',add_special_tokens=False)
            h=sums_from_original(builder,original_h,inputs_h,indices_h)
            hp=sums_from_original(builder,original_hp,inputs_hp,indices_hp)
            reference=snapshot(builder,text_h,text_hp,indices_h,indices_hp,original_h,original_hp,torch)
            del original_h,original_hp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimized_valid,optimized_h,optimized_hp=storage_check(builder,text_h,text_hp,indices_h,indices_hp,torch,trace)
            observed=snapshot(builder,text_h,text_hp,indices_h,indices_hp,optimized_h,optimized_hp,torch)
            detail=builder._storage_last_record
            comparisons={}
            for label in ('h','hp'):
                require(len(reference[label]['keys'])==len(observed[label]['keys']),'Key-head count differs')
                comparisons[label]={'logits':tensor_comparison(reference[label]['logits'],observed[label]['logits'],torch),
                    'keys':[tensor_comparison(a,b,torch) for a,b in zip(reference[label]['keys'],observed[label]['keys'])]}
            attention_equal=h==detail['attention_h'] and hp==detail['attention_hp']
            tensor_equal=all(value['logits']['exact_equal'] and all(k['exact_equal'] for k in value['keys']) for value in comparisons.values())
            report.update(validation_case=dict(example_index=example_index,triplet_index=triplet_index,
                text_h_sha256=text_sha(text_h),text_hp_sha256=text_sha(text_hp),
                original_included=bool(valid),optimized_included=bool(optimized_valid),
                original_attention_h=h,original_attention_hp=hp,
                optimized_attention_h=detail['attention_h'],optimized_attention_hp=detail['attention_hp'],
                selected_attention_sums_exact_equal=attention_equal,
                comparisons=comparisons),selected_layers=list(builder.layers),
                attention_threshold=builder.attention_threshold,model_dtype='torch.float32',
                verification_forward_calls=4)
            require(bool(valid)==bool(optimized_valid) and attention_equal and tensor_equal,
                    'Storage adapter did not preserve inclusion, selected attention sums, keys and logits exactly')
            report['equivalence_verified']=True
            return
    raise ValueError('No eligible source triplet found for equivalence verification')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['validate','build'],required=True)
    parser.add_argument('--upstream-module',required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--validation-report',type=Path)
    parser.add_argument('upstream_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.upstream_args[:1]==['--']:
        args.upstream_args=args.upstream_args[1:]
    args.report=args.report.expanduser().resolve()
    require(not args.report.exists(),'Use a new storage-adapter report path')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report=dict(status='running',mode=args.mode,wrapper_sha256=sha(__file__),
        upstream_module=args.upstream_module,upstream_args=args.upstream_args,
        adaptation_label='official adaptive builder with attention-output storage optimization',
        arithmetic_changed=False,dtype_changed=False,samples_truncated_or_dropped_by_adapter=False,
        threshold_logic_changed=False,standard_algorithm_reproduction_complete=False)
    code=0
    base=None
    original_check=None
    original_run=None
    trace=None
    try:
        import torch
        import transformers
        from src.model.projection_builder_base_adapsvd import ProjectionBuilderBase
        base=ProjectionBuilderBase
        original_check=base._check_attention_increase
        original_run=base.run
        report.update(torch=torch.__version__,transformers=transformers.__version__,
                      upstream_builder_sha256=sha(inspect.getfile(base)),
                      upstream_checker_sha256=text_sha(inspect.getsource(original_check)))
        model_arg=argument(args.upstream_args,'--model')
        require(model_arg is not None,'Upstream --model argument required')
        model=Path(model_arg).expanduser().resolve()
        report['model_path']=str(model)
        report['model_config_sha256']=sha(model/'config.json')
        data_arg=argument(args.upstream_args,'--data')
        report['calibration_data_sha256']=sha(Path(data_arg).expanduser().resolve()) if data_arg and Path(data_arg).is_file() else None
        trace_path=args.report.with_suffix('.selection.jsonl')
        trace=trace_path.open('x')
        report['selection_trace_path']=str(trace_path)
        if args.mode=='validate':
            def validation_run(self,output_dir):
                validate_one(self,original_check,report,torch,trace)
            base.run=validation_run
        else:
            require(args.validation_report is not None,'A successful equivalence report is required before the full build')
            previous=json.loads(args.validation_report.read_text())
            require(previous.get('status')=='complete_verified' and previous.get('equivalence_verified') is True,
                    'Equivalence validation was not successful')
            for key in ['wrapper_sha256','upstream_builder_sha256','upstream_checker_sha256','model_path',
                        'model_config_sha256','calibration_data_sha256','torch','transformers','upstream_module']:
                require(previous.get(key)==report.get(key),'Validation provenance differs: '+key)
            report['validation_report_sha256']=sha(args.validation_report)
            def optimized_check(self,*values):
                require(list(self.layers)==previous['selected_layers'],'Selected layers differ from validated configuration')
                require(self.attention_threshold==previous['attention_threshold'],'Acceptance threshold differs from validation')
                return storage_check(self,*values,torch,trace)
            base._check_attention_increase=optimized_check
        sys.argv=[args.upstream_module]+args.upstream_args
        runpy.run_module(args.upstream_module,run_name='__main__')
        require(args.mode!='validate' or report.get('equivalence_verified') is True,'Validation entrypoint was not executed')
        report['status']='complete_verified' if args.mode=='validate' else 'complete_pending_projection_audit'
    except BaseException as error:
        code=1
        report['status']='failed'
        report['error']=type(error).__name__+': '+str(error)
        traceback.print_exc()
    finally:
        if base is not None:
            if original_check is not None:
                base._check_attention_increase=original_check
            if original_run is not None:
                base.run=original_run
        if trace is not None:
            trace.close()
            report['selection_trace_sha256']=sha(trace_path)
        report['elapsed_seconds']=time.monotonic()-started
        with args.report.open('x') as handle:
            json.dump(report,handle,indent=2)
            handle.write('\n')
    print(json.dumps({'status':report['status'],'mode':args.mode,'arithmetic_changed':False,'dtype_changed':False}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
