"""CPU reproduction of LoReFT Householder rotation collapse under AdamW decay.
Uses random synthetic hidden states and a frozen linear readout, not task evaluation.
"""
import inspect,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path('/home/gcx/aster-e5-gaps-20261112')
VENDOR=Path('/home/gcx/aster_cr_repair_v2_run_20261108/vendor/python')
sys.path.insert(0,str(VENDOR))
from pyreft import LoreftIntervention

def snapshot(module,hidden,readout,labels):
    original=module.rotate_layer.parametrizations.weight.original
    with torch.no_grad():
        rotation=module.rotate_layer.weight
        output=module(hidden)
        return dict(rotation_norm=float(rotation.norm()),
            orthogonality_error=float((rotation.T@rotation-torch.eye(rotation.shape[1])).abs().max()),
            raw_diagonal=original.diagonal().tolist(),
            cast_diagonal=original.diagonal().int().tolist(),
            output_delta_max_abs=float((output-hidden).abs().max()),
            output_delta_norm=float((output-hidden).norm()),
            loss=float(F.binary_cross_entropy_with_logits(output@readout,labels)))

def reproduce(dim,decay_rotation):
    torch.manual_seed(20261108)
    module=LoreftIntervention(embed_dim=dim,low_rank_dimension=1,dtype=torch.float32)
    hidden=torch.randn(8,dim)
    readout=torch.randn(dim)/(dim**0.5)
    labels=torch.tensor([0,1,0,1,1,0,1,0],dtype=torch.float32)
    rotation_params=list(module.rotate_layer.parameters())
    rot_ids={id(p) for p in rotation_params}
    other_params=[p for p in module.parameters() if id(p) not in rot_ids]
    optimizer=torch.optim.AdamW([
        dict(params=rotation_params,weight_decay=0.01 if decay_rotation else 0.0),
        dict(params=other_params,weight_decay=0.01)
    ],lr=0.01)
    before=snapshot(module,hidden,readout,labels)
    steps=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(module(hidden)@readout,labels)
        loss.backward()
        original=module.rotate_layer.parametrizations.weight.original
        grad=dict(rotation_grad_norm=float(original.grad.norm()),
                  rotation_diag_grad=original.grad.diagonal().tolist(),
                  learned_weight_grad_norm=float(module.learned_source.weight.grad.norm()),
                  learned_bias_grad_norm=float(module.learned_source.bias.grad.norm()))
        torch.nn.utils.clip_grad_norm_(list(module.parameters()),1.0)
        optimizer.step()
        steps.append(dict(step=step+1,gradient_before_step=grad,after=snapshot(module,hidden,readout,labels)))
    return dict(embed_dim=dim,rank=1,lr=0.01,
        rotation_weight_decay=0.01 if decay_rotation else 0.0,
        learned_source_weight_decay=0.01,
        initial=before,steps=steps)

def main():
    torch.set_num_threads(4)
    runs=[reproduce(dim,decay) for dim in [32,4096] for decay in [True,False]]
    from torch.nn.utils.parametrizations import _Orthogonal
    source=inspect.getsource(_Orthogonal.forward)
    out=ROOT/'reports/aster_e5_gaps_20261112/loreft_effectiveness/cpu_reproduction.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    data=dict(scope='CPU diagnostic on synthetic hidden states; not task performance or a trained baseline.',
        torch_version=torch.__version__,pyreft_vendor=str(VENDOR/'pyreft'),
        loss='binary_cross_entropy_with_logits with frozen random linear readout',
        seed=20261108,batch_size=8,runs=runs,
        torch_orthogonal_forward_source=source,
        conclusion='AdamW decay moves the non-differentiable Householder sign diagonal from -1 toward 0. The integer cast becomes zero and collapses the projected rotation. Excluding only rotation parameters from decay preserves it.',
        historical_checkpoint='/home/gcx/aster_cr_repair_v2_run_20261108/qwen/baselines/loreft_lr0.01.pt')
    with out.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(out),results=[dict(dim=x['embed_dim'],decay=x['rotation_weight_decay'],
        first_rotation_norm=x['steps'][0]['after']['rotation_norm'],
        first_delta=x['steps'][0]['after']['output_delta_max_abs'],
        second_grad_norm=x['steps'][1]['gradient_before_step']['rotation_grad_norm']) for x in runs])))

if __name__=='__main__': main()
