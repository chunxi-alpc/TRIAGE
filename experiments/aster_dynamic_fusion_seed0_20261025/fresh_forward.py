"""Fresh matched-path controls on the unchanged 800 development records."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'aster_action_mass_seed0_20261025'))
import run

def configs():
    out=[('original',{})]
    for layers in ('all','last_half'):
        out.append((f'zero_{layers}',dict(mode='action_bias',lam=0.,layers=layers)))
    out += [('mass_zero_control',dict(mode='mass',lam=0.,layers='all'))]
    for strength,layers in ((1.,'all'),(2.,'all'),(2.,'last_half')):
        out.append((f'action_bias_{strength}_{layers}',dict(mode='action_bias',lam=strength,layers=layers)))
    for layers in ('all','last_half'):
        out.append((f'mass_0.5_{layers}',dict(mode='mass',lam=.5,layers=layers)))
    out.append(('mass_1.0_last_half',dict(mode='mass',lam=1.,layers='last_half')))
    return out

if __name__=='__main__':
    run.configs=configs
    run.main()
