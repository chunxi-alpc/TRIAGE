
"""Score new cached coordinates with a saved trace-only editor; no backbone."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from search import Editor
def main(model_path,coordinates,output):
 torch.set_num_threads(2)
 state=torch.load(model_path,map_location='cpu',weights_only=False)
 cfg=state['config'];mu=state['mu'];scale=state['scale']
 model=Editor(len(mu),cfg['width'],cfg['depth'])
 model.load_state_dict(state['state_dict']);model.eval()
 arrays=np.load(coordinates);x=arrays[cfg['feature']]
 z=((x-mu)/scale).astype('float32')
 with torch.no_grad():delta=np.concatenate([model(torch.from_numpy(z[i:i+1024])).numpy() for i in range(0,len(z),1024)])
 np.savez_compressed(output,ids=arrays['ids'],delta=delta,b=state['b'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--coordinates',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();main(a.model,a.coordinates,a.output)

