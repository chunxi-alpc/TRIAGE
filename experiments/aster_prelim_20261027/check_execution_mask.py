import json
from pathlib import Path
import torch
from attention_engine import Engine
eng=Engine.__new__(Engine)
eng.state={'roles':torch.tensor([False,True,False,True,False,False]),'branches':torch.tensor([-1,0,-1,1,-1,-1]),'segments':torch.tensor([-1,0,-1,1,-1,-1]),'phase':'execute','selected':0,'lam':0}
allow=torch.ones(6,6,dtype=torch.bool).tril()[None,None]
def mask():return eng.mask(None,(),{'hidden_states':torch.zeros(1,6,4),'attention_mask':allow})[1]['attention_mask']
zero=mask();assert torch.equal(zero==0,allow)
eng.state['hard']=True;hard=mask();assert (hard[0,0,:,3]<-1e4).all();assert hard[0,0,4,1]==0
eng.state.update(hard=False,lam=1);soft=mask();assert soft[0,0,4,1]==0 and soft[0,0,4,3]<-9
eng.state.update(phase='blind');blind=mask();assert blind[0,0,4,1]<-1e4 and blind[0,0,4,3]<-1e4
result={'status':'PASS','checks':['zero-strength execution equals native causal mask','hard gate blocks competitor at every query','execution retains applicable-action access','finite bias applies to competitor','blind isolation retained'],'scope':'CPU mask regression, not model performance'}
(Path(__file__).parent/'results/execution_mask_regression.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
