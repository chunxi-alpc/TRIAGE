"""Necessary intervention and M1 checks using a real tiny Qwen on CPU."""
import tempfile
import unittest
import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM
from .engine import Engine
from .m1 import run_m1, select_prefix

class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        return [5 if text=="\u662f" else 6]
    def apply_chat_template(self,messages,**kwargs):
        return messages[-1]["content"]
    def __call__(self,text,**kwargs):
        ids=[1]+[7+ord(c)%43 for c in text]+[2,3]
        x=torch.tensor([ids])
        return BatchEncoding(dict(input_ids=x,attention_mask=torch.ones_like(x)))

class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
            head_dim=8,max_position_embeddings=512,attention_dropout=0.)
        config._attn_implementation="sdpa"
        cls.engine=Engine(device="cpu",model=Qwen3ForCausalLM(config),tokenizer=Tokenizer())

    def test_zero_and_prefix_cache(self):
        e=self.engine
        ctx=e.prepare("A realistic immutable prefix.")
        prefix=[(k.clone(),v.clone()) for k,v in ctx.prefix]
        baseline=e.run(ctx)
        self.assertEqual(baseline["heads"].shape,(3,4,8))
        self.assertAlmostEqual(baseline["ell"],e.full_score(ctx.prompt),places=6)
        identity={2:dict(N=torch.empty(8,0),B=torch.empty(0,8),mu=torch.randn(8),S=torch.empty(8,0))}
        self.assertEqual(baseline["ell"],e.run(ctx,operators=identity)["ell"])
        e.run(ctx,replacements={2:torch.ones(8)*3})
        self.assertEqual(baseline["ell"],e.run(ctx)["ell"])
        for (k,v),(a,b) in zip(ctx.prefix,prefix):
            self.assertTrue(torch.equal(k,a) and torch.equal(v,b))

    def test_only_target_query_head_and_position(self):
        e=self.engine
        ctx=e.prepare("Only the final query position is replayed.")
        captured={}
        handles=[]
        for i,layer in enumerate(e.layers):
            def hook(mod,args,out,layer=i):
                captured[layer]=args[0].detach().clone()
            handles.append(layer.self_attn.o_proj.register_forward_hook(hook))
        try:
            e.run(ctx)
            natural={i:x.clone() for i,x in captured.items()}
            target=5
            donor=torch.ones(8)*1.5
            e.run(ctx,replacements={target:donor})
            self.assertEqual(captured[1].shape,(1,1,32))
            for j in range(4):
                actual=captured[1].reshape(4,8)[j]
                expected=donor if j==1 else natural[1].reshape(4,8)[j]
                self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(captured[0],natural[0]))
        finally:
            for h in handles:h.remove()

    def test_natural_and_joint_clamp_gradients(self):
        e=self.engine;ctx=e.prepare("Frozen weights still permit intervention derivatives.")
        base=e.run(ctx,grad=True)
        z=base["heads"].reshape(-1,8)
        self.assertTrue(torch.isfinite(base["grads"]).all())
        self.assertTrue(all(p.grad is None for p in e.model.parameters()))
        epsilon=.001
        direction=torch.tensor([1.,0,0,0,0,0,0,0])
        lo=e.run(ctx,replacements={0:z[0]-epsilon*direction})["ell"]
        hi=e.run(ctx,replacements={0:z[0]+epsilon*direction})["ell"]
        self.assertAlmostEqual((hi-lo)/(2*epsilon),float(base["grads"][0,0,0]),delta=2e-4)
        donors={0:z[0],8:z[8]}
        clamp=e.run(ctx,replacements=donors,clamp_heads=[0,8],grad=True)
        self.assertAlmostEqual(clamp["ell"],base["ell"],places=6)
        lo=e.run(ctx,replacements={0:z[0]-epsilon*direction,8:z[8]})["ell"]
        hi=e.run(ctx,replacements={0:z[0]+epsilon*direction,8:z[8]})["ell"]
        self.assertAlmostEqual((hi-lo)/(2*epsilon),float(clamp["grads"][0,0,0]),delta=2e-4)

    def test_low_rank_operator_at_target(self):
        e=self.engine;ctx=e.prepare("Erase one direction before output projection.")
        base=e.run(ctx);z=base["heads"].reshape(-1,8)
        n=torch.zeros(8,1);n[2,0]=1
        mu=torch.arange(8)*.02
        op={10:dict(N=n,B=n.T,mu=mu,S=torch.empty(8,0))}
        expected=z[10]-n@(n.T@(z[10]-mu))
        actual=e.run(ctx,operators=op)["ell"]
        direct=e.run(ctx,replacements={10:expected})["ell"]
        self.assertEqual(actual,direct)

class LinearEngine:
    num_heads=3;n_heads=3
    def __init__(self):
        self.counts={"final_forwards":0}
        self.weights=torch.tensor([2.,-1.,.5])
    def prepare(self,prompt):
        return float(prompt)
    def run(self,context,replacements=None,capture=True,grad=False):
        z=torch.ones(3,1)*context
        for h,x in (replacements or {}).items():
            z[h]=x
        self.counts["final_forwards"]+=1
        return dict(ell=float(z[:,0]@self.weights),heads=z.reshape(1,3,1),
                    grads=self.weights.reshape(1,3,1))

class TestM1(unittest.TestCase):
    def test_reverse_denominator_and_unclipped_ranking(self):
        e=LinearEngine()
        views={};features={};pairs=[]
        for split in ("fit","dev"):
            for value in (0,1):
                key=split+str(value)
                views[key]={"prompt":str(value)}
                features[key]=e.run(value,grad=True)
            pairs.append(dict(pair_id=split,kind="A",split=split,left=split+"0",right=split+"1"))
        with tempfile.TemporaryDirectory() as d:
            result=run_m1("A",e,views,pairs,features,d)
            self.assertEqual(result["H"],[0])
            self.assertAlmostEqual(result["ranking"][0]["nie"],4/3)
            self.assertEqual(len(result["curve"]),4)
            self.assertTrue(result["target_met"])
            count=e.counts["final_forwards"]
            result2=run_m1("A",e,views,pairs,features,d)
            self.assertEqual(result2["H"],[0])
            self.assertEqual(e.counts["final_forwards"],count)

    def test_prefix_fallback_and_empty(self):
        self.assertEqual(select_prefix([dict(k=0,recovery=0.),dict(k=1,recovery=-.2)]),(0,False))
        self.assertEqual(select_prefix([dict(k=0,recovery=0.),dict(k=1,recovery=.5),
                                        dict(k=2,recovery=.5)]),(1,False))
        self.assertEqual(select_prefix([dict(k=0,recovery=0.),dict(k=1,recovery=.8),
                                        dict(k=2,recovery=1.)]),(1,True))


class TestMechanisms(unittest.TestCase):
    def test_h1_h2_and_joint_clamp_theory_execute_and_resume(self):
        import json
        from pathlib import Path
        from .mechanisms import run_mechanisms
        torch.set_num_threads(2)
        torch.manual_seed(37)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        config._attn_implementation="sdpa"
        model=Qwen3ForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.mul_(1000)
        e=Engine(device="cpu",model=model,tokenizer=Tokenizer())
        views={};features={};pairs=[]
        for i in range(2):
            for side,text in enumerate(("condition evidence "+str(i),"changed action longer "+str(i))):
                key=f"g{i}_{side}"
                views[key]=dict(prompt=text,source_id=f"s{i}",group_id=f"g{i}",
                                split="dev",cohort="main",original=(side==0))
                features[key]=e.run(e.prepare(text))
            self.assertGreaterEqual(abs(features[f"g{i}_1"]["ell"]-features[f"g{i}_0"]["ell"]),.1)
            pairs.append(dict(pair_id=f"pair{i}",kind="A",left=f"g{i}_0",right=f"g{i}_1",
                              split="dev",source_id=f"s{i}",group_id=f"g{i}",cohort="main"))
        n=torch.zeros(8,1);n[2,0]=1.
        ops={h:dict(N=n,B=n.T,mu=torch.zeros(8),S=torch.empty(8,0)) for h in (0,8)}
        selection=dict(H=[0,8],H_A=[0,8],H_P=[],T_h={"0":["A"],"8":["A"]})
        with tempfile.TemporaryDirectory() as d:
            out=run_mechanisms(e,views,pairs,features,ops,selection,d,split="dev")
            self.assertEqual(out["effective_pairs"],2)
            self.assertEqual(out["local_theory"]["nodes"],2)
            local=[json.loads(x) for x in (Path(d)/"local_theory_dev.jsonl").read_text().splitlines()]
            for row in local:
                self.assertAlmostEqual(row["nonlinear_error"]+row["cross_layer_change"],
                                       row["total_prediction_error"])
            count=e.counts["final_forwards"]
            again=run_mechanisms(e,views,pairs,features,ops,selection,d,split="dev")
            self.assertEqual(again["effective_pairs"],2)
            self.assertEqual(e.counts["final_forwards"],count)

if __name__=="__main__":
    unittest.main()
