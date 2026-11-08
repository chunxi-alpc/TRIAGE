import numpy as np
from experiments.ordinary_readout import fit_correction,predict,make_transform,project,reachable_scores,execute,summarize

def test_zero_correction_matches_original_including_tie():
 e=np.array([-1.,0.,1.]);z=np.ones((3,2));s=predict(dict(mode='linear',coef=np.zeros(3)),e,z)
 assert np.array_equal(s,e) and np.array_equal(s>0,[False,False,True])

def test_readout_can_use_information_absent_from_score():
 z=np.array([[-2.],[-1.],[1.],[2.]]);e=np.zeros(4);y=np.array([0,0,1,1])
 model=fit_correction(e,z,y,.01)
 assert np.array_equal(predict(model,e,z)>0,y)
 assert model['parameters']==2

def test_logistic_does_not_reverse_score_order():
 e=np.arange(-4.,5.);model=fit_correction(e,np.zeros((9,1)),e>1,mode='logistic')
 assert model['coef'][1]>=0 and np.all(np.diff(predict(model,e))>=0)

def test_transform_uses_fit_only_and_is_reusable():
 rng=np.random.default_rng(1);fit=rng.normal(size=(20,5));t=make_transform(fit,3);mean=t['mean'].copy();basis=t['pca'].components_.copy()
 projected=project(t,fit);project(t,np.full((8,5),1e6))
 assert projected.shape==(20,3) and np.allclose(projected.mean(0),0,atol=1e-12)
 assert np.array_equal(mean,t['mean']) and np.array_equal(basis,t['pca'].components_)

def test_autonomous_paths_do_not_follow_gold_branches():
 guide={'groups':[{'group_id':'G0','rules':[{'rule_id':'R0Y','condition':'first','action':'G1'},{'rule_id':'R0N','condition':'first','action':'L0:no'}]},{'group_id':'G1','rules':[{'rule_id':'R1Y','condition':'second','action':'L1:yes'},{'rule_id':'R1N','condition':'second','action':'L2:no'}]}]}
 case=dict(guideline=guide,gold_path=[dict(gid='G0',label=0)],gold_leaf='L0');seen=[]
 def score(gid):seen.append(gid);return {'a':1.,'b':-1.}
 scores=reachable_scores(case,score,['a','b'])
 assert seen==['G0','G1']
 assert not execute(case,scores['a'])['correct'] and execute(case,scores['b'])['correct']

def test_summary_keeps_matched_denominator_and_new_errors():
 a=[dict(case_id='a',source_id=1,correct=True),dict(case_id='b',source_id=2,correct=False)]
 b=[dict(case_id='a',source_id=1,correct=False),dict(case_id='b',source_id=2,correct=True)]
 s=summarize(a,b)
 assert s['n']==2 and s['corrected']==1 and s['new_errors']==1 and s['difference_by_source_bootstrap']['estimate']==0
