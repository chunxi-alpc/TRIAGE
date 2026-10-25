import copy,unittest
from analyze import predict,fit,metrics,GRID

class AnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'p{t}',tree=t,variant=v,gold_option=j%2,covered=True,option_S=[.2,.8],
            scores={'original':[1.,0.],'zero_midlate4':[1.1,.1],'dynamic_midlate4':[1.3,.4]},answers={'original':0})
            for t in ('t1','t2') for j,v in enumerate(('NC','AC','IC','SW'))]
    def test_label_does_not_enter_prediction(self):
        r=self.rows()[0];p=predict(r,GRID[0],'dynamic_midlate4');r['gold_option']=1-r['gold_option']
        self.assertEqual(p,predict(r,GRID[0],'dynamic_midlate4'))
    def test_heldout_labels_do_not_select_parameters(self):
        a=self.rows();b=copy.deepcopy(a)
        for r in b:
            if r['tree']=='t1':r['gold_option']=1-r['gold_option']
        x=fit(a,'static_fusion');y=fit(b,'static_fusion')
        self.assertEqual(x['folds']['t1'],y['folds']['t1'])
    def test_zero_and_no_residual_are_identical(self):
        for c in GRID:
            for r in self.rows():
                self.assertEqual(predict(r,c,'zero_strength'),predict(r,c,'no_residual'))
    def test_zero_support_falls_back_to_model(self):
        r=self.rows()[0];r['option_S']=[0.,0.]
        self.assertEqual(predict(r,{'beta':1.,'t0':1.,'ts':1.},'static_fusion'),0)
    def test_metrics_keep_entire_population(self):
        rows=self.rows();m=metrics(rows,[1]*len(rows));self.assertEqual(m['n'],8)
        self.assertEqual(sum(v['n'] for v in m['by_variant'].values()),8)
        self.assertEqual(m['corrected'],4);self.assertEqual(m['regressed'],4)
    def test_every_candidate_retained(self):
        x=fit(self.rows(),'static_fusion');self.assertEqual(len(x['all_candidates']),27)
        self.assertTrue(all(len(c['predictions'])==8 for c in x['all_candidates']))
if __name__=='__main__':unittest.main()
