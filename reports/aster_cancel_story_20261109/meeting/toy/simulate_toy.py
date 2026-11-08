from pathlib import Path
import numpy as np,json,csv
TOY=Path(__file__).resolve().parent
SEED=20261109;N_FIT=512;N_TEST=4096;SIGMA_X=.05;SIGMA_Y=.25;RIDGE_PER_PAIR=.02
toy=[]
for angle in [15,30,45,60]:
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    rng=np.random.default_rng(SEED+angle)
    def sample(n):
        e=rng.normal(size=n);nu=rng.normal(size=(n,2))
        epsx=rng.normal(0,SIGMA_X,size=(n,2));epsy=rng.normal(0,SIGMA_Y,size=(n,2))
        hx=e[:,None]+c*nu+epsx;hy=s*nu+epsy
        return e,hx,hy
    ef,xf,yf=sample(N_FIT);ex,xx,yx=sample(N_TEST)
    dx=xf[:,1]-xf[:,0];dy=yf[:,1]-yf[:,0]
    w=-np.dot(dx,dy)/(np.dot(dy,dy)+N_FIT*RIDGE_PER_PAIR)
    coeffs={'raw':(1.,0.),'erase_q':(1.,0.),'orthogonal_erase_b':(s*s,-c*s),'oblique_preserve_e':(1.,-c/s),'paired_ridge_cancel':(1.,float(w))}
    d0=xx[:,1]-xx[:,0]
    for method,(a,b) in coeffs.items():
        score=a*xx+b*yx;delta=score[:,1]-score[:,0]
        predicted=d0-delta
        r2=1-np.sum((d0-predicted)**2)/np.sum((d0-d0.mean())**2)
        toy.append({'data_origin':'toy_simulation_not_clinical','angle_degrees':angle,'method':method,'seed':SEED+angle,
          'fit_pairs':N_FIT,'test_pairs':N_TEST,'sigma_x':SIGMA_X,'sigma_y':SIGMA_Y,'ridge_lambda':N_FIT*RIDGE_PER_PAIR,
          'readout_x_coefficient':a,'readout_y_coefficient':b,'paired_difference_rms':float(np.sqrt(np.mean(delta**2))),
          'prediction_r2':float(r2),'variance_reduction':float(1-np.var(delta)/np.var(d0)),
          'evidence_response_retention':a,'output_mse_to_clean_evidence':float(np.mean((score-ex[:,None])**2))})
with (TOY/'simulation_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    wri=csv.DictWriter(f,fieldnames=list(toy[0]));wri.writeheader();wri.writerows(toy)
(TOY/'simulation_settings.json').write_text(json.dumps({'data_origin':'simplified simulation, not clinical results','seed':SEED,'angles':[15,30,45,60],'fit_pairs':N_FIT,'test_pairs':N_TEST,'sigma_x':SIGMA_X,'sigma_y':SIGMA_Y,'ridge_penalty_per_fit_pair':RIDGE_PER_PAIR,'model':'h=(e+cos(theta)*nu+eps_x, sin(theta)*nu+eps_y); score=h[0]; trace=h[1]','same_truth_pairs':'same e, independent nu and observation noise; no labels used to fit coefficient','evidence_test':'change e from -1 to +1 with nu and noise fixed; response retention is output coefficient on e'},indent=2)+'\n')
print('Simulation only; not clinical results. Wrote simulation_results.csv and simulation_settings.json.')
