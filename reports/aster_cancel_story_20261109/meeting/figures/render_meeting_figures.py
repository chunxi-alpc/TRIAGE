from pathlib import Path
import json,csv,collections,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

BASE=Path('/home/gcx/aster_cancel_story_20261109')
FIG=BASE/'meeting/figures'
TOY=BASE/'meeting/toy'
FIG.mkdir(parents=True,exist_ok=True);TOY.mkdir(parents=True,exist_ok=True)
font=Path('/home/gcx/.cache/aster_fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    font_family=font_manager.FontProperties(fname=str(font)).get_name()
else:
    font_family=['Noto Sans CJK SC','WenQuanYi Zen Hei','Microsoft YaHei','DejaVu Sans']
plt.rcParams.update({'font.family':font_family,
 'font.size':17,'axes.titlesize':21,'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':15,
 'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
 'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white'})
C={'blue':'#668DB8','ltblue':'#C1D8E9','green':'#7BA99B','peach':'#E6AE98','red':'#BD7D79','gray':'#A3AAB2','dark':'#334253','lav':'#DBDDEF'}
captions=[]
def save(fig,name,caption):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig);captions.append({'file':name,'caption':caption})
def tidy(ax):
    ax.grid(axis='y',color='#E6E8EB',linewidth=.7)
    ax.set_axisbelow(True)
def box(ax,x,y,w,h,text,color):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.02',
        linewidth=1.1,edgecolor='#728194',facecolor=color,transform=ax.transAxes)
    ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',transform=ax.transAxes,fontsize=18,color=C['dark'])
def arrow(ax,a,b,color='#64748A',style='-|>'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=17,
        linewidth=1.6,color=color,transform=ax.transAxes,connectionstyle='arc3'))

fig,ax=plt.subplots(figsize=(17.5,8.0));ax.axis('off')
ax.text(.03,.96,'ASTER-CR\uff1a\u6309\u75d5\u8ff9\u4f30\u8ba1\u5e72\u6270\uff0c\u5728\u8f93\u51fa\u7aef\u5bf9\u6d88',fontsize=27,weight='bold',transform=ax.transAxes,color=C['dark'])
box(ax,.035,.64,.135,.13,'\u5b8c\u6574\u8f93\u5165',C['ltblue'])
box(ax,.235,.64,.15,.13,'\u51bb\u7ed3\u6a21\u578b',C['ltblue'])
box(ax,.47,.72,.15,.12,'\u539f\u5206\u6570  \u2113',C['ltblue'])
box(ax,.775,.64,.19,.14,'\u6700\u7ec8\u5206\u6570\n\u539f\u5206\u6570 + \u4fee\u6b63 + b',C['lav'])
arrow(ax,(.17,.705),(.235,.705));arrow(ax,(.385,.74),(.47,.78));arrow(ax,(.62,.78),(.775,.74))
box(ax,.235,.405,.15,.13,'\u9009\u4e2d\u5934 h',C['ltblue'])
box(ax,.445,.405,.21,.13,'\u6295\u5f71\u8bfb\u53d6\n\u907f\u5f00\u4e3b\u8981\u8bc1\u636e\u65b9\u5411',C['ltblue'])
box(ax,.715,.405,.105,.13,'\u75d5\u8ff9 t',C['ltblue'])
box(ax,.865,.405,.105,.13,'\u7ebf\u6027\n\u4fee\u6b63',C['lav'])
arrow(ax,(.31,.64),(.31,.535));arrow(ax,(.385,.47),(.445,.47));arrow(ax,(.656,.47),(.715,.47));arrow(ax,(.82,.47),(.865,.47));arrow(ax,(.917,.535),(.917,.64),C['blue'])
box(ax,.06,.10,.23,.13,'\u540c\u8bc1\u636e A/P \u914d\u5bf9\n\u0394\u2113\u3001\u0394t', '#F0F2F4')
box(ax,.365,.10,.23,.13,'\u5dee\u5206\u5cad\u56de\u5f52\n\u62df\u5408\u7cfb\u6570 w', '#F0F2F4')
box(ax,.68,.10,.27,.13,'\u539f\u59cb\u8f93\u5165\u4e0e\u771f\u503c\n\u5355\u72ec\u62df\u5408\u622a\u8ddd b', '#F0F2F4')
arrow(ax,(.29,.165),(.365,.165))
arrow(ax,(.59,.23),(.86,.405),C['blue'])
arrow(ax,(.87,.23),(.99,.64),C['gray'])
ax.text(.035,.31,'\u79bb\u7ebf\u62df\u5408',fontsize=15,color='#667383',transform=ax.transAxes)
ax.text(.03,-.015,'\u63a8\u7406\u53ea\u9700\u4e00\u6b21\u524d\u5411\uff1b\u4fdd\u6301\u5168\u6587\u4e0e\u5185\u90e8\u8868\u793a\uff0c\u4fee\u6b63\u6700\u7ec8\u5224\u65ad\u5206\u6570\u3002',fontsize=18,transform=ax.transAxes,color=C['dark'])
save(fig,'01_method_workflow','\u65b9\u6cd5\u6d41\u7a0b\u793a\u610f\uff1aw\u53ea\u7528\u540c\u8bc1\u636e A/P \u914d\u5bf9\u7684\u5dee\u5206\u62df\u5408\uff0cb\u5355\u72ec\u4f7f\u7528\u539f\u59cb\u8f93\u5165\u6807\u7b7e\u6821\u51c6\uff1b\u63a8\u7406\u65f6\u4fdd\u6301\u8f93\u5165\u548c\u5185\u90e8\u8868\u793a\uff0c\u53ea\u4fee\u6b63\u5206\u6570\u3002')

metrics=json.loads((BASE/'trace/metrics.json').read_text())
a16=next(r for r in metrics['results'] if r['group']=='A' and r['rank']=='16')
fig,axs=plt.subplots(1,2,figsize=(15,6.4),gridspec_kw={'wspace':.55})
for ax in axs:tidy(ax);ax.set_xticks([])
axs[0].bar([0],[100*a16['transmission_rms_ratio']],width=.48,color=C['peach'])
axs[0].set(xlim=(-.7,.7),ylim=(0,20),ylabel='RMS \u4f20\u5bfc\u7387\uff08%\uff09',title='\u4f20\u5bfc\uff1a\u66ff\u6362\u75d5\u8ff9\u65b9\u5411')
axs[0].text(0,100*a16['transmission_rms_ratio']+.65,f"{100*a16['transmission_rms_ratio']:.2f}%",ha='center',fontsize=27,color=C['dark'])
axs[0].text(.5,-.13,'RMS(\u75d5\u8ff9\u4fee\u8865\u6548\u5e94) / RMS(\u603b\u6548\u5e94)',ha='center',transform=axs[0].transAxes,fontsize=15)
axs[1].bar([0],[a16['trace_r2']],width=.48,color=C['blue'])
axs[1].set(xlim=(-.7,.7),ylim=(0,1),ylabel='\u7559\u51fa\u914d\u5bf9\u56de\u5f52 R\u00b2',title='\u6307\u793a\uff1a\u7528\u75d5\u8ff9\u9884\u6d4b\u5206\u6570\u5dee')
axs[1].text(0,a16['trace_r2']+.045,f"{a16['trace_r2']:.3f}",ha='center',fontsize=27,color=C['dark'])
axs[1].text(.5,-.13,'1 \u2212 \u9884\u6d4b\u6b8b\u5dee\u5e73\u65b9\u548c / \u603b\u79bb\u5dee\u5e73\u65b9\u548c',ha='center',transform=axs[1].transAxes,fontsize=15)
fig.suptitle('\u5b9e\u6d4b\uff1a\u75d5\u8ff9\u4f20\u5bfc\u8f83\u5c11\uff0c\u4f46\u53ef\u9884\u6d4b\u90e8\u5206\u5e72\u6270',fontsize=25,y=1.04,color=C['dark'])
save(fig,'02_measured_transmission_predictability','\u5b9e\u6d4b\u7f13\u5b58\u7ed3\u679c\uff1a\u540c\u62791194\u4e2aA\u5f00\u53d1\u914d\u5bf9\u3001\u540c\u4e00rank16\u5b50\u7a7a\u95f4\uff0cRMS\u4f20\u5bfc\u738713.9057%\uff0c\u914d\u5bf9\u9884\u6d4bR\u00b2=0.406333\u3002\u4f20\u5bfc\u7387\u4e0eR\u00b2\u4f7f\u7528\u72ec\u7acb\u9762\u677f\uff0c\u4e8c\u8005\u4e0d\u80fd\u4f5c\u540c\u4e00\u79cd\u767e\u5206\u6bd4\u76f4\u63a5\u6bd4\u8f83\u3002')

pair_labels={r['pair_id']:r for r in map(json.loads,(Path('/home/gcx/aster_cr_repair_v2_run_20261108/data/pairs.jsonl')).open())}
old_rows=list(csv.DictReader(Path('/home/gcx/TRIAGE-worktrees/aster-cr-20261108/reports/aster_h2_alignment_20261108/pair_results.csv').open()))
rank_rows=[]
for rank in ['4','8','16','full']:
    r=[x for x in old_rows if x['method']=='ridge_r'+rank and x['kind'] in ['C','F']]
    before=np.array([float(x['original_difference']) for x in r]);after=np.array([float(x['residual_difference']) for x in r])
    direction=np.array([pair_labels[x['pair_id']]['right_label']-pair_labels[x['pair_id']]['left_label'] for x in r])
    a=next(x for x in metrics['results'] if x['rank']==rank and x['group']=='A')
    rank_rows.append({'rank':rank,'A_pairs':a['n'],'A_prediction_r2':a['trace_r2'],'CF_pairs':len(r),
      'CF_signed_response_before':float(np.mean(direction*before)),'CF_signed_response_after':float(np.mean(direction*after)),
      'CF_evidence_response_retention':float(np.sum(direction*after)/np.sum(direction*before)),
      'CF_correction_rms':float(np.sqrt(np.mean((before-after)**2)))})
with (FIG/'03_measured_rank_tradeoff.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rank_rows[0]));w.writeheader();w.writerows(rank_rows)
fig,axs=plt.subplots(1,2,figsize=(16,6.2),gridspec_kw={'wspace':.34})
x=np.arange(4);labs=['4','8','16','full']
vals=[r['A_prediction_r2'] for r in rank_rows]
axs[0].plot(x,vals,'o-',lw=2.7,ms=9,color=C['blue'])
axs[0].set(xticks=x,xticklabels=labs,ylim=(0,.66),xlabel='\u75d5\u8ff9\u5b50\u7a7a\u95f4\u7ef4\u5ea6',ylabel='A \u7c7b\u914d\u5bf9\u9884\u6d4b R\u00b2',title='\u7ef4\u5ea6\u589e\u52a0\uff0c\u5e72\u6270\u66f4\u53ef\u9884\u6d4b')
for xx,v in zip(x,vals):axs[0].annotate(f'{v:.3f}',(xx,v),xytext=(0,11),textcoords='offset points',ha='center',fontsize=16)
vals=[100*r['CF_evidence_response_retention'] for r in rank_rows]
axs[1].plot(x,vals,'o-',lw=2.7,ms=9,color=C['red']);axs[1].axhline(100,lw=1,ls='--',color=C['gray'])
axs[1].set(xticks=x,xticklabels=labs,ylim=(73,105),xlabel='\u75d5\u8ff9\u5b50\u7a7a\u95f4\u7ef4\u5ea6',ylabel='C/F \u8bc1\u636e\u54cd\u5e94\u4fdd\u7559\u7387\uff08%\uff09',title='\u540c\u65f6\u68c0\u67e5\u8bfb\u51fa\u540e\u7684\u8bc1\u636e\u54cd\u5e94')
for xx,v in zip(x,vals):axs[1].annotate(f'{v:.2f}%',(xx,v),xytext=(0,-23 if v>96 else 10),textcoords='offset points',ha='center',fontsize=16)
for ax in axs:tidy(ax)
fig.suptitle('\u5b9e\u6d4b\uff1a\u53ef\u9884\u6d4b\u6027\u589e\u52a0\uff0c\u4e5f\u53ef\u80fd\u524a\u5f31\u8bc1\u636e\u54cd\u5e94',fontsize=25,y=1.04,color=C['dark'])
save(fig,'03_measured_rank_tradeoff','\u5b9e\u6d4b\u65e7H2\u5f00\u53d1\u8bca\u65ad\uff1ar4/r8/r16/full\u7684A\u9884\u6d4bR\u00b2\u4f9d\u6b21\u4e3a0.0574/0.1911/0.4063/0.5481\uff0cC/F\u8bc1\u636e\u54cd\u5e94\u4fdd\u7559\u7387\u4f9d\u6b21\u4e3a99.30%/99.12%/94.81%/80.79%\u3002\u6295\u5f71\u6392\u9664\u4e3b\u8981\u8bc1\u636e\u65b9\u5411\uff0c\u5e76\u4e0d\u4fdd\u8bc1\u8bfb\u51fa\u4fee\u6b63\u4e0e\u8bc1\u636e\u65e0\u5173\u3002')

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
fig,axs=plt.subplots(1,3,figsize=(19,6.5),gridspec_kw={'wspace':.32})
names={'raw':'\u4e0d\u4fee\u6b63','erase_q':'\u64e6 q','orthogonal_erase_b':'\u6b63\u4ea4\u64e6 b','oblique_preserve_e':'\u4fdd\u7559\u8bc1\u636e\u7684\u659c\u6295\u5f71','paired_ridge_cancel':'\u914d\u5bf9\u56de\u5f52\u5bf9\u6d88'}
colors={'raw':C['gray'],'erase_q':C['dark'],'orthogonal_erase_b':C['peach'],'oblique_preserve_e':C['red'],'paired_ridge_cancel':C['blue']}
angles=[15,30,45,60]
for method in names:
    rr=[next(r for r in toy if r['method']==method and r['angle_degrees']==a) for a in angles]
    axs[0].plot(angles,[r['paired_difference_rms'] for r in rr],marker='o' if method!='erase_q' else 'x',lw=2.5 if method=='paired_ridge_cancel' else 1.8,ms=6,ls='--' if method=='erase_q' else '-',label=names[method],color=colors[method])
rr=[next(r for r in toy if r['method']=='orthogonal_erase_b' and r['angle_degrees']==a) for a in angles]
axs[1].plot(angles,[100*r['evidence_response_retention'] for r in rr],'o-',color=C['peach'],lw=2.6,label='\u6b63\u4ea4\u64e6 b')
axs[1].plot(angles,[100]*4,'o-',color=C['blue'],lw=2.6,label='\u5176\u4f59\u56db\u79cd\u65b9\u6cd5')
rr=[next(r for r in toy if r['method']=='paired_ridge_cancel' and r['angle_degrees']==a) for a in angles]
axs[2].plot(angles,[r['prediction_r2'] for r in rr],'o-',color=C['blue'],lw=2.7,ms=8)
for a,r in zip(angles,rr):axs[2].annotate(f"{r['prediction_r2']:.2f}",(a,r['prediction_r2']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=17)
axs[0].set(ylabel='\u540c\u8bc1\u636e A/P \u914d\u5bf9\u5206\u6570\u5dee RMS\uff08\u6a21\u62df\u5355\u4f4d\uff09',title='\u6a21\u62df\uff1a\u5e72\u6270\u4e0e\u8bfb\u6570\u566a\u58f0',ylim=(0,1.6))
axs[1].set(ylabel='\u8bc1\u636e\u54cd\u5e94\u4fdd\u7559\u7387\uff08\u6a21\u62df\uff0c%\uff09',title='\u6a21\u62df\uff1a\u5206\u6570\u7a33\u5b9a\u4ecd\u53ef\u80fd\u4f24\u8bc1\u636e',ylim=(0,110))
axs[2].set(ylabel='\u5bf9\u6d88\u7684\u7559\u51fa\u9884\u6d4b R\u00b2\uff08\u6a21\u62df\uff09',title='\u6a21\u62df\uff1a\u75d5\u8ff9\u8d8a\u53ef\u8bfb\uff0c\u5bf9\u6d88\u8d8a\u6709\u6548',ylim=(0,1.04))
axs[0].legend(fontsize=12.5,loc='upper right');axs[1].legend(fontsize=13.5,loc='lower right')
for ax in axs:tidy(ax);ax.set(xticks=angles,xlabel='\u8bc1\u636e\u4e0e\u5e72\u6270\u5939\u89d2 \u03b8\uff08\u5ea6\uff09')
fig.suptitle('\u7b80\u5316\u6a21\u578b\u6a21\u62df\uff0c\u975e\u4e34\u5e8a\u7ed3\u679c',fontsize=29,y=1.04,color=C['red'],weight='bold')
save(fig,'04_toy_simulation','\u7b80\u5316\u6a21\u578b\u6a21\u62df\uff0c\u975e\u4e34\u5e8a\u7ed3\u679c\u3002\u4e8c\u7ef4\u6a21\u578b\u4e2d\uff0c\u64e6q\u4e0d\u6539\u53d8\u539f\u5206\u6570\uff1b\u6b63\u4ea4\u64e6b\u524a\u5f31\u8bc1\u636e\u54cd\u5e94\uff1b\u4fdd\u7559\u8bc1\u636e\u7684\u7cbe\u786e\u659c\u6295\u5f71\u4f1a\u653e\u5927\u75d5\u8ff9\u566a\u58f0\u3002\u914d\u5bf9\u56de\u5f52\u5728\u5e26\u566a\u75d5\u8ff9\u4e0a\u6536\u7f29\u7cfb\u6570\uff1b\u6548\u679c\u968f\u53ef\u8bfb\u6027\u53d8\u5316\u3002\u672c\u6a21\u62df\u4e0d\u6bd4\u8f83\u771f\u5b9e\u6a21\u578b\u4e34\u5e8a\u51c6\u786e\u7387\u3002')

new=json.loads((BASE/'meeting/paired_fit/metrics.json').read_text())
method_order=['raw','bias_only','old_selected_CR','paired_APF0']
labels=['\u539f\u6a21\u578b','\u4ec5\u6821\u51c6 b','\u65e7\u8054\u5408\u76ee\u6807','\u914d\u5bf9\u56de\u5f52\u5bf9\u6d88']
summary_rows=[]
for method,label in zip(method_order,labels):
    r=new['methods'][method];ap=r['pair_groups']['AP'];cf=r['pair_groups']['CF'];natural=r['nodes']['original']
    summary_rows.append({'method':method,'label':label,'AP_RMS':ap['difference_rms'],'AP_prediction_R2':ap['nuisance_prediction_r2'],
      'AP_variance_reduction':ap['difference_variance_reduction'],'CF_response_retention':cf['signed_response_retention'],
      'CF_correction_in_nuisance_units':cf['correction_difference_rms_over_ap_interference_rms'],
      'natural_correct':natural['correct'],'natural_total':natural['n'],'correction_label_Pearson':natural['correction_label_pearson']})
with (FIG/'05_measured_refit_development.csv').open('w',encoding='utf-8-sig',newline='') as f:
    ww=csv.DictWriter(f,fieldnames=list(summary_rows[0]));ww.writeheader();ww.writerows(summary_rows)
fig,axs=plt.subplots(1,2,figsize=(17,6.4),gridspec_kw={'wspace':.32})
x=np.arange(4);cols=[C['gray'],C['ltblue'],C['peach'],C['blue']]
vals=[r['AP_RMS'] for r in summary_rows]
axs[0].bar(x,vals,color=cols,width=.6);axs[0].set(xticks=x,xticklabels=labels,ylabel='A/P \u914d\u5bf9\u5206\u6570\u5dee RMS\uff08logit\uff09',ylim=(0,5.6),title='\u5b9e\u6d4b\uff1a\u65b0\u62df\u5408\u964d\u4f4e\u5e72\u6270\u654f\u611f\u5ea6')
for xx,v in zip(x,vals):axs[0].text(xx,v+.12,f'{v:.2f}',ha='center',fontsize=20)
vals=[100*r['CF_response_retention'] for r in summary_rows]
axs[1].bar(x,vals,color=cols,width=.6);axs[1].axhline(99,color=C['red'],ls='--',lw=1.5,label='\u539f\u6545\u4e8b\u793a\u4f8b\uff1a99%')
axs[1].set(xticks=x,xticklabels=labels,ylabel='C/F \u8bc1\u636e\u54cd\u5e94\u4fdd\u7559\u7387\uff08%\uff09',ylim=(0,112),title='\u5b9e\u6d4b\uff1a\u8bc1\u636e\u4fdd\u7559\u6761\u4ef6\u4ecd\u672a\u6ee1\u8db3')
for xx,v in zip(x,vals):axs[1].text(xx,v+1.7,f'{v:.2f}%',ha='center',fontsize=19)
axs[1].legend(loc='lower right',fontsize=14)
for ax in axs:tidy(ax);ax.tick_params(axis='x',labelsize=15)
fig.suptitle('\u6309 5.5 \u91cd\u4f30\uff1a\u673a\u5236\u5f97\u5230\u90e8\u5206\u652f\u6301\uff0c\u65b9\u6cd5\u5c1a\u5f85\u5b8c\u5584',fontsize=26,y=1.04,color=C['dark'])
save(fig,'05_measured_refit_development','\u5b9e\u6d4b\u5f00\u53d1\u96c6\u3001\u975e\u6d4b\u8bd5\u7ed3\u679c\u3002\u63095.5\u91cd\u4f30\u540eA/P\u914d\u5bf9\u5deeRMS\u4ece3.879\u964d\u81f32.980\uff0c\u65b9\u5dee\u4e0b\u964d41.01%\uff1bC/F\u8bc1\u636e\u54cd\u5e94\u4fdd\u755994.64%\uff0c\u672a\u8fbe\u5230\u539f\u6545\u4e8b\u793a\u4f8b\u768499%\u3002\u56e0\u6b64\u652f\u6301\u75d5\u8ff9\u53ef\u9884\u6d4b\u5e72\u6270\uff0c\u5c1a\u4e0d\u80fd\u58f0\u79f0\u4fee\u6b63\u4e0d\u4f24\u8bc1\u636e\u3002')

(FIG/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(FIG/'README.zh.md').write_text('# \u6c47\u62a5\u56fe\u8868\n\n'+ '\n\n'.join(f"## {r['file']}\n\n{r['caption']}" for r in captions)+'\n\n01\u4e3a\u65b9\u6cd5\u793a\u610f\uff0c02/03/05\u4e3a\u7f13\u5b58\u5b9e\u6d4b\u5f00\u53d1\u7ed3\u679c\uff0c04\u660e\u786e\u4e3a\u7b80\u5316\u6a21\u578b\u6a21\u62df\u3002\u6bcf\u56fe\u63d0\u4f9bPNG\u548cSVG\u3002\u6570\u503c\u53ef\u7531render_meeting_figures.py\u590d\u7b97\uff0c\u6a21\u62df\u539f\u59cb\u7ed3\u679c\u4e0e\u8bbe\u5b9a\u4fdd\u5b58\u5728../toy/\u3002\n',encoding='utf-8')
print(json.dumps({'figures':[r['file'] for r in captions],'toy_cancel_results':[r for r in toy if r['method']=='paired_ridge_cancel'],'rank_results':rank_rows},ensure_ascii=False))

