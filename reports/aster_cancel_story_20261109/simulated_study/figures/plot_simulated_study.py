from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap

ROOT=Path('/home/gcx/aster_cancel_story_20261109/simulated_study')
OUT=ROOT/'figures';OUT.mkdir(parents=True,exist_ok=True)
font=Path('/home/gcx/.cache/aster_fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    font_family=font_manager.FontProperties(fname=str(font)).get_name()
else:font_family=['Noto Sans CJK SC','Microsoft YaHei','DejaVu Sans']
plt.rcParams.update({'font.family':font_family,'font.size':16,'axes.titlesize':19,'axes.labelsize':16,
    'xtick.labelsize':13,'ytick.labelsize':14,'axes.unicode_minus':False,
    'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'path',
    'axes.spines.top':False,'axes.spines.right':False})
BLUE='#668DB8';PEACH='#E6AE98';RED='#BD7D79';GRAY='#A3AAB2';DARK='#334253'
MAIN_MODELS=['Qwen3-8B','Llama-3.1-8B','Gemma-2-9B-IT']
SUPPLEMENTARY_MODELS=['MedGemma-27B','Gemma-3-27B-IT']
ALL_MODELS=MAIN_MODELS+SUPPLEMENTARY_MODELS
SELECTED_METHODS=['aster_cr','paired_n','lora']
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def number(row,key):return float(row[key])
def task_label(task):
    s=task.lower()
    if 'mega' in s:return 'MEGA'
    if 'nli' in s:return 'NLI4CT'
    return '\u4e09\u6bb5\u8bba'
def model_label(model):
    s=model.lower()
    if 'qwen' in s:return 'Qwen3-8B'
    if 'llama' in s:return 'Llama-3.1-8B'
    if 'medgemma' in s:return 'MedGemma-27B'
    if 'gemma-2' in s or 'gemma2' in s:return 'Gemma-2-9B-IT'
    if 'gemma-3' in s or 'gemma3' in s:return 'Gemma-3-27B-IT'
    return model
def sort_key(key):
    model,task,scenario=key
    mm=ALL_MODELS.index(model_label(model))
    tt={'MEGA':0,'NLI4CT':1,'\u4e09\u6bb5\u8bba':2}[task_label(task)]
    return (mm,tt,scenario)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
def dump_csv(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

main=read('main_summary.csv')
lookup={(r['model'],r['task'],r['scenario'],r['method']):r for r in main}
all_keys=sorted({(r['model'],r['task'],r['scenario']) for r in main},key=sort_key)
all_vr=np.array([100*number(lookup[(*k,m)],'variance_reduction_mean') for k in all_keys for m in SELECTED_METHODS])
all_er=np.array([100*number(lookup[(*k,m)],'evidence_retention_mean') for k in all_keys for m in SELECTED_METHODS])
if all_vr.min()<0:
    norm_vr=TwoSlopeNorm(vcenter=0,vmin=min(-1,float(all_vr.min())),vmax=max(1,float(all_vr.max())))
    cmap_vr=LinearSegmentedColormap.from_list('vr',[PEACH,'#FFFFFF',BLUE])
else:
    norm_vr=Normalize(vmin=0,vmax=max(1,float(all_vr.max())))
    cmap_vr=LinearSegmentedColormap.from_list('vr',['#F4F6F9',BLUE])
norm_er=TwoSlopeNorm(vcenter=100,vmin=min(95,float(all_er.min())-1),vmax=max(105,float(all_er.max())+1))
cmap_er=LinearSegmentedColormap.from_list('er',[PEACH,'#FFFFFF','#B0BBCD'])

def heatmap_group(models,name,title):
    keys=[k for k in all_keys if model_label(k[0]) in models]
    selected=[lookup[(*key,m)] for key in keys for m in SELECTED_METHODS]
    dump_csv(name+'_data.csv',selected)
    vr=np.array([[100*number(lookup[(*key,m)],'variance_reduction_mean') for m in SELECTED_METHODS] for key in keys])
    er=np.array([[100*number(lookup[(*key,m)],'evidence_retention_mean') for m in SELECTED_METHODS] for key in keys])
    vr_sd=np.array([[100*number(lookup[(*key,m)],'variance_reduction_std') for m in SELECTED_METHODS] for key in keys])
    er_sd=np.array([[100*number(lookup[(*key,m)],'evidence_retention_std') for m in SELECTED_METHODS] for key in keys])
    height=8.8 if len(keys)>6 else 7.2
    fig,axs=plt.subplots(1,2,figsize=(18.1,height))
    fig.subplots_adjust(left=.20,right=.945,top=.79,bottom=.13,wspace=.40)
    fig.suptitle(title,fontsize=28,color=RED,y=.98,weight='bold')
    fig.text(.5,.91,'\u672a\u8c03\u7528\u6a21\u578b\u3001\u672a\u4f7f\u7528\u771f\u5b9e\u4efb\u52a1\u6837\u672c\u3001\u672a\u5b9e\u73b0\u8fd9\u4e9b\u57fa\u7ebf\uff1b\u5355\u5143\u683c\u4e3a 5 \u4e2a\u6a21\u62df seed \u7684\u5747\u503c \u00b1 SD',ha='center',fontsize=15,color=DARK)
    for j,(ax,mat,sd,panel,cmap,norm) in enumerate(zip(axs,[vr,er],[vr_sd,er_sd],
       ['\u5e72\u6270\u65b9\u5dee\u4e0b\u964d VR\uff08%\uff0c\u8d8a\u9ad8\u8d8a\u597d\uff09','\u8bc1\u636e\u54cd\u5e94 ER\uff08%\uff0c\u76ee\u6807\u4e3a 100\uff09'],[cmap_vr,cmap_er],[norm_vr,norm_er])):
        im=ax.imshow(mat,aspect='auto',cmap=cmap,norm=norm)
        ax.set_xticks(range(3),['\u6700\u7ec8\u65b9\u6cd5\u60c5\u666f','\u914d\u5bf9\u5199\u56de\u60c5\u666f','LoRA \u60c5\u666f'])
        ax.set_yticks(range(len(keys)),[model_label(k[0])+' \u00b7 '+task_label(k[1]) for k in keys] if j==0 else ['']*len(keys))
        ax.set_title(panel,pad=18)
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                ax.text(x,y,f'{mat[y,x]:.1f}\n\u00b1 {sd[y,x]:.1f}',ha='center',va='center',fontsize=12,color='#243346')
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(keys),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.ax.tick_params(labelsize=12)
    save(fig,name)
    return len(keys)

main_count=heatmap_group(MAIN_MODELS,'01_simulated_main_vr_er','\u6a21\u62df\u7ed3\u679c\uff1a\u4e3b\u5b9e\u9a8c\u7684\u4e5d\u79cd\u53c2\u6570\u914d\u7f6e')
supp_count=heatmap_group(SUPPLEMENTARY_MODELS,'03_simulated_supplementary_27b_vr_er','\u6a21\u62df\u7ed3\u679c\uff1a\u4e24\u79cd 27B \u8865\u5145\u914d\u7f6e\u5bf9\u7167')

abl=[r for r in read('ablation_seed_metrics.csv') if model_label(r['model']) in MAIN_MODELS]
abl_methods=['full','without_w','without_b','no_orth_U','random_subspace','label_joint']
abl_labels=['\u5b8c\u6574\u65b9\u6cd5','\u53bb\u6389 w','\u53bb\u6389 b','\u4fdd\u7559 U \u4e2d\u8bc1\u636e','\u968f\u673a\u5b50\u7a7a\u95f4','\u6807\u7b7e\u8054\u5408\u8bad\u7ec3']
cols=[BLUE,GRAY,'#A7BFD7',PEACH,RED,'#AAA2C4']
agg=[]
for method,label in zip(abl_methods,abl_labels):
    rr=[r for r in abl if r['method']==method]
    seeds=sorted({r['seed'] for r in rr})
    for metric in ['variance_reduction','evidence_retention','native_accuracy']:
        seed_values=[np.mean([number(r,metric) for r in rr if r['seed']==seed]) for seed in seeds]
        agg.append({'status':'simulated','model_group':'main_three_models','method':method,'method_label':label,'metric':metric,
          'scenario_count':len({r['scenario'] for r in rr}),'seed_count':len(seeds),
          'mean':float(np.mean(seed_values)),'std':float(np.std(seed_values,ddof=1)),
          'aggregation':'equal main-scenario weights within seed; sample SD over 5 seed means'})
dump_csv('02_simulated_ablation_data.csv',agg)
fig,axs=plt.subplots(1,3,figsize=(19,7.3))
fig.subplots_adjust(left=.065,right=.985,top=.77,bottom=.30,wspace=.29)
fig.suptitle('\u6a21\u62df\u7ed3\u679c\uff1a\u4e09\u79cd\u4e3b\u5b9e\u9a8c\u914d\u7f6e\u7684\u5173\u952e\u6d88\u878d',fontsize=28,color=RED,y=.98,weight='bold')
fig.text(.5,.90,'\u4ec5\u6c47\u603b Qwen3-8B\u3001Llama-3.1-8B\u3001Gemma-2-9B-IT\uff1b\u4e5d\u79cd\u914d\u7f6e\u7b49\u6743\uff0c\u8bef\u5dee\u6761\u4e3a 5 \u4e2a seed \u5747\u503c\u7684 SD\u3002',ha='center',fontsize=15,color=DARK)
metric_titles=[('variance_reduction','\u5e72\u6270\u65b9\u5dee\u4e0b\u964d VR\uff08%\uff09'),('evidence_retention','\u8bc1\u636e\u54cd\u5e94 ER\uff08%\uff0c\u76ee\u6807 100\uff09'),('native_accuracy','\u5408\u6210\u6837\u672c\u5224\u65ad\u51c6\u786e\u7387\uff08%\uff09')]
for ax,(metric,title) in zip(axs,metric_titles):
    rr=[next(r for r in agg if r['method']==m and r['metric']==metric) for m in abl_methods]
    means=[100*r['mean'] for r in rr];stds=[100*r['std'] for r in rr]
    for i,(mean,sd) in enumerate(zip(means,stds)):
        ax.errorbar(i,mean,yerr=sd,fmt='o',ms=9,capsize=4,color=cols[i],elinewidth=1.8)
        ax.annotate(f'{mean:.1f}',(i,mean),xytext=(0,12),textcoords='offset points',ha='center',fontsize=13)
    ax.set_xticks(range(len(abl_labels)),abl_labels,rotation=32,ha='right')
    ax.set_ylabel(title);ax.grid(axis='y',color='#E6E8EB');ax.set_axisbelow(True)
    if metric=='evidence_retention':ax.axhline(100,color=GRAY,ls='--',lw=1)
    low=min(m-s for m,s in zip(means,stds));high=max(m+s for m,s in zip(means,stds))
    span=max(4,high-low);ax.set_ylim(low-.20*span,high+.36*span)
save(fig,'02_simulated_ablation')

common='\u6a21\u62df\u7ed3\u679c\uff0c\u975e\u771f\u5b9ebenchmark\u3002\u6a21\u578b\u4e0e\u4efb\u52a1\u540d\u79f0\u4ec5\u5bf9\u5e94\u9884\u8bbe\u53c2\u6570\u914d\u7f6e\uff1b\u672a\u8c03\u7528\u6a21\u578b\u3001\u4f7f\u7528\u771f\u5b9e\u4efb\u52a1\u6837\u672c\u6216\u5b9e\u73b0\u76f8\u5e94\u57fa\u7ebf\u3002VR\u8d8a\u9ad8\u8868\u793a\u914d\u5bf9\u5dee\u65b9\u5dee\u8d8a\u5c0f\uff1bER\u4ee5100%\u4e3a\u76ee\u6807\uff0c\u8d85\u8fc7100%\u8bf4\u660e\u54cd\u5e94\u653e\u5927\u3002\u8f83\u7eaf\u75d5\u8ff9\u4e0e\u8f83\u9ad8ER\u5747\u6765\u81ea\u989d\u5916\u6a21\u62df\u5047\u8bbe\uff0c\u4e0d\u80fd\u66ff\u4ee3\u771f\u5b9e\u5f00\u53d1\u96c694.64%\u7684ER\u3002'
captions=[
 {'figure':'01_simulated_main_vr_er','status':'simulated','model_group':'main','caption':common+'\u4e3b\u5b9e\u9a8c\u4e3aQwen3-8B\u3001Llama-3.1-8B-Instruct\u3001Gemma-2-9B-IT\u4e09\u7ec4\u914d\u7f6e\uff0c\u5404\u542bMEGA\u3001NLI4CT\u3001\u4e09\u6bb5\u8bba\u4e09\u79cd\u4efb\u52a1\u914d\u7f6e\u3002\u6bcf\u683c\u4e3a5\u4e2a\u6a21\u62dfseed\u5747\u503c\u00b1\u6837\u672cSD\u3002'},
 {'figure':'02_simulated_ablation','status':'simulated','model_group':'main','caption':'\u6a21\u62df\u7ed3\u679c\uff0c\u4ec5\u6c47\u603b\u4e09\u79cd\u4e3b\u6a21\u578b\u7684\u4e5d\u4e2a\u573a\u666f\u3002\u6bcf\u4e2aseed\u5185\u7b49\u6743\u6c47\u603b\u573a\u666f\uff0c\u518d\u8ba1\u7b975\u4e2aseed\u5747\u503c\u4e0e\u6837\u672cSD\u3002\u53bb\u6389b\u4e0d\u6539\u53d8\u5206\u6570\u5dee\u5206\uff0c\u4f46\u53ef\u6539\u53d8\u9608\u503c\u5224\u65ad\u51c6\u786e\u7387\u3002\u6d88\u878d\u53cd\u6620\u9884\u8bbe\u7279\u5f81\u4e0e\u566a\u58f0\u53d8\u6362\uff0c\u4e0d\u8bc1\u660e\u771f\u5b9e\u6a21\u578b\u6027\u80fd\u3002'},
 {'figure':'03_simulated_supplementary_27b_vr_er','status':'simulated','model_group':'supplementary','caption':common+'\u8865\u5145\u5bf9\u7167\u4e3aMedGemma-27B\u4e0eGemma-3-27B-IT\uff0c\u5404\u542b\u4e09\u79cd\u4efb\u52a1\u914d\u7f6e\u3002\u8272\u6807\u4e0e\u4e3b\u5b9e\u9a8c\u56fe\u5b8c\u5168\u76f8\u540c\uff0c\u4fbf\u4e8e\u5bf9\u7167\uff1b\u6a21\u578b\u540d\u5bf9\u5e94\u72ec\u7acb\u9884\u8bbe\u6761\u4ef6\uff0c\u4e0d\u80fd\u636e\u6b64\u63a8\u65ad\u5b9e\u9645\u533b\u5b66\u5fae\u8c03\u6216\u53c2\u6570\u89c4\u6a21\u7684\u56e0\u679c\u6548\u5e94\u3002'}]
(OUT/'captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'figure_groups.json').write_text(json.dumps({'status':'simulated','main_models':MAIN_MODELS,'supplementary_models':SUPPLEMENTARY_MODELS,'main_scenarios':main_count,'supplementary_scenarios':supp_count,'selected_methods':SELECTED_METHODS,'heatmap_color_scales_shared':True,'ablation_group':'main_models_only'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'README.zh.md').write_text('# \u4e94\u79cd\u6a21\u578b\u914d\u7f6e\u7684\u6a21\u62df\u56fe\u8868\n\n\u6240\u6709\u56fe\u5747\u9192\u76ee\u6807\u6ce8\u201c\u6a21\u62df\u7ed3\u679c\u201d\u3002\u6570\u503c\u6765\u81ea\u5171\u540c\u6f5c\u5728\u53d8\u91cf\u4e0e\u9884\u8bbe\u53d8\u6362\uff0c\u4e0d\u662f\u5b9e\u9645\u6a21\u578b\u6216\u57fa\u7ebf\u5b9e\u9a8c\u3002\n\n'+ '\n\n'.join(r['figure']+'\uff1a'+r['caption'] for r in captions)+'\n\n\u7ed8\u56fe\u6765\u6e90\uff1a../main_summary.csv\u3001../ablation_seed_metrics.csv\u3002\u6bcf\u5f20\u56fe\u7684\u8f93\u5165\u6570\u503c\u53e6\u5b58\u4e3a\u540c\u540d_data.csv\uff1b\u4e3b\u5b9e\u9a8c\u548c\u8865\u5145\u5b9e\u9a8c\u5206\u7ec4\u89c1figure_groups.json\u3002\n\n\u672c\u6b21\u53ea\u66f4\u65b0\u7ed8\u56fe\uff0c\u4e0d\u91cd\u65b0\u5ba1\u8ba1\u3002\u6b64\u524d\u4e09\u6a21\u578b\u6570\u503c\u68c0\u67e5\u4fdd\u7559\u4e3anumerical_checks_previous_3_model_snapshot.json\uff0c\u4e0d\u80fd\u89c6\u4f5c\u672c\u6b21\u4e94\u6a21\u578b\u68c0\u67e5\u3002\n',encoding='utf-8')
print(json.dumps({'main_scenarios':main_count,'supplementary_scenarios':supp_count,'figures':[r['figure'] for r in captions]}))

