from pathlib import Path
import csv,json,collections
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
    if 'llama' in s:return 'Llama3.1-8B'
    return 'MedGemma-27B'
def sort_key(key):
    model,task,scenario=key
    mm={'Qwen3-8B':0,'Llama3.1-8B':1,'MedGemma-27B':2}[model_label(model)]
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
selected_methods=['aster_cr','paired_n','lora']
keys=sorted({(r['model'],r['task'],r['scenario']) for r in main},key=sort_key)
lookup={(r['model'],r['task'],r['scenario'],r['method']):r for r in main}
selected=[lookup[(*key,method)] for key in keys for method in selected_methods]
dump_csv('01_simulated_main_vr_er_data.csv',selected)
vr=np.array([[100*number(lookup[(*key,m)],'variance_reduction_mean') for m in selected_methods] for key in keys])
er=np.array([[100*number(lookup[(*key,m)],'evidence_retention_mean') for m in selected_methods] for key in keys])
vr_sd=np.array([[100*number(lookup[(*key,m)],'variance_reduction_std') for m in selected_methods] for key in keys])
er_sd=np.array([[100*number(lookup[(*key,m)],'evidence_retention_std') for m in selected_methods] for key in keys])
fig,axs=plt.subplots(1,2,figsize=(17.5,8.8))
fig.subplots_adjust(left=.18,right=.94,top=.79,bottom=.12,wspace=.40)
fig.suptitle('\u6a21\u62df\u573a\u666f\u63a8\u6f14\uff1a\u5e72\u6270\u6291\u5236\u4e0e\u8bc1\u636e\u4fdd\u7559',fontsize=28,color=RED,y=.98,weight='bold')
fig.text(.5,.91,'\u672a\u8c03\u7528\u6a21\u578b\u3001\u672a\u4f7f\u7528\u771f\u5b9e\u4efb\u52a1\u6837\u672c\u3001\u672a\u5b9e\u73b0\u8fd9\u4e9b\u57fa\u7ebf\uff1b\u5355\u5143\u683c\u4e3a 5 \u4e2a\u6a21\u62df seed \u7684\u5747\u503c \u00b1 SD',ha='center',fontsize=15,color=DARK)
if vr.min()<0:
    norm1=TwoSlopeNorm(vcenter=0,vmin=min(-1,float(vr.min())),vmax=max(1,float(vr.max())))
    cmap1=LinearSegmentedColormap.from_list('vr',[PEACH,'#FFFFFF',BLUE])
else:
    norm1=Normalize(vmin=0,vmax=max(1,float(vr.max())))
    cmap1=LinearSegmentedColormap.from_list('vr',['#F4F6F9',BLUE])
norm2=TwoSlopeNorm(vcenter=100,vmin=min(95,float(er.min())-1),vmax=max(105,float(er.max())+1))
cmap2=LinearSegmentedColormap.from_list('er',[PEACH,'#FFFFFF','#B0BBCD'])
for j,(ax,mat,sd,title,cmap,norm) in enumerate(zip(axs,[vr,er],[vr_sd,er_sd],
    ['\u5e72\u6270\u65b9\u5dee\u4e0b\u964d VR\uff08%\uff0c\u8d8a\u9ad8\u8d8a\u597d\uff09','\u8bc1\u636e\u54cd\u5e94 ER\uff08%\uff0c\u76ee\u6807\u4e3a 100\uff09'],[cmap1,cmap2],[norm1,norm2])):
    im=ax.imshow(mat,aspect='auto',cmap=cmap,norm=norm)
    ax.set_xticks(range(3),['\u6700\u7ec8\u65b9\u6cd5\u60c5\u666f','\u914d\u5bf9\u5199\u56de\u60c5\u666f','LoRA \u60c5\u666f'])
    ax.set_yticks(range(len(keys)),[model_label(k[0])+' \u00b7 '+task_label(k[1]) for k in keys] if j==0 else ['']*len(keys))
    ax.set_title(title,pad=18)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            ax.text(x,y,f'{mat[y,x]:.1f}\n\u00b1 {sd[y,x]:.1f}',ha='center',va='center',fontsize=12,color='#243346')
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(keys),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04)
    cb.ax.tick_params(labelsize=12)
save(fig,'01_simulated_main_vr_er')

abl=read('ablation_seed_metrics.csv')
abl_methods=['full','without_w','without_b','no_orth_U','random_subspace','label_joint']
abl_labels=['\u5b8c\u6574\u65b9\u6cd5','\u53bb\u6389 w','\u53bb\u6389 b','\u4fdd\u7559 U \u4e2d\u8bc1\u636e','\u968f\u673a\u5b50\u7a7a\u95f4','\u6807\u7b7e\u8054\u5408\u8bad\u7ec3']
cols=[BLUE,GRAY,'#A7BFD7',PEACH,RED,'#AAA2C4']
agg=[]
for method,label in zip(abl_methods,abl_labels):
    rr=[r for r in abl if r['method']==method]
    seeds=sorted({r['seed'] for r in rr})
    for metric in ['variance_reduction','evidence_retention','native_accuracy']:
        seed_values=[np.mean([number(r,metric) for r in rr if r['seed']==seed]) for seed in seeds]
        agg.append({'status':'simulated','method':method,'method_label':label,'metric':metric,
          'scenario_count':len({r['scenario'] for r in rr}),'seed_count':len(seeds),
          'mean':float(np.mean(seed_values)),'std':float(np.std(seed_values,ddof=1)),
          'aggregation':'equal scenario weights within seed; sample SD over 5 seed means'})
dump_csv('02_simulated_ablation_data.csv',agg)
fig,axs=plt.subplots(1,3,figsize=(19,7.3))
fig.subplots_adjust(left=.065,right=.985,top=.77,bottom=.30,wspace=.29)
fig.suptitle('\u6a21\u62df\u6d88\u878d\uff1a\u5404\u7ec4\u4ef6\u6539\u53d8\u54ea\u4e9b\u6307\u6807',fontsize=28,color=RED,y=.98,weight='bold')
fig.text(.5,.90,'\u5171\u540c\u6f5c\u5728\u53d8\u91cf\u4e0a\u7684\u9884\u8bbe\u53d8\u6362\uff1b\u5404\u573a\u666f\u7b49\u6743\u6c47\u603b\u3002\u8bef\u5dee\u6761\u4e3a 5 \u4e2a\u6a21\u62df seed \u5747\u503c\u7684 SD\u3002',ha='center',fontsize=15,color=DARK)
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
caption=[
 {'figure':'01_simulated_main_vr_er','status':'simulated','caption':'\u6a21\u62df\u573a\u666f\u63a8\u6f14\uff0c\u975e\u771f\u5b9ebenchmark\u30023\u79cd\u6a21\u578b\u540d\u79f0\u00d73\u79cd\u4efb\u52a1\u540d\u79f0\u53ea\u662f9\u7ec4\u53c2\u6570\u914d\u7f6e\u3002\u6700\u7ec8\u65b9\u6cd5\u3001\u914d\u5bf9\u5199\u56de\u4e0eLoRA\u5747\u4e3a\u5171\u540c\u6f5c\u5728\u53d8\u91cf\u4e0a\u7684\u9884\u8bbe\u8bfb\u51fa\u53d8\u6362\uff0c\u672a\u5b9e\u9645\u5b9e\u73b0\u76f8\u5e94\u57fa\u7ebf\u3002VR\u8d8a\u9ad8\u8868\u793a\u914d\u5bf9\u5dee\u65b9\u5dee\u8d8a\u5c0f\uff1bER\u7684\u76ee\u6807\u662f100%\uff0c\u8d85\u8fc7100%\u8868\u793a\u54cd\u5e94\u653e\u5927\uff0c\u5e76\u975e\u989d\u5916\u4f18\u52bf\u3002\u6bcf\u683c\u4e3a5\u4e2a\u6a21\u62dfseed\u5747\u503c\u00b1\u6837\u672cSD\u3002\u8f83\u7eaf\u7684\u75d5\u8ff9\u3001\u8f83\u9ad8\u8bc1\u636e\u4fdd\u7559\u7387\u5747\u6765\u81ea\u989d\u5916\u6a21\u62df\u5047\u8bbe\uff0c\u4e0d\u80fd\u66ff\u6362\u771f\u5b9e\u5f00\u53d1\u96c694.64%\u7684\u8bc1\u636e\u54cd\u5e94\u4fdd\u7559\u7387\u3002'},
 {'figure':'02_simulated_ablation','status':'simulated','caption':'\u6a21\u62df\u6d88\u878d\uff0c\u975e\u771f\u5b9e\u6a21\u578b\u8bad\u7ec3\u7ed3\u679c\u3002\u6bcf\u4e2aseed\u5185\u5bf9\u53ef\u7528\u573a\u666f\u7b49\u6743\u5e73\u5747\uff0c\u518d\u8ba1\u7b975\u4e2aseed\u5747\u503c\u4e0e\u6837\u672cSD\u3002\u53bb\u6389b\u4fdd\u6301\u6240\u6709\u914d\u5bf9\u5dee\u5206\u6307\u6807\u4e0d\u53d8\uff0c\u53ef\u80fd\u6539\u53d8\u9608\u503c\u5224\u65ad\u51c6\u786e\u7387\u3002\u5176\u4f59\u6d88\u878d\u4f53\u73b0\u9884\u8bbe\u7279\u5f81\u3001\u566a\u58f0\u548c\u8bfb\u51fa\u65b9\u5f0f\u7684\u5dee\u5f02\u3002ER\u63a5\u8fd1100%\u4e3a\u7406\u60f3\uff1b\u672c\u56fe\u4e0d\u636e\u6a21\u62df\u51c6\u786e\u7387\u58f0\u79f0\u771f\u5b9e\u6570\u636e\u4e0a\u7684\u6027\u80fd\u63d0\u5347\u3002'}]
(OUT/'captions.json').write_text(json.dumps(caption,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'README.zh.md').write_text('# \u6a21\u62df\u7814\u7a76\u56fe\u8868\n\n\u6240\u6709\u56fe\u5747\u4e3a\u53c2\u6570\u5316\u573a\u666f\u63a8\u6f14\uff0c\u4e0d\u662f\u5b9e\u9645\u8c03\u7528\u6a21\u578b\u3001\u8fd0\u884cbenchmark\u6216\u5b9e\u73b0\u57fa\u7ebf\u5f97\u5230\u7684\u7ed3\u679c\u3002\n\n'+ '\n\n'.join(r['figure']+'\uff1a'+r['caption'] for r in caption)+'\n\n\u7ed8\u56fe\u6765\u6e90\uff1a../main_summary.csv\u548c../ablation_seed_metrics.csv\uff1b\u56fe\u5185\u4f7f\u7528\u7684\u6570\u503c\u53e6\u5b58\u4e3a\u540c\u540d_data.csv\u3002\n',encoding='utf-8')
print('Wrote two simulated figure sets and source CSV.')

