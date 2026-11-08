from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'meeting/figures'
font=Path('/home/gcx/.cache/aster_fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
'font.size':16,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path'})
m=json.loads((BASE/'meeting/evidence_preserving/metrics.json').read_text())
def vals(key):
 d=m['methods'][key]['splits']['dev']
 return 100*d['CF']['correction_rms_over_ap_rms'],100*d['AP']['variance_reduction']
keys=['bounded_'+str(x) for x in [0.,.05,.1,.25,.5,1.,2.]]
xy=[vals(k) for k in keys]
fig,ax=plt.subplots(figsize=(10.6,6.8))
ax.plot([x[0] for x in xy],[x[1] for x in xy],color='#A3AAB2',lw=1.6,zorder=1)
ax.scatter([x[0] for x in xy],[x[1] for x in xy],color='#A3AAB2',s=54,zorder=2,label='预算敏感性')
x,y=vals('bounded_0.1');ax.scatter(x,y,color='#537DA8',marker='*',s=240,zorder=4,label='固定预算 τ=0.1')
ax.annotate(f'τ=0.1\nVR {y:.2f}%',(x,y),xytext=(16,-1.5),textcoords='data',
arrowprops=dict(arrowstyle='-',lw=.8,color='#537DA8'),color='#334253',fontsize=14)
for key,label,off in [('bounded_0.25','τ=0.25',(-10,12)),('bounded_1.0','τ=1',(7,-18))]:
 x,y=vals(key);ax.annotate(label,(x,y),xytext=off,textcoords='offset points',fontsize=14)
x,y=vals('unconstrained_AP')
ax.scatter(x,y,color='#BD7D79',s=95,zorder=3,label='无证据约束')
ax.annotate(f'无约束\nVR {y:.2f}%',(x,y),xytext=(-8,-45),ha='right',textcoords='offset points',fontsize=14)
ax.set(xlabel='证据配对修正 RMS / 原干扰 RMS（%）',ylabel='干扰方差下降 VR（%）',
xlim=(-3,88),ylim=(-4,46),title='实测：保护证据需要控制对消幅度')
ax.grid(color='#E5E8EC',lw=.7,alpha=.8);ax.set_axisbelow(True)
ax.legend(frameon=False,loc='upper left',fontsize=14)
fig.text(.12,.02,'Qwen3-8B · repair-v2 开发集 · 同一16维特征 · 一次CPU缓存重拟合',fontsize=12,color='#556273')
fig.tight_layout(rect=(0,.045,1,1))
for ext in ['png','svg','pdf']:
 fig.savefig(OUT/('06_evidence_constraint_tradeoff.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
