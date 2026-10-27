"""P2 paired contrasts; only this run's complete dev records enter analysis."""
from pathlib import Path
import collections,csv,json,random,shutil,statistics
from run_p1 import dump
BASE=Path(__file__).resolve().parent

def ci(bytree):
 rng=random.Random(20261027);trees=sorted(bytree);out=[]
 for _ in range(5000):
  x=[v for t in rng.choices(trees,k=len(trees)) for v in bytree[t]];out.append(100*sum(x)/len(x))
 out.sort();return out[124],out[4874]
def writecsv(p,rs):
 if rs:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 res=BASE/'results/p2';res.mkdir(parents=True,exist_ok=True);summ=[];bins=[];contrasts=[];raw=[];states={}
 for model in ['qwen','llama']:
  src=BASE/'private'/('p2_'+model)
  if not (src/'DONE.json').exists():states[model]='not_completed';continue
  states[model]='completed';rs=[json.loads(s) for s in (src/'predictions.jsonl').read_text().splitlines()];assert len(rs)==json.loads((BASE/'results/data_audit.json').read_text())['base_count']*3*3*2
  cb=[json.loads(s) for s in (src/'closed_book.jsonl').read_text().splitlines()];cbmap={r['id']:r for r in cb};raw+=rs
  shutil.copyfile(src/'closed_book.jsonl',res/(model+'_closed_book.jsonl'));shutil.copyfile(src/'manifest.json',res/(model+'_manifest.json'));shutil.copyfile(BASE/'private'/('p2_'+model+'.log'),res/(model+'_run.log'))
  index={(r['id'],r['kind'],r['control'],r['condition']):r for r in rs};ids=sorted(cbmap)
  for kind in ['familiar','unfamiliar','neutral']:
   for control in ['standard','no_options','reversed']:
    b=[index[cid,kind,control,'blind'] for cid in ids];v=[index[cid,kind,control,'visible'] for cid in ids];bt=collections.defaultdict(list)
    for x,y in zip(b,v):bt[x['tree']].append(int(x['correct'])-int(y['correct']))
    low,high=ci(bt)
    summ.append({'model':model,'kind':kind,'control':control,'n':len(ids),'blind_accuracy_pct':100*sum(x['correct'] for x in b)/len(b),'visible_accuracy_pct':100*sum(x['correct'] for x in v)/len(v),'blind_minus_visible_pp':100*sum(int(x['correct'])-int(y['correct']) for x,y in zip(b,v))/len(b),'cluster_ci_low_pp':low,'cluster_ci_high_pp':high,'parse_failures':sum(x['parse_failure'] for x in b+v)})
   for stratum in ['low','middle','high']:
    ii=[cid for cid in ids if cbmap[cid]['stratum']==stratum];b=[index[cid,kind,'standard','blind'] for cid in ii];v=[index[cid,kind,'standard','visible'] for cid in ii]
    bins.append({'model':model,'kind':kind,'stratum':stratum,'n':len(ii),'confidence_min':min(cbmap[cid]['original_action_probability'] for cid in ii),'confidence_max':max(cbmap[cid]['original_action_probability'] for cid in ii),'mean_confidence':statistics.mean(cbmap[cid]['original_action_probability'] for cid in ii),'blind_minus_visible_pp':100*sum(int(x['correct'])-int(y['correct']) for x,y in zip(b,v))/len(ii)})
  for control in ['standard','no_options','reversed']:
   bt=collections.defaultdict(list)
   for cid in ids:
    f=index[cid,'familiar',control,'visible'];n=index[cid,'neutral',control,'visible'];bt[f['tree']].append(int(n['correct'])-int(f['correct']))
   low,high=ci(bt);contrasts.append({'model':model,'control':control,'n':len(ids),'neutral_minus_familiar_accuracy_pp':100*sum(sum(x) for x in bt.values())/len(ids),'cluster_ci_low_pp':low,'cluster_ci_high_pp':high})
 writecsv(res/'summary.csv',summ);writecsv(res/'confidence_strata.csv',bins);writecsv(res/'neutral_vs_familiar.csv',contrasts)
 (res/'raw_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
 complete=all(s=='completed' for s in states.values());neutral=[r for r in summ if r['kind']=='neutral' and r['control']=='standard'];familiar=[r for r in summ if r['kind']=='familiar' and r['control']=='standard']
 gap=statistics.mean(r['blind_minus_visible_pp'] for r in neutral) if complete else None
 status='STOP_VISIBILITY_NARRATIVE' if complete and gap>=3 else 'PASS_NEUTRAL_GATE' if complete else 'INCOMPLETE'
 gate={'P2':status,'models':states,'neutral_model_macro_gap_pp':gap,'familiar_model_macro_gap_pp':statistics.mean(r['blind_minus_visible_pp'] for r in familiar) if complete else None,'large_neutral_threshold_pp':3,'unfamiliar_is_proxy_only':True,'P3':'NOT_RUN','P4':'NOT_RUN','P5':'NOT_RUN','P6':'NOT_RUN','test_inference_count':0}
 dump(res/'gate_status.json',gate)
 lines=['# P2 诱饵对照实测','','P1 SW 门槛通过后才启动本轮。Llama 修正单 BOS 后重新推理；Qwen 仅复用逐词元与权重核对一致的历史输出。全部为相同的 169 个修订开发病例；每模型 3 类诱饵 × 3 个控制条件 × 2 种可见性，另做 169 次闭卷前向。','',f'状态：**{status}**；中性代号的两模型平均“盲判−明判”差距为 {gap} 个百分点。','','| 模型 | 诱饵 | 控制 | 盲判 % | 明判 % | 盲判−明判（百分点） |','|---|---|---|---:|---:|---:|']
 for r in summ:lines.append(f"| {r['model']} | {r['kind']} | {r['control']} | {r['blind_accuracy_pct']:.2f} | {r['visible_accuracy_pct']:.2f} | {r['blind_minus_visible_pp']:+.2f} |")
 lines+=['','| 模型 | 诱饵 | 闭卷置信度档 | 例数 | 盲判−明判（百分点） |','|---|---|---|---:|---:|']
 for r in bins:lines.append(f"| {r['model']} | {r['kind']} | {r['stratum']} | {r['n']} | {r['blind_minus_visible_pp']:+.2f} |")
 lines+=['','实施范围：','','- 熟悉处置使用来源原正确动作；陌生处置采用另一开发树且不在本题选项内的动作。这只是预先固定的操作性代理，未独立证明该动作对模型陌生。','- 中性代号按模型匹配完整可见提示的 token 数；病例、条件、问题、选项和另一分支的处置不变。代号只有字母序列，不代表真实治疗。','- 去选项条件仍保留原问题；顺序控制只反转显示次序，分支编号与金标不变。','- 闭卷置信度是选项字母集合内归一化的原动作概率；按模型分成三等份，不按置信度剔除病例。它不是模型校准后的绝对正确概率。','- 全部解析失败计错，原始逐例输出和概率均保留；相同提示仅计算一次。区间按 4 个树聚类，不能作为正式测试结论。','- “中性差距较大”以 3 个百分点作本轮事前操作定义。模型间差异和熟悉−中性直接对照单列，不用平均数隐去。']
 if status=='STOP_VISIBILITY_NARRATIVE':lines+=['','**触发 P2 止损：中性代号仍产生较大差距。不能将 P1 的全部效应归因为熟悉处置。应先转为可见性／提示信息影响的叙事；严格按用户要求，不进入 P3–P6。**']
 (BASE/'P2_REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(gate))
if __name__=='__main__':main()
