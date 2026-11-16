"""Produce the Chinese report and compact result figure from measured artifacts."""
from pathlib import Path
import json


def report(out):
    out=Path(out)
    read=lambda n:json.loads((out/n).read_text())
    cv,test=read('cv_summary.json'),read('test_summary.json')
    cfg,status=read('configuration.json'),read('status.json')
    metric=lambda data,method,key:data['models'][method]['metrics'][key]['estimate']
    counts=lambda data,method:data['models'][method]['counts']
    a,b=counts(cv,'E1'),counts(cv,'full_complement')
    def change(data,key):
        d=data['full_minus_archived_E1'][key]
        return f"{100*d['estimate']:+.3f} 个百分点，95% 来源区间 [{100*d['low']:.3f}, {100*d['high']:.3f}]"
    cf_over=metric(cv,'full_complement','cf_over_budget_rate')
    outcome=('本次结果未达到同时改善干扰消减并维持留出证据保护的目标。'
             if cf_over>metric(cv,'E1','cf_over_budget_rate') else '逐项结果及配对区间见下表。')
    text=['# E1 全证据正交补线性修正：任务 #70','',
          f'五折 A/P VR 差为 {change(cv,"ap_vr")}。全部训练约束满足原预算；外折 C/F 超预算比例为 {cf_over:.2%}。{outcome}下表均来自服务器缓存上的真实新拟合和预测。','',
          '## 五折 Dataset A 结果','',
          '80 来源、2,540 对；每折 64 来源拟合、16 来源留出。归档 E1 的 Dataset B 配对预测按 Dataset A 配对 ID 提取为相同评价集合；旧方法未重新拟合或评分。', '']
    labels=[('ap_score_change_rms','A/P RMS'),('ap_vr','A/P 中心化 VR'),('cf_er','C/F ER'),
            ('cf_correction_rms','C/F 修正 RMS'),('cf_over_budget_rate','C/F 超预算比例'),
            ('cf_direction_degradation_rate','C/F 方向退化比例'),('ap_paired_accuracy','A/P 双端全对'),
            ('cf_paired_accuracy','C/F 双端全对'),('natural_node_accuracy','自然节点正确率'),
            ('natural_path_accuracy','自然路径正确率'),('interference_path_accuracy','干扰路径正确率')]
    def table(data):
        t=['|指标|归档 E1|全维修正 [95% 来源区间]|全维−E1 [95% 来源区间]|','|---|---:|---:|---:|']
        for k,title in labels:
            a=data['models']['E1']['metrics'][k];b=data['models']['full_complement']['metrics'][k];d=data['full_minus_archived_E1'][k]
            t.append(f"|{title}|{a['estimate']:.6f}|{b['estimate']:.6f} [{b['low']:.6f}, {b['high']:.6f}]|{d['estimate']:.6f} [{d['low']:.6f}, {d['high']:.6f}]|")
        return t
    text+=table(cv)
    vr=cv['models']['full_complement']['metrics']['ap_vr']['estimate']
    oldvr=cv['models']['E1']['metrics']['ap_vr']['estimate']
    cf=cv['models']['full_complement']['metrics']['cf_over_budget_rate']['estimate']
    text+=['',f'全维修正的外折 A/P VR 为 {vr:.2%}，归档 E1 为 {oldvr:.2%}；C/F 外折超预算比例为 {cf:.2%}。训练约束和留出来源的证据保护分别统计。',
           f'干扰路径正确数变化 {b["interference_path_correct"]-a["interference_path_correct"]:+d}/{b["interference_path_count"]}，正确率差为 {change(cv,"interference_path_accuracy")}。自然路径正确数变化 {b["natural_path_correct"]-a["natural_path_correct"]:+d}/{b["natural_path_count"]}，自然节点变化 {b["natural_node_correct"]-a["natural_node_correct"]:+d}/{b["natural_node_count"]}。区间跨零的指标按点估计改善报告。',
           f'C/F 修正 RMS 为归档 {metric(cv,"E1","cf_correction_rms"):.4f}、全维 {metric(cv,"full_complement","cf_correction_rms"):.4f} logit，{b["cf_over_budget_count"]}/{b["cf_pairs"]} 对超过原预算，最大修正为 {b["cf_max_abs_correction"]:.4f} logit，出现 {b["cf_direction_degradation_count"]} 对方向退化。有向平均 ER 为 {metric(cv,"full_complement","cf_er"):.4%}，逐对预算指标显示修正的具体幅度。',
           '三个候选的开发分数和六个模型所选 λ 全部保留。完成后未扩大网格或追加变体。当前结果显示全维线性修正在留出来源上的证据保护泛化问题。',
           '各折先拼接再计算总体比率，2,000 次来源重采样使用相同抽样计算配对差。这些来源沿用历史研究集合，外折来源隔离按原协议保持。','',
           '## 已有 40 来源受控留出','',
           f'该集合沿用历史 test 配对、自然节点和动态路径协议，与五折 Dataset A 分开报告。A/P {counts(test,"full_complement")["ap_pairs"]} 对、C/F {counts(test,"full_complement")["cf_pairs"]} 对（覆盖 {counts(test,"full_complement")["cf_sources"]} 来源）、F0 {counts(test,"full_complement")["f0_pairs"]} 对。F0 按历史单独类别保留。', '']
    text+=table(test)
    text+=['', f'40 来源留出的 VR 差为 {change(test,"ap_vr")}；自然路径正确率差为 {change(test,"natural_path_accuracy")}；干扰路径正确率差为 {change(test,"interference_path_accuracy")}。C/F 超预算为 {counts(test,"full_complement")["cf_over_budget_count"]}/{counts(test,"full_complement")["cf_pairs"]}，方向退化 {counts(test,"full_complement")["cf_direction_degradation_count"]} 对。',
           '', '## 各折配对结果', '', '|折|A/P 对|C/F 对|归档 VR|全维 VR|全维 C/F 超预算|','|---|---:|---:|---:|---:|---:|']
    from .run import rows, csvwrite
    from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as old
    fold_rows=[]
    for k in range(1,6):
        rr=rows(out/f'evaluation/fold{k}/pairs.jsonl.gz')
        boot=old.SourceBootstrap([r['source_id'] for r in rr])
        for method in ('E1','full_complement'):
            mm,cc,_=old.pair_statistics([r for r in rr if r['method']==method],boot)
            fold_rows.append(dict(fold=k,method=method,**(cc|{key:float(v[0]) for key,v in mm.items()})))
        a,b=fold_rows[-2:]
        text.append(f"|{k}|{b['ap_pairs']}|{b['cf_pairs']}|{a['ap_vr']:.2%}|{b['ap_vr']:.2%}|{b['cf_over_budget_rate']:.2%}|")
    csvwrite(out/'fold_results.csv',fold_rows)
    text+=['','## 方法、选参和数值求解','',
           '复用每折及 full 的 heads、mu、scale 和秩 16 的 S；以双精度 QR 重正交 S 的相同列空间。特征为 r=z−S(Sᵀz)，保留全部 6,528 个坐标，不读取或估计 U，不截断 A/P 特征。',
           '目标为原 E1 加权 A/P 平方差与 λ‖w‖²，每一条拟合 C/F 配对均约束 |Δrᵀw|≤0.2923349。类别及类内来源权重不变，A/P 权重总和为 4,887。',
           '对偶求解用完整 A/P 行空间 QR 应用 Q⁻¹，只分解配对规模矩阵；没有 6,528×6,528 Hessian 或高维 SLSQP。λ>0 保证唯一原始解，w=0 严格可行，满足强对偶条件。实现同时记录原始约束残差、驻点残差、对偶间隙和 OSQP 收敛状态。',
           '固定 λ 网格为 0.01、1、100。每折训练来源内部划分 48/16，full 内部划分 64/16。按开发集 C/F 超预算比例、方向退化比例、|ER−1|、修正 RMS、A/P RMS、较大 λ 的字典序选择。内部选择条件于复用的外层 M1/M2，M1/M2 的来源范围包含内部开发来源；外层留出来源从未用于对应折的 M1/M2/M3 或选参。',
           '输出修正在训练自然节点均值上中心化，并在同一批训练自然节点拟合 BCE 截距。这与原 E1 的独立截距校准等价；中心化与截距的常数均在配对差分中抵消。M4 使用原 score>0 决策及动态树路径执行。','',
           '|模型|λ|训练约束最大超额|相对对偶间隙|求解秒数|','|---|---:|---:|---:|---:|']
    for name in [f'fold{k}' for k in range(1,6)]+['full']:
        d=read(f'models/{name}/done.json');s=d['solver']
        text.append(f"|{name}|{d['selected_lambda']}|{s['constraint_max_violation']:.3g}|{s['relative_duality_gap']:.3g}|{s['seconds']:.2f}|")
    text+=['','所有 λ 尝试、开发分数和选择理由见 models/*/selection.json；每次求解的所有 C/F 约束见 *_constraints.csv，数值诊断见 *_solver.json。',
           '', '## 运行与复现','',f"代码版本：`{cfg['base_commit']}`。依赖：`{cfg['dependencies']}`。",
           f"状态：{status['state']}；实际运行 {status['seconds']/60:.2f} 分钟；进程峰值 RSS {status['peak_RSS_GiB']:.2f} GiB；GPU 显存 0、新模型前向 0。",'',
           '```bash','source /home/gcx/gcx261014/bin/activate',
           'export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8',
           'python -m unittest experiments.aster_full_complement_70.test_solver',
           'python -m experiments.aster_full_complement_70.run --out reports/aster_full_complement_70','```','',
           f"缓存根目录：`{cfg['cache_root']}`。读取 models/E1/{{full,fold*}}/m2.npz、folds.json、data、runtime/fit_heads.npy 和 cache/features；路径结构读取原 historical_root/data/path_cases.jsonl。原始病例文本和大型激活留在服务器，发布包含数值、ID、模型小型参数、测试、配置和图表。",'',
           '运行前检查 main 的 experiments/reports 和服务器已有 M3 联合实验，未发现同构的全维 S⊥ 加逐对硬预算完整结果。历史 16/32/99 维及其他任务模型未运行。',
           '', '## 结果文件','',
           '- cv_summary.json / test_summary.json：全部指标、来源区间和配对差。',
           '- evaluation/*：逐配对、逐节点、逐路径及全输入分数。',
           '- cv_source_metrics.csv / test_source_metrics.csv：逐来源指标。',
           '- models/*：所有超参尝试、来源划分、最终参数、每条训练约束和求解诊断。',
           '- comparison.png：五折主要结果。',
           '', '求解器接口与收敛定义参考 [OSQP 官方文档](https://osqp.org/docs/solver/index.html)。']
    (out/'REPORT.zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,(k,title) in zip(axes,[('ap_vr','A/P centered VR'),('cf_over_budget_rate','C/F budget exceedance'),('interference_path_accuracy','Interference path accuracy')]):
        vals=[cv['models'][m]['metrics'][k]['estimate']*100 for m in ('E1','full_complement')]
        ax.bar(['Archived E1','Full complement'],vals,color=['#8da0cb','#66c2a5']);ax.set_title(title);ax.set_ylabel('%')
        for i,v in enumerate(vals):ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    import sys
    report(Path(sys.argv[1]))
