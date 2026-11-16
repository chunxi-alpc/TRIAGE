"""Produce the Chinese report and compact result figure from measured artifacts."""
from pathlib import Path
import json


def report(out):
    out=Path(out)
    read=lambda n:json.loads((out/n).read_text())
    cv,test=read('cv_summary.json'),read('test_summary.json')
    cfg,status=read('configuration.json'),read('status.json')
    text=['# E1 全证据正交补线性修正：任务 #70','',
          '本实验在原证据预算下比较全维线性修正与归档的 16 维 E1。下表均来自服务器缓存上的真实新拟合和预测。','',
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
           '改进判断同时依据 A/P 消减、C/F 保护和任务正确率；完整结果包括所有负向变化。各折先拼接再计算总体比率，2,000 次来源重采样使用相同抽样计算配对差。','',
           '## 已有 40 来源受控留出','',
           '该集合沿用历史 test 配对、自然节点和动态路径协议，与五折 Dataset A 分开报告。', '']
    text+=table(test)
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
