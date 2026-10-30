# 代理 C 审计与已完成结果

本次只在服务器执行，官方配置/代码/结果未被此代理修改。协议先写，后实现与测试。7项CPU测试通过；CPU复算读取confirm43条原benchmark参考病例，来自2个来源簇。医生裁定的通用临床三态对仍为0。

## 主要结果

根分类43/43、根覆盖率100%，但完整参考路径可达且正确仅12/43（27.91%），直接映射叶正确同为12/43。参考路径362个节点中，未应用门控的分类正确266，门控后正确194；不可达组编辑0，已保存plan与冻结参数重算0处不同。深度8及以上的参考路径节点门控正确数为0。这说明本确认集的主要瓶颈位于后续节点分类与误差传播；根正确不等于路径正确。以上“正确”均相对benchmark参考路径，不等于医生验证的临床适用性。

补充risk01：362个参考路径节点中225可达、198被接受（194正确、4错误），接受风险2.02%，接受覆盖54.70%；27个可达节点预测NONE，137个节点因祖先被阻断，阈值弃权0、接口失败0。31条未完成参考路径主要是27次NONE停止和4次错分，低阈值并没有消除语义NONE。组数仅为描述性计数，独立单位仍是来源簇。

现有生成：baseline19/43，spectral20/43，mean/random/reverse均19/43，text_clip25/43，direct_map12/43。spectral相对baseline纠正1、退化0，病例等权差异+2.326个百分点，来源等权+2个百分点；来源簇bootstrap区间[0,+4个百分点]仅为2簇下的探索性区间，不能据此确认稳定收益。text_clip纠正7、退化1，结果包含分类加文本限制后生成的整体能力，不能解释为分类器更准确。

## 真实bug与最小修复

|证据|发现|影响|最小修复建议/本分支实现|
|---|---|---|---|
|experiments/aster_manuscript_clinical_20261030/engine.py:36–46，尤其42|parse_prediction用dict.fromkeys静默去除重复规则；grade据此比较精确路径|循环/重复路径可被误判精确正确。CPU实际找到text_append的2例（匿名单元数2；详见独立scoring_differences.json）重复R1N，原结果exact_path=true。当前spectral未见此差异|保留原路径序列，重复/同节点冲突独立标invalid。c/audit.py strict_parse/strict_score独立修复；未改官方评分|
|src/aster/manuscript_core.py:329–338|不可达组且分类为None时reason记录classification_failure，而有预测时记录unreachable|reason是互斥摘要，单纯加总不能同时回答“不可达多少、接口失败多少”|分别按reachable和原始raw_decisions统计；本诊断独立记录不可达与classification_interface_failures，不改增益机制|
|experiments/aster_manuscript_clinical_20261030/report.py:7–11|来源簇bootstrap数学实现按簇采样正确，但未根据实际簇数限制解读|confirm仅2簇，窄区间不能代表广泛来源泛化|同时报告来源簇数与来源等权估计，将少簇区间标注exploratory；本脚本实现|

第一项为评分实现bug；第二项为报告字段信息损失；第三项为统计解读限制，不是算子bug。未发现祖先门控越界编辑或缓存plan与冻结参数不一致。概率阈值和margin阈值皆0不等于所有组都编辑：NONE与祖先不可达仍阻止编辑，exact tie仍弃权，CPU测试已验证。

## 独立GPU诊断

gpu_oracle.py提供同一新执行路径上的predicted/oracle两路，默认spectral与text_clip、confirm43，172个方法运行加2个零编辑不变性控制。首logit及输出零编辑等价通过才继续；不拿不同数值执行下的旧predicted与新oracle作纯路由因果对照。oracle只用于诊断，不是部署能力或新临床金标。结果和实测资源以独立输出目录的summary.json为准。

## 对论文主张建议

将“根分类准确”与“完整路径适用性”分开；现有冻结零阈值不能宣称有经过确认的选择性风险保证。将direct_map设为分类路径传递对照，固定路由生成与oracle路由诊断单列。对最终收益仅报告配对纠正/退化和来源簇区间，不能用当前2簇确认集确证普遍有效。注意力读数变化不自动等于叶节点任务改善。主稿算子保持原配对键差分、有向均值投影与缓存键平移；此代理没有混入PR41算子。

实际产物：/home/gcx/aster_abc_diagnostics_20261030/c/cpu01/summary.json；tests.log；可执行命令见README.md。GPU/人工新增裁定若未完成，必须在总报告中单列，不能声称完成。


## 已完成的匹配GPU路由诊断（2026-10-31）

独立运行oracle01实际完成172/172个方法单元，无技术失败，另完成2次零编辑路由不变性控制；两路首logit与生成词元完全一致，最大logit差0。实际耗时864.24秒（14.40分钟），峰值显存20.821GB。使用同一GPU执行路径、同一冻结增益0.25、同一生成预算512、同一分块eager注意力和capture=False；predicted与oracle都在独立诊断目录新生成，未把不同执行路径下的旧缓存当作纯路由对照。

|方法/路由|叶正确/43|精确路径正确/43|根正确/43|技术失败|
|---|---:|---:|---:|---:|
|spectral / predicted|20|21|43|0|
|spectral / 金标路径诊断|19|20|43|0|
|text_clip / predicted|25|22|43|0|
|text_clip / 金标路径诊断|43|43|43|0|

|金标路径减预测路由|纠正/退化|病例等权差异|来源簇等权差异|来源簇bootstrap 95%区间|
|---|---:|---:|---:|---:|
|spectral|0 / 1|−2.326个百分点|−2.000个百分点|[−4.000, 0.000]个百分点|
|text_clip|18 / 0|+41.860个百分点|+43.778个百分点|[+32.000, +55.556]个百分点|

每项配对成功覆盖率均100%；只有2个独立来源簇，区间只作探索性描述，不支持广泛来源泛化或显著性主张。43个病例、172个方法单元以及路径节点都不能当作独立来源样本。

### 结果解释与对归因的限制

金标路径输入没有改善本次固定主稿算子的最终叶正确率。spectral两路可见输入均为平均7290.79词元；金标路径使平均启用组数从4.65增至8.42，实际改键向量从969.86增至1877.02，但叶结果从20降至19。这排除了“仅把分类路由换成参考路径就足以恢复收益”的解释；不能据此断言所有增益或所有数据上的键编辑都无效。本次没有用确认集重选增益、阈值、头或方向。

text_clip金标路径结果43/43表明，在保留参考路径、去掉竞争分支后的提示上，模型可以完成当前输出任务。其平均输入从predicted的3694.56词元降至oracle的1750.63词元，并且oracle剪枝按完整参考路径保留规则，已经给出了通往参考终点的路径结构。因此，该结果是带参考路径的流程诊断，不是可部署系统准确率、独立临床适用性金标或分类器自身能力。spectral与text_clip之间还存在可见指南删除、竞争分支数量和提示长度的差异，不能把跨算子差异全部归因于路由。

这两种方法内的predicted/oracle比较均采用匹配执行，能够描述“改变路由计划在该固定算子下”的整体影响；路由计划同时改变编辑覆盖/剪枝结构，不能进一步拆成单独某个注意力头或某一步分类的因果效应。根两路均43/43，叶和精确路径仍有差异，论文应继续分开报告根、完整路径和终点结果。

### 复现与状态

输入：/home/gcx/aster_manuscript_20261030/clinical01 的冻结配置、expert、Qwen3-8B本地权重与confirm43；没有运行官方测试样本或新增临床标注。原官方配置与结果未修改。

实际输出：/home/gcx/aster_abc_diagnostics_20261030/c/oracle01/{manifest.json,reference.json,records.jsonl,status.json,summary.json}。records留在服务器，不公开病例文本。此诊断已完成，不需要继续监控或新实验。

如需独立复现须换新输出目录：

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_abc_diagnostics_20261030.c.gpu_oracle \
  --run /home/gcx/aster_manuscript_20261030/clinical01 \
  --cpu-audit /home/gcx/aster_abc_diagnostics_20261030/c/cpu01 \
  --output /home/gcx/aster_abc_diagnostics_20261030/c/oracle02
```

该命令仅供后续复现，本次没有重复运行。医生裁定的通用临床三态对仍为0。
