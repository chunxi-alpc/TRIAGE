# 五模型 × 三数据集 × 四类基线：真实实验与证据清单

本目录是全量实测的执行规范和核查器，**不是已经完成的五模型实验报告**。matrix_spec.json列出五个模型、三个数据集、13种外部方法、3项通用控制，以及原模型和ASTER-CR线性／非线性两个版本；matrix_status_285.csv是待评价状态快照。run_full_factorial.py能够调度已经注册的真实适配器，并从逐例预测核验结果，**但不实现或冒充13种外部算法本身**。

## 所有模型必须完整参加

| 模型 | 官方权重标识 | 本轮完整统一评测 |
|---|---|---|
| Qwen3-8B | Qwen/Qwen3-8B | 尚未完成；有部分历史MEGA／NLI结果 |
| Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | 尚未完成 |
| Gemma-2-9B-IT | google/gemma-2-9b-it | 尚未完成 |
| Gemma-3-27B-IT | google/gemma-3-27b-it | 尚未完成 |
| MedGemma-27B-Text-IT | google/medgemma-27b-text-it | 尚未完成 |

原有configs/models.json记录Qwen、Llama和MedGemma权重曾下载，不代表当前服务器的文件已现场核验；Gemma-2及Gemma-3未列在现有模型清单。当前可执行环境没有GPU、五套权重或服务器访问，GitHub仓库也没有可直接调度此次任务的工作流，故本轮没有新增GPU模型成绩。

三项正式测试：MEGA-CDP官方7,458条动态路径，NLI4CT-2024正式测试5,500条声明，MedPIC官方467题、2,335选项。MEGA另测受控留出2,293个A/P干扰对和40个C/F证据对；NLI另测500原声明的宏平均F1、4,136个标签保持关系与864个标签翻转关系；MedPIC的历史244个翻转选项对和216个保持选项对须重新核验逐例映射，不能冒充官方原生配对总体。

**每一项外部方法或通用控制都须在每个模型、每个数据集上评价**，即5×3×16=240个核心比较单元。原模型、ASTER-CR线性和非线性另外占45个单元，合计285个。两个27B模型不能只放模拟或附录。

## 主实验协议

所有五个模型分别在MEGA来源隔离的fit80训练、dev20选择；拟合池含4,887个A/P及135个C/F配对。方法、权重和阈值冻结后，分别在MEGA官方测试、NLI正式测试和MedPIC官方测试上评分。三个任务的固定提示模板可以不同，但不能在NLI或MedPIC测试标签上调参。尤其MedPIC没有独立官方训练集，不能从467道测试题里取一部分训练，再把全467道当独立测试。

若另用NLI训练集1,700条标签适配，必须称作任务内补充实验；若批量校准使用NLI或MedPIC目标批次的无标签分布，也须单列转导协议。不能把两者与源任务训练、冻结迁移的主轨混排。模型之间不得共享Qwen的头索引、痕迹基、阈值或权重；每个主干分别训练和选择方法。LEACE、R-LACE与非线性LEOPARD使用拟合集冻结的干扰概念定义；额外新读出的KRaM、kSAL、核化对抗擦除及Obliviator单列。

所有模型使用各自官方聊天模板。Qwen显式关闭思考模式。答案可能经过不同数量的词元；比较时必须采用可复核的完整答案序列条件对数似然，不能直接沿用Qwen单词元标签假设。MEGA必须按方法预测动态遍历；NLI保持标签不代表证据未变；MedPIC整题集合全对不能由单项准确率推导。

按照论文附录B.7，每种方法、每个模型最多搜索12个完整配置，开发集预先固定证据代价档；确定配置后的随机训练使用种子11、22、33。不同数据集报告自己的原生任务指标、干扰响应、证据改动、双端配对正确、方向退化、证据预算超限和来源簇不确定性，不能跨模型直接平均原始logit或将任务得分相加。

## 运行与检查

从仓库根目录先执行：

    python reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/run_full_factorial.py check

仅检查组合数量是否严格为240与285，**不运行模型**。随后执行：

    python reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/run_full_factorial.py doctor --model-root /path/to/model-links --data-root /path/to/input-manifests --providers /path/to/verified_provider_commands.json

模型目录由模型键标识，每个目录必须包含config.json。必要时把服务器上已核验的模型目录映射为同名符号链接。数据目录需提供mega_cdp_official_test.jsonl、nli4ct_2024_official_test.jsonl、medpic_official_test.jsonl；现有服务器资料原本分散于/home/gcx/aster_next_20261109和历史MedPIC目录，必须先核对来源、清单和授权，不能盲目复制标签到模型提示。

方法提供程序配置为JSON字典，键是matrix_spec.json中的方法标识，值是已经真实实现且通过烟测的命令参数数组。可用模板参数有model_id、model_path、dataset_id、method_id、input_path、output_dir、run_id及seed，均写在命令字符串的大括号中。**本目录没有为不存在的算法编造适配器或预测**。如果某项方法未提供真实可执行程序，运行会以缺少提供程序报错，不能算已评价。

示例：已注册LEACE程序后可运行一个真实单元：

    python reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/run_full_factorial.py run --model-root /path/to/model-links --data-root /path/to/input-manifests --providers /path/to/verified_provider_commands.json --results /path/to/real_results --model qwen3_8b --dataset mega_cdp --method leace --seed 11

不带筛选器时将覆盖285项，但必须先有对应的真实适配器、完整模型和输入。随机训练选定配置后可另运行种子22和33，不能根据附加种子选择有利结果。

## 必须保存且可复核的产物

成功任务保存于结果根目录下的模型键／数据集键／方法键／seed11／metrics.json，包含身份、执行成功状态、明确的真实模型执行来源、模型修订、算法代码修订、拟合与开发来源、输入清单文件和SHA-256、评分协议、训练种子及指标。同时保存predictions.jsonl，逐例只需case_id、gold、prediction；它须与metrics.json记录的预测文件路径和SHA-256一致。

对MEGA，gold与prediction应是执行顺序一致的路径列表；对NLI是两个类别之一；MedPIC是整题选项集合。核查器从逐例重算正确数、检查ID唯一、输入与预测文件哈希、任务的官方分母及所需机制指标；不能只手填百分比。MedPIC集合比较忽略选项排列顺序，MEGA路径必须保留顺序。MedPIC历史244翻转关系若逐例核验后变化，应先更新关系清单、明确说明，而不是为了满足脚本修改成绩。

执行真实结果校验：

    python reports/aster_cancel_story_20261109/meeting/full_factorial_20261110/run_full_factorial.py collect --results /path/to/real_results --summary /path/to/results_verified.csv

输出只将完整通过核验的单元标为verified_real_run，其余区分invalid_result与not_run。矩阵初始状态即使含Qwen历史部分结果，也不能代替五模型共用协议的确认性成绩。

## 当前证据边界

Qwen3-8B在MEGA受控806路径上已有仅截距、监督线性读出、LEACE、ITI、LoFiT和LoRA的旧冻结分数；LoReFT旧缓存与原模型一致，适配器生效尚未核验。Qwen的NLI正式测试另有原模型、仅截距和ASTER-CR冻结结果。MedPIC目前主要是未归档逐例文件的历史汇总；其余四个模型的完整四类基线在三个主数据集上没有本轮可验证的运行结果。

**没有运行不是模型失效，缺失不是数值零。** 新增论文结果必须来自上述真实执行和核验记录，不能拿文献指标、假设分数或Qwen结果填进其他模型的单元。