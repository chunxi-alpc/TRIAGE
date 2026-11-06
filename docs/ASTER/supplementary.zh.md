# 旧版处置盲表示干预：方法补充与复现

> 本文对应可训练的处置盲表示干预及89.21%的旧版结果。最新可学习 M3 方案见[当前正文](manuscript.zh.md)，原闭式方案见[历史正文](manuscript-closed-form.zh.md)。下文“主方法”和“论文”均指旧版。

主方法在 Qwen3-8B 的第 18 层修改即将产生 Y/N 选择的查询表示。参考来自同一模型对患者、祖先条件和当前条件的另一次前向。模型只取参考的隐藏表示参与干预。测试时按实际选中的分支继续遍历，最终处置取自预测终点。公式与训练目标见[论文第 3 节](manuscript-representation.zh.md#3-方法)。

冻结主方法使用 `no_js` 检查点。对已经生成的路径前缀，解码器按指南树得到合法后继。只有一个合法延续时直接推进，遇到 Y/N 分叉时从前缀读出当前节点，取该节点的盲参考，并编辑产生 Y/N 分数的最后一个查询位置。实际路径上的每个分叉都调用干预。[分支定位与参考读取](../../experiments/aster_representation/engine.py#L145-L168)

盲读返回隐藏表示和诊断用的 T/F/U 概率。主方法只把指定层的隐藏表示传给编辑算子。算子将它映射为参考目标，再与完整表示的子空间坐标一起计算门控，最后按门控比例进行局部替换。门控值是替换系数，实际编辑幅度还取决于两个坐标之间的距离。[强度计算与局部编辑](../../experiments/aster_representation/intervention.py#L19-L30)

| 组成 | 实现 |
|---|---|
| 条件提示、全指南提示、处置重排和合法路径 | [inputs.py](../../experiments/aster_representation/inputs.py) |
| 正交子空间、参考映射、门控及损失 | [intervention.py](../../experiments/aster_representation/intervention.py) |
| 参考读取、决策位置干预及缓存更新 | [engine.py](../../experiments/aster_representation/engine.py) |
| 原指南和重排视图的监督 | [training.py](../../experiments/aster_representation/training.py) |
| 正式评测使用的四个参数张量 | [no_js_layer18_epoch3.safetensors](../../experiments/aster_representation/no_js_layer18_epoch3.safetensors) |

## 训练与干预细节

普通训练病例使用金标路径的根、中间和末端决策节点，重复节点只计一次。两视图的分支交叉熵取平均。保持项在未编辑模型已选对的视图上取 $\mathrm{KL}(p_0\Vert p_\theta)$ 的均值，无符合条件的视图时取零。幅度项在两视图上平均 $\|\widetilde o_g-o_g\|_2^2/\max(\|o_g\|_2^2,10^{-8})$。两项权重依次为 0.1 和 0.001，节点损失在病例内平均。全树终点处置的循环置换保留全树处置集合，各中间节点子树的处置集合可能发生变化。

子空间基通过可微 QR 分解保持正交，强度初值为 0.01。编辑在决策位置使用临时 KV 缓存。取得分支分数后丢弃临时缓存，再由未编辑模型编码实际选中的分支词元，形成下一决策的历史。

## 冻结配置

含 JS 的版本在开发集上比较第 18、24、30 层，选出第 18 层。普通低秩、上下文和无 JS 版本沿用该层位。各轮检查点先比较开发集原指南的路径与终点同时正确数，若相同则依次比较原指南与重排视图的路径一致率、原指南的平均编辑幅度（越小越优）。主方法取第 3 轮。开发集与内部确认集都参与了主方法版本的选择。

训练集包含 200 个普通病例，另有 8 组年龄最小对，共 16 个不同的患者状态。每个病例的贡献按节点数平均。AdamW 学习率为 0.001，权重衰减为 0.01，单病例批次，梯度累积 8，梯度范数上限为 1，训练 3 轮。骨干为 BF16，干预参数为 FP32。盲参考和完整前向均关闭思考模式。路径解码只允许当前节点的合法转移，最多生成 512 个词元。

## 使用冻结权重复现官方单轮结果

从仓库根目录运行。环境采用 Python 3.12、PyTorch 2.8.0 和 Transformers 4.57.6。准备 [MEGA-CDP 数据](https://huggingface.co/datasets/nchennnn/MEGA-CDP) 和 [Qwen3-8B 固定版本](https://huggingface.co/Qwen/Qwen3-8B/tree/26028140be3ee69b82b1d1450179ab71bb1121b9) 的本地目录。数据目录下应包含 `test/` 和 `train_val/`。随代码发布的权重只包含学习得到的干预参数。

```bash
python -m pip install -r experiments/aster_representation/requirements.txt
MEGA=/path/to/MEGA-CDP
MODEL=/path/to/Qwen3-8B
python -m experiments.aster_representation prepare --dataset "$MEGA" --split official_test --output work/test.jsonl
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_representation predict --model "$MODEL" --cases work/test.jsonl --output work/no_js.jsonl
python -m experiments.aster_representation score --cases work/test.jsonl --predictions work/no_js.jsonl
```

`predict` 默认加载仓库内的冻结权重。患者、指南和问题进入预测，金标路径进入评分。若已有冻结实验的预测，直接运行 `score`，其 `--predictions` 也接受存放原实验分片 JSONL 的目录。该命令报告严格完整路径正确率。论文对应计数为 **6,653／7,458**。表1的 DTW-PCC 已使用本地 BERTScore 完成全量评价。Acc 使用单独的结论语义评分，表中注明有效例数。

## 从训练病例复现干预参数

数据准备按[登记的病例编号](../../experiments/aster_representation/splits.json)从公开数据重建病例。年龄最小对按原病例和登记的年龄改动重建。以下命令复现选定层位的训练。不同轮次的权重写入新的输出目录。

```bash
python -m experiments.aster_representation prepare --dataset "$MEGA" --split train --output work/train.jsonl
python -m experiments.aster_representation prepare --dataset "$MEGA" --split age_train --output work/age_train.jsonl
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_representation train --model "$MODEL" --cases work/train.jsonl --age-cases work/age_train.jsonl --output work/trained
```

用训练所得权重执行预测时，向 `predict` 传入 `--weights work/trained/epoch3.safetensors`。开发集和内部确认集可分别用 `prepare --split development` 和 `prepare --split confirmation` 重建。

## 官方测试的配对净提升

主方法与普通低秩逐例比较。原指南中分支序列完全正确记为1，无效输出和技术失败记为0。正向不一致497例，反向不一致127例，因此净提升为 $(497-127)/7458=4.96\%$，即4.96个百分点。两个方法的配对表如下。

| 普通低秩 | 主方法正确 | 主方法错误 |
|---|---:|---:|
| 正确 | 6,156 | 127 |
| 错误 | 497 | 678 |

[来源簇计数](results/pathacc-by-source.csv)按对照方法和来源簇保留这四个格子；[结果数据](results/single-turn.json)同时提供来源等权结果、中英文分组和去除历史暴露标记后的计数。按病例加权的来源簇重采样区间为4.01至5.94个百分点，来源等权的净提升为4.13个百分点（95%区间3.30至4.99）。使用公开计数即可在 CPU 上复算：

```bash
python -m experiments.aster_representation.paired --source-counts docs/ASTER/results/pathacc-by-source.csv --control ordinary
```

如已有两个方法的逐例预测，也可以从原始单轮输出复算。下面的目录可包含多个方法的 JSONL 分片，或者将两个文件作为独立参数传给 `--predictions`：

```bash
python -m experiments.aster_representation.paired --cases work/test.jsonl --predictions work/official_predictions --control ordinary
```

此入口以登记病例为单位，读取原指南视图，用严格路径解析器重算正确与否。来源簇按病例清单中的首次出现顺序记录。区间使用 NumPy 随机种子20261102进行2,000次来源簇重采样。[配对统计实现](../../experiments/aster_representation/paired.py)

## 交换、事实翻转与同一表示的对照

开发集和内部确认集中的交换采用全树终点文本循环置换。在40棵不同的树中，根节点的处置多重集合全部保留，但829个非根节点中有819个节点的局部处置多重集合发生变化。因此，该扰动检验的是全树重排后的正确性。在相同病例上，以下列出主方法与概率上下文的正反向配对数：

| 数据与判据 | 配对数 | 仅主方法正确 | 仅概率上下文正确 |
|---|---:|---:|---:|
| 开发集，原指南与重排视图均正确 | 100 | 12 | 1 |
| 内部确认集，原指南与重排视图均正确 | 100 | 22 | 4 |
| 开发集，限原指南两方法均正确，再比较交换后 | 84 | 4 | 0 |
| 内部确认集，限原指南两方法均正确，再比较交换后 | 72 | 6 | 3 |

已核验的年龄最小对中，主方法与概率上下文都在原指南上答对全部8组训练对的两个状态。交换后分别为8/8和7/8。两组历史诊断对在两个视图中均为2/2，现有年龄诊断的规模不足以区分两种方法。

同一表示实验另取18个未参与选参的来源，每个来源一个节点，使用单事实构造记录，固定合法祖先前缀。交换只在两个兄弟终点之间进行，保留条件、转移、终点编号和节点内处置集合。两组均读取同一第18层盲表示和完整决策表示，使用相同216个训练状态、586个训练节点、原训练双视图、损失、优化器、32,777个参数及第3轮检查点。输入侧将条件软提示施加到决策查询前一个既有词元的嵌入；其幅度项和内部干预一样，在第18层决策注意力输出上计算。每个节点的四个条件和两种方法的正确标记见[逐节点结果](results/matched-nodes.csv)。[节点选择与事实模板](https://github.com/chunxi-alpc/TRIAGE/blob/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/matched_data.py) · [同一表示的训练、推断和配对汇总](https://github.com/chunxi-alpc/TRIAGE/blob/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/matched.py)

## 路径语义评分与原实验归档

全部六种方法的 BERTScore 结果覆盖官方单轮原指南7,458例。同一病例内合并相同步骤序列后有11,807条不同路径，按语言与步骤文本去重后共计算482,871个不同的文本对。表1的 DTW-PCC 和表2a均使用这次本地评测。中、英文编码器、层、重缩放参数及配对区间见[结果数据](results/single-turn.json)。表1的生成式 Acc 和表2b保留各自分母与裁判名称。[原运行的 BERTScore 实现](https://github.com/chunxi-alpc/TRIAGE/blob/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/bertscore_local.py)

[固定实验快照](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/experiments/aster_representation_20261101/release)保留普通低秩、概率上下文、盲表示分类器、匹配软提示和主方法的推理检查点，以及训练、单轮推断、交换构造和评分入口。早期43例的裁剪与键编辑、100例开发和120例验证的诊断可在[历史实验报告](https://github.com/chunxi-alpc/TRIAGE/blob/d8cf67d53d6b068d9b382114de4a5fa4b22f633b/docs/ASTER/bottleneck-results-20261031.zh.md)中查阅。运行代码保留当时服务器的目录约定，主方法在其他环境的运行方式见上文。
