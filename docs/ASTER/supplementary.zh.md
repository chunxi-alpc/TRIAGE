# 方法补充与复现

主方法在 Qwen3-8B 的第 18 层修改即将产生 Y/N 选择的查询表示。参考来自同一模型对患者、祖先条件和当前条件的另一次前向。模型只取参考的隐藏表示参与干预。测试时按实际选中的分支继续遍历，最终处置取自预测终点。公式与训练目标见[论文第 3 节](manuscript.zh.md#3-方法)。

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

`predict` 默认加载仓库内的冻结权重。患者、指南和问题进入预测，金标路径进入评分。若已有冻结实验的预测，直接运行 `score`，其 `--predictions` 也接受存放原实验分片 JSONL 的目录。该命令报告严格完整路径正确率。论文对应计数为 **6,653／7,458**。DTW-PCC 和 Acc 另需对步骤和结论作语义评分。

## 从训练病例复现干预参数

数据准备按[登记的病例编号](../../experiments/aster_representation/splits.json)从公开数据重建病例。年龄最小对按原病例和登记的年龄改动重建。以下命令复现选定层位的训练。不同轮次的权重写入新的输出目录。

```bash
python -m experiments.aster_representation prepare --dataset "$MEGA" --split train --output work/train.jsonl
python -m experiments.aster_representation prepare --dataset "$MEGA" --split age_train --output work/age_train.jsonl
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_representation train --model "$MODEL" --cases work/train.jsonl --age-cases work/age_train.jsonl --output work/trained
```

用训练所得权重执行预测时，向 `predict` 传入 `--weights work/trained/epoch3.safetensors`。开发集和内部确认集可分别用 `prepare --split development` 和 `prepare --split confirmation` 重建。
