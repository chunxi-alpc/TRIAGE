# SEKA BiasBios 协议核对

核对日期2026-10-26；官方源码固定`waylonli/SEKA@aab9cfed80dc012fb9e83f8bdefb8675c33f40e7`。本次仅查源码和论文，未运行本地实验，未读取临床test。协调器已报告：官方`BIOS.pkl`按作者流程取原始前50000条，转换后19804条；重复source name ID记录保留，不按名字去重。准确文件哈希与转换日志由协调器归档。

## 生成长度：官方README明确为64

- [根README的BiasBios命令，第108–116行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/README.md#L108)显式传`--max_new_tokens 64`。
- [benchmarks/README的BiasBios段落](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/README.md#biasbios)也显式传64；该README将所列命令说明为论文设置。该示例模型为Qwen3-4B-Base；并没有另列8B专用长度。因此可写“沿用官方BiasBios示例的64新token预算”，不能写“已独立核实论文8B运行日志采用64”。
- [驱动参数第186–188行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/eval_bias_gen.py#L186)的`max_length`和`max_new_tokens`均默认None；[评价器第149–150行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L149)在两者都None时改成`max_length=150`。这里是**输入加输出总长度150**，不是150新token，也不是64默认值。
- [论文附录F](https://arxiv.org/html/2603.01281v1#A6)明确采用贪心解码和4500:5000验证、5000:10000评价。没有找到BiasBios单独的生成长度声明。论文另一任务lost-in-the-middle的60-token上限不能套用到BiasBios。

因此本轮显式固定64有官方发布说明支持；Qwen3-8B替代Base检查点等其他适配仍须独立披露。64为上限，遇EOS可以提前结束；所有基线/干预沿用相同上限。

## 原始指标

[论文附录E.2](https://arxiv.org/html/2603.01281v1#A5.SS2)给出的主指标是28类职业中的top-1准确率，大小写不敏感。[发布评价器](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L161)的实际计算如下：

1. 当前评价slice的`target_mediated`转小写并去重排序，形成候选职业集合。因此正式5000:10000要登记实际候选数；小试slice不足28类时不与论文28类准确率混用。
2. 每个职业构造小写、首字母大写、全大写三种形式，分别取**首token**；在首个生成步的log-softmax概率中取三种形式的最大值作为职业分数。Qwen路径不额外加前导空格。[第161–178行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L161)，[第324–341行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L324)。
3. 分数最高的职业与目标比较形成top1_accuracy；代码还报告top3_accuracy、生成文本的n-gram熵和TF-IDF一致性。这不是将整段生成文本与职业字符串作精确匹配。[第365–388行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L365)。

top1只依赖首步职业概率；后续64-token生成影响运行成本、fluency和consistency，不能把仅生成1token的加速结果冒称完整64-token生成协议。TF-IDF向量器拟合完整处理文件的`source.bio`，参考职业原文来自当前评价slice，仅用于生成后的consistency计算；这是官方辅助指标的已知取数方式。[第111–183行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L111)。驱动实际保存`result.json`，包括逐例结果和metrics；README泛称`metric_result.json`不能当真实文件名。

## 原始预处理与输入

[转换器主入口第194–197行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/reformat_dataset.py#L194)传`bio_min_words=10, sent_min_words=3, sents_choice="all", attr_sent_idx=0`。函数签名的`limit=50000`仍生效，按原始顺序取前50000条；不能把函数签名默认的` sents_choice=1`误当CLI实际行为。

使用spaCy `en_core_web_sm`分句，保留至少3词的句子并按原顺序拼回全部保留句；姓名变体规范为first name。丢弃黑名单名字、规范后名字出现多于一次、总词数少于10的记录；名字未出现时加简短姓名引导。职业中的下划线改为空格。`attribute`为第一保留句中姓名之后的部分（姓名不在该句则用该句）；保留完整处理后context作为输入，不只取属性句。[第68–179行](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/reformat_dataset.py#L68)。

作者转换器写JSON时漏换行；协调器只修复JSONL分隔，不改变转换筛选。模型输入为context加两个换行和职业补全提示；默认不启用`--attribute-no-entity`缩短上下文。[preprocess.py](https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/preprocess.py)。默认评价slice为处理后`train[5000:10000]`，即零基行5000至9999，共5000条；不对重名ID去重或重新洗牌。

## 逐例导出标识

转换器第140行直接将完整姓名片段用下划线连接成`id`，**它不是匿名ID，也不是唯一行键**。PR导出应生成稳定行标识，例如：

```python
sample_id = sha256(json.dumps([
    "SEKA-BiasBios-v1", processed_file_sha256,
    "5000:10000", absolute_processed_row_index, original_source_name_id
], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
```

其中`absolute_processed_row_index=5000+slice_row_index`。原始姓名ID及其映射只在服务器保存；PR不导出prompt、source或原始传记全文。跨方法/seed使用同一个哈希规则和处理文件SHA。重复名字仍占各自的行和分母，不能按原ID合并结果。用于统计的cluster_id与唯一sample_id分开登记；名字重复不自动证明同一患者，也不能以姓名哈希宣称真实患者映射已经核实。

源码blob指纹：根README`46bd09cb227b302ad6247a97acebcd0081ae7d13`；benchmarks README`71d95634be46c63629f4a34971d91773e4fabd34`；驱动`ad9d207d2c5493e551058de39fef74e7f73e3df8`；evaluate`450bb2acc1ec2b82bfff9729abd41e97944b5043`；reformat`fd710150a15d14343d6c233a17cb54065751b9c1`；preprocess`beef2cfdeb1e3b38669870e348cd84aae4bc1682`。
