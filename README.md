# TRIAGE · ASTER

**离线学习编辑方向，在线判断适用性，再执行规则键编辑。**

ASTER（Action-blind Selection, Then Eigen-subspace Re-routing）研究不适用兄弟分支上的处置是否会干扰指南分支判断。当前主稿按三个独立职责组织，模块一和模块二分别向模块三提供输入，不串联训练。

| 模块 | 作用 | 输出 |
|---|---|---|
| 一：对比适用性子空间学习 | 用独立金标最小对离线学习 | 有效键值头与编辑方向 |
| 二：处置盲适用性评估 | 同一冻结骨干只读患者与条件 | 分支概率、预测与弃权 |
| 三：适用性门控的规则键编辑 | 按病例分配增益并修改独立缓存 | 正常生成的答案或路径 |

**研究状态：方法与实验方案。** 参考实现覆盖模块二的分类、温度拟合和模块三的规则增益接口；模块一方向学习、实际键编辑及完整流水线仍待实现。没有新增模型性能结果，历史实验不等于新版验证。

## 阅读入口

[**论文主稿**](docs/ASTER/manuscript.zh.md)是唯一方法正文；[方法补充](docs/ASTER/supplementary.zh.md)保存推导与实现约定；[实验协议](docs/ASTER/experiments.zh.md)保存数据、基线、指标与冻结规则；[引用清单](docs/ASTER/references.zh.md)说明来源和支持范围。

可运行接口及实现状态统一见 [src/aster](src/aster/README.md)。旧脚本 `run_aster_module1` 保留兼容名称，现对应论文模块二；类名与输出格式未改。

```bash
python scripts/check_aster_markdown_math.py
python -m unittest discover -s tests -p 'test_aster*.py' -v
```

上述源码检查不替代模型验证。历史方案、旧实验、审计和重复模板统一从[归档入口](docs/ASTER/archive/README.md)访问，不再并列为当前主线。
