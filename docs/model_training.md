# 模型与训练方法

## 数据处理

构图完成后，训练脚本首先删除样本数少于 30 的类别；当最大类别与最小类别样本量之比超过 200 时，同时删除样本量小于最大类别样本量 `1/200` 的类别。原始设备标签保存在 `y` 中，保留类别另行映射为从 0 开始的连续训练标签 `y_mapped`。

数据按原始类别分层划分。默认先抽取 20% 作为测试集，再从其余 80% 中抽取 10% 作为验证集，因此总体比例约为 72%/8%/20%。节点特征的均值与标准差仅在训练集上估计，并用于训练集、验证集和测试集的标准化。

## 图编码分支

图编码器由四层带二维边属性的 GATv2Conv 组成。各层使用多头注意力、LayerNorm 和 LeakyReLU，最后一层节点表示同时进行全局平均池化和注意力池化，二者拼接形成 110 维图表示。图表示经过两层分类器得到 `graphLogits`。

## 目的端口模式分支

模式库只使用训练集的 `dstPortSequence` 构建。端口序列先映射为离散 token，再提取长度为 3 和 4 的连续 n-gram。每个类别保留满足最小支持度的高频模式，并根据类别规模自适应确定 top-K 上限。

对每张图计算的 PatternFeature 由三部分构成：

```text
PatternFeature =
    patternScore
    + uniquePatternHitRatioByClass
    + [sharedPatternHitRatio, zeroHitFlag]
```

若保留类别数为 `C`，PatternFeature 维度为 `2C+2`。该特征经过独立分类器得到 `patternLogits`。

## Conservative Dense Gate

Gate 网络以 PatternFeature 为输入，输出两个经 softmax 归一化的权重 `wGraph` 和 `wPattern`。为限制模式分支对图分支的扰动，最终分类得分采用残差式融合：

```text
finalLogits = graphLogits + lambda * wPattern * patternLogits
```

默认 `lambda=0.2`。`wGraph` 用于记录 Gate 对两个专家的相对偏好；当前残差公式保留完整的图分支，并只用 `wPattern` 缩放模式分支。

## 优化方法

训练使用类别加权 Focal Loss：

```text
FL = -alpha_y * (1 - p_y)^gamma * log(p_y)
```

其中 `gamma=2`，类别权重与训练集中该类别样本数平方根的倒数成正比。优化器为 Adam，默认学习率为 `1e-3`，每次更新前将梯度范数裁剪到 1.0。`ReduceLROnPlateau` 根据验证集准确率降低学习率，Early Stopping 默认在连续 10 轮未提升后停止。

训练阶段不启用自适应拒绝。最佳模型仅依据验证集无拒绝准确率选择，F1 作为辅助指标记录。最佳权重确定后，才使用验证集预测置信度拟合拒绝阈值，并在测试集上同时记录识别指标、拒绝率和覆盖率。

## 防止数据泄漏

- 节点标准化参数只由训练集估计；
- 端口模式库只由训练集构建；
- 拒绝阈值只由验证集估计；
- 测试集不参与模型选择、模式库构建或阈值拟合。

## 输出产物

每次运行会创建独立结果目录，保存配置、过滤统计、训练历史、总体指标、逐类别指标、混淆矩阵和逐样本预测。checkpoint 还包含模型权重、优化器状态、类别映射、最佳轮次及测试指标。

