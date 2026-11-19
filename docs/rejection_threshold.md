# 百分位自适应拒绝阈值

## 阈值拟合集合

拒绝阈值仅在验证集上拟合。模型先对验证集执行无拒绝推理，对每个样本计算 softmax 最大概率及其预测类别；随后按照“预测类别”分组置信度，而不是按照真实类别分组。该设计使测试阶段能够只依据模型输出选择相应阈值，并避免使用测试集标签。

## Pq 的计算

设预测为类别 `c` 的验证样本最大 softmax 概率集合为：

```text
S_c = {s_1, s_2, ..., s_n}
```

将其升序排列为 `s_(0) <= ... <= s_(n-1)`。对于百分位 `q`，代码使用 NumPy 的线性插值法计算：

```text
h = (n - 1) * q / 100
j = floor(h)
gamma = h - j
Q_q = (1 - gamma) * s_(j) + gamma * s_(j+1)
```

当 `j=n-1` 时取 `Q_q=s_(n-1)`。类别 `c` 的最终阈值还受到上下界约束：

```text
T_c = min(max(Q_q, lowerBound), upperBound)
```

若验证集中没有样本被预测为类别 `c`，则使用全局回退阈值，并同样施加上下界约束。

当前发布代码的默认参数为 `q=5`、`lowerBound=0.30`、`upperBound=0.65`。若要实现严格的 `P10Floor50`，可设置 `rejectPercentile=10`、`minRejectThreshold=0.50` 和 `maxRejectThreshold=1.0`；`P5Floor50` 则将 `rejectPercentile` 设为 5。

## 测试阶段的拒绝判定

测试样本的预测类别为 `c`、最大 softmax 概率为 `s` 时：

```text
s < T_c   -> 拒绝，预测标签记为 -1
s >= T_c  -> 接受，输出类别 c
```

当前实现还提供 `maxRejectRatePerClass` 保护。若某个预测类别的候选拒绝比例超过该上限，只拒绝其中置信度最低的样本；默认每个预测类别最多拒绝 90%。若需要完全按照百分位阈值执行，可将该参数设为 1.0。

## 指标解释

拒绝后的 Accuracy、Precision、Recall 和 F1 只在被接受样本上计算；拒绝样本不参与这些识别指标。代码同时报告拒绝率和覆盖率，其中：

```text
rejectionRate = rejectedSampleCount / totalSampleCount
coverage = 1 - rejectionRate
```

因此，评价拒绝策略时应同时报告识别性能和覆盖率，不能只比较拒绝后的准确率。

