# 时间流图构建方法

## 图的基本定义

构图模块将 flow-level 网络流记录转换为 PyTorch Geometric 图。每条双向 flow 被拆分为正向节点和反向节点，节点分别保存对应方向的统计特征。节点按照设备视角的源 MAC 地址聚合，因此每张图描述一个设备在局部时间窗口内的通信行为。

节点特征共 28 维，包括流持续时间、字节率、包率、包长与载荷统计、到达间隔统计、应用载荷统计、方向标记、反向流存在标记、双向时延特征以及节点入度和出度。完整特征顺序由 `DevTGBTime.NODE_FEATURE_NAMES` 给出。

## 时间窗口

同一源 MAC 下的单向流按照开始时间升序排列。代码从当前最早流开始建立子图窗口，并依次纳入满足“流结束时间与窗口首流开始时间之差不超过 60 秒”的后续流；窗口完成后继续处理下一批流。节点数少于 3 的窗口不进入训练数据。

## 时间边

窗口内的流再次按照开始时间排序，只考察排序后相邻的两条流。设相邻流分别为 `flow_i` 和 `flow_(i+1)`，并满足 `start_i < start_(i+1)`，定义开始时间差：

```text
timeDiff = start_(i+1) - start_i
```

当且仅当 `0 < timeDiff <= timeThreshold` 时建立一条从较早流指向较晚流的有向边。默认 `timeThreshold=10.0` 秒。

重叠关系定义为：

```text
start_i < start_(i+1) < end_i
```

边属性为二维向量：

```text
edge_attr = [timeDiff, overlap]
overlap = 1   # 两条流重叠
overlap = -1  # 两条流不重叠
```

其中 `-1` 是“不重叠”属性标记，不表示负边。当前算法不再依据相同目的 MAC 地址额外构造负边。

## 端口序列

每张图还保存按流开始时间排列的 `srcPortSequence` 和 `dstPortSequence`。训练模块默认使用 `dstPortSequence` 构建目的端口访问模式。端口映射遵循以下优先级：

1. 无效端口映射为 `UNKNOWN_PORT`；
2. 已知应用端口保留具体端口号；
3. 其他端口按区间映射为 `WELL_KNOWN_OTHER`、`REGISTERED_1` 至 `REGISTERED_6`、`DYNAMIC_1` 至 `DYNAMIC_4`。

REGISTERED 端口分片如下：

| 端口区间 | Token |
| --- | --- |
| 1024–2047 | `REGISTERED_1` |
| 2048–4095 | `REGISTERED_2` |
| 4096–8191 | `REGISTERED_3` |
| 8192–16383 | `REGISTERED_4` |
| 16384–32767 | `REGISTERED_5` |
| 32768–49151 | `REGISTERED_6` |

## 伪代码

```text
for each flow record:
    split flow into forward node and backward node
    assign each valid node to its source-MAC group

for each source-MAC group:
    sort nodes by start time
    divide nodes into local time windows

    for each window:
        sort window nodes by start time
        for each adjacent pair (earlier, later):
            timeDiff = later.start - earlier.start
            if 0 < timeDiff <= 10 seconds:
                overlap = 1 if earlier.start < later.start < earlier.end else -1
                add directed edge earlier -> later
                set edge_attr = [timeDiff, overlap]

        append in-degree and out-degree to node features
        save graph, label, source-port sequence and destination-port sequence
```

## 缓存

构图结果保存在数据集根目录的 `processed_v4_time_overlap_edges/` 中。该目录包含 `data_<graph_id>.pt`、源/目的端口统计库和完成标记。构图规则或原始 CSV 发生变化时，应删除对应缓存后重新构图。

