# GNN-IoT：基于时间流图与端口访问模式的设备识别

本项目将网络流记录转换为面向 IoT 设备识别的时间流图，并使用 GATv2 图编码器、目的端口访问模式和 Conservative Dense Gate 完成图级分类。公开目录只包含最终方法的构图、模式挖掘、训练与评估代码，不包含消融实验、私有数据、训练日志、模型权重或实验结果。

## 目录结构

```text
.
├── DevTGBTime.py
├── portPatternMining.py
├── DevGNNClassifierConservativeDenseGate.py
├── requirements.txt
├── docs/
│   ├── graph_construction.md
│   ├── model_training.md
│   └── rejection_threshold.md
└── utils/
    └── visualGraph_v1.py
```

主要文件的职责如下：

| 文件 | 功能 |
| --- | --- |
| `DevTGBTime.py` | 从 flow-level CSV 构建时间流图，并缓存 PyG `Data` 对象 |
| `portPatternMining.py` | 端口映射、n-gram 模式库构建及 PatternFeature 计算 |
| `DevGNNClassifierConservativeDenseGate.py` | 数据划分、模型训练、验证集阈值拟合、测试评估及产物保存 |
| `utils/visualGraph_v1.py` | 可选的图结构可视化 |

详细方法见[构图方法](docs/graph_construction.md)、[模型训练方法](docs/model_training.md)和[拒绝阈值方法](docs/rejection_threshold.md)。

## 运行环境

代码已在以下环境中检查：

- Python 3.13.9
- PyTorch 2.9.0+cu128
- PyTorch Geometric 2.7.0
- CUDA 12.8 运行时

建议创建独立环境后安装依赖：

```bash
conda create -n gnnTorch_env python=3.13 -y
conda activate gnnTorch_env
pip install -r requirements.txt
```

`torch-scatter` 应与本机 PyTorch 和 CUDA 版本匹配。如果普通安装无法获得兼容的二进制包，请按照 PyTorch Geometric 官方安装说明选择对应 wheel。

## 数据准备

每个数据集使用一个独立目录，CSV 文件直接放在该目录下。例如：

```text
/path/to/csvDataset/
└── Benign/
    ├── part_001.csv
    └── part_002.csv
```

CSV 必须包含 `DevTGBTime.REQUIRED_CSV_COLUMNS` 中定义的字段。数据集和生成的图缓存不纳入版本控制。

## 运行训练

训练脚本默认自动使用 CUDA；没有可用 GPU 时回退到 CPU。通过环境变量指定数据集路径和批大小：

```bash
export GNN_DATASET_PATH=/absolute/path/to/csvDataset/Benign
export GNN_BATCH_SIZE=8
python DevGNNClassifierConservativeDenseGate.py
```

可选环境变量：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `GNN_DATASET_PATH` | 数据集根目录 | `csvDataset/Web-Based` |
| `GNN_BATCH_SIZE` | 训练批大小 | `8` |
| `GNN_CSV_WORKERS` | CSV 构图进程数 | `1` |
| `GNN_LOG_DIR` | 日志目录 | `logs` |
| `GNN_STDOUT_ONLY` | 设为 `1` 时只输出到终端 | 未启用 |

首次运行会在数据集目录下生成 `processed_v4_time_overlap_edges/` 图缓存。训练结果写入 `results/`，模型权重写入 `checkpoints/`，日志写入 `logs/`。

一次完整运行会保存：

- 运行配置与类别过滤统计；
- 逐轮训练历史；
- 测试集总体指标和逐类别指标；
- 混淆矩阵与逐样本预测；
- 最佳验证轮次的模型 checkpoint。

## 复现约定

数据按类别分层划分为训练集、验证集和测试集，随机种子默认为 42。节点标准化统计量和端口模式库仅由训练集估计；自适应拒绝阈值仅由验证集估计；测试集仅用于最终评估。修改构图语义后应删除旧的 `processed_v4_time_overlap_edges/`，重新生成图缓存。

