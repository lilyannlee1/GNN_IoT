"""
IoT-DFIG 图分类实验主脚本。

当前默认用途：
    运行 GNN + dstPort patternFeature conservativeDenseGate 图分类实验。
    每张图先由 GNN pooling 得到 graphEmbedding，
    再与训练集 patternLibrary 计算得到的 patternFeature 分别输出 logits 后残差相加。
    finalLogits = graphLogits + self.patternLambda * wPattern * patternLogits

默认融合数据流：
1. 使用 DevTGBTime.TrafficGraphBuild 加载或构建时间重叠 PyG Data 图；
2. 每张图应包含端口序列字段，例如 srcPortSequence 或 dstPortSequence；
3. 过滤极度不平衡类别，并生成连续训练标签 y_mapped；
4. 分层划分 train / val / test；
5. 只用 trainDataset 构建 patternLibrary，避免数据泄漏；
6. train / val / test 都只匹配训练集 patternLibrary 并写入 patternFeature；
7. GNN 使用 graphLogits + patternLambda * wPattern * patternLogits 输出 final logits。

重要配置：
    trainConfig["patternLambda"] 控制 patternLogits 的残差权重。
    trainConfig["portSequenceField"] 控制使用哪个端口序列字段。
    例如 "dstPortSequence" 表示使用目的端口序列。
"""

import copy
import csv
import datetime
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np  # 导入numpy用于矩阵操作
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as confusionMatrix
from sklearn.metrics import precision_recall_fscore_support as precisionRecallFscoreSupport
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentionalAggregation, GATv2Conv, global_mean_pool as globalMeanPool

from DevTGBTime import (
    APP_PORT_LABEL as appPortLabel,
    GRAPH_CACHE_DIR_NAME as graphCacheDirName,
    TrafficGraphBuild,
)
from portPatternMining import (
    buildClassPatternLibrary,
    computePatternFeature,
)
from utils.visualGraph_v1 import visualize_graph as visualizeGraph  # 导入可视化函数


# =========================
# 1. 可选池化工具
# =========================
# 当前主模型没有使用 globalMinPool，保留它是为了后续实验可切换池化策略。

def globalMinPool(x, batch):
    """按图对节点特征做最小值池化，输入和 PyG 的 batch 向量配合使用。"""
    from torch_scatter import scatter_min as scatterMin

    out, _ = scatterMin(x, batch, dim=0)
    return out


# =========================
# 2. 日志配置
# =========================
currentTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
logFilename = None


def setupLogging() -> str:
    """Initialize logging, optionally using stdout only for a unified outer log."""
    global logFilename
    if logFilename is not None:
        return logFilename

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stdoutOnly = os.getenv("GNN_STDOUT_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if stdoutOnly:
        logFilename = "<stdout-only>"
    else:
        logDir = os.getenv("GNN_LOG_DIR", "logs")
        os.makedirs(logDir, exist_ok=True)
        logFilename = os.path.join(logDir, f"gnn_training_{currentTime}.log")
        logPath = os.path.abspath(logFilename)
        hasFileHandler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == logPath
            for handler in rootLogger.handlers
        )
        if not hasFileHandler:
            fileHandler = logging.FileHandler(logFilename, encoding='utf-8')
            fileHandler.setFormatter(formatter)
            rootLogger.addHandler(fileHandler)

    hasStreamHandler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in rootLogger.handlers
    )
    if not hasStreamHandler:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        rootLogger.addHandler(streamHandler)

    return logFilename

logger = logging.getLogger(__name__)


# =========================
# 3. 训练配置
# =========================
# 这里集中管理本次实验参数。类别数不在这里手写，而是从过滤后的数据集自动得到。
trainConfig = {
    "datasetPath": "csvDataset/Web-Based",
    "dataSource": "builder",  # builder: 从原始数据构图；processed: 读取 processed/*.pt
    "batchSize": 8,
    "epochs": 100,
    "lr": 0.001,
    "weightDecay": 0.0,
    "dropout": 0.3,
    "standardizeNodeFeatures": True,
    "nodeFeatureStdEpsilon": 1e-6,
    "patience": 10,
    "schedulerPatience": 5,
    "schedulerFactor": 0.1,
    "testRatio": 0.2,
    "valRatioInTrainVal": 0.1,
    "seed": 42,
    "rejectMode": "adaptive",
    "rejectScoreSource": "softmaxProbability",
    "rejectionThreshold": 0.5,
    "useAdaptiveRejectThreshold": True,
    "rejectPercentile": 5,
    "minRejectThreshold": 0.3,
    "maxRejectThreshold": 0.65,
    "maxRejectRatePerClass": 0.9,
    "imbalanceRatioThreshold": 200.0,
    "minCountThreshold": 30,
    "numWorkers": None,
    "pinMemory": True,
    "visualizeFirstGraph": False,
    "resultsDir": "results",
    "checkpointDir": "checkpoints",
    "experimentName": "conservativeDenseGate",
    "experimentGroup": "finalMethod",
    "fusionMode": "conservativeDenseGate",
    "patternLambda": 0.2,
    "ngramLengths": [3, 4],
    "topK": 50,
    "maxTopK": 100,
    "topKPower": 0.25,
    "minSupport": 2,
    "scoreMode": "tfIdfHitRatio",
    "portSequenceField": "dstPortSequence",
}


def resolveBatchSize(defaultBatchSize: int, environmentValue: str = None) -> int:
    """Resolve an optional positive GNN_BATCH_SIZE override."""
    if environmentValue is None:
        return int(defaultBatchSize)
    try:
        batchSize = int(environmentValue)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GNN_BATCH_SIZE must be a positive integer, got {environmentValue!r}."
        ) from exc
    if batchSize <= 0:
        raise ValueError(
            f"GNN_BATCH_SIZE must be a positive integer, got {environmentValue!r}."
        )
    return batchSize


# =========================
# 4. 通用工具函数
# =========================

def setRandomSeed(seed: int) -> None:
    """固定随机种子，减少不同训练运行之间的随机波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def makeRunDir(baseDir: str, prefix: str = "gnn_iot") -> str:
    """为一次训练创建独立结果目录，避免覆盖历史实验结果。"""
    runDir = os.path.join(baseDir, f"{prefix}_{currentTime}")
    os.makedirs(runDir, exist_ok=True)
    return runDir


def toJsonable(value: Any) -> Any:
    """把 Counter、numpy、Tensor 等对象转换成 json.dump 可保存的普通 Python 类型。"""
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): toJsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [toJsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def saveJson(path: str, data: Dict[str, Any]) -> None:
    """以 UTF-8 JSON 格式保存配置、指标或统计信息。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(toJsonable(data), f, ensure_ascii=False, indent=2)


def getLabelValue(y) -> int:
    """统一读取图标签，兼容 data.y 是 int 或单元素 Tensor 的情况。"""
    if isinstance(y, torch.Tensor):
        return int(y.item())
    return int(y)


def getRejectMode(rejectMode=None, useAdaptiveRejectThreshold: bool = False) -> str:
    """Resolve rejectMode while keeping compatibility with the old adaptive boolean."""
    if rejectMode is None:
        return "adaptive" if bool(useAdaptiveRejectThreshold) else "global"

    resolvedRejectMode = str(rejectMode).strip().lower()
    validRejectModes = {"none", "global", "adaptive"}
    if resolvedRejectMode not in validRejectModes:
        raise ValueError(
            f"Unsupported rejectMode={rejectMode}; expected one of {sorted(validRejectModes)}."
        )
    return resolvedRejectMode


# =========================
# 5. 损失函数
# =========================

class FocalLoss(nn.Module):
    """
    Focal Loss 用于缓解类别不平衡。

    输入:
        logits: 模型输出，形状 [batch_size, num_classes]
        targets: 训练标签，形状 [batch_size]
    输出:
        一个标量 loss。

    gamma 控制对难分类样本的关注程度；alpha 用于给不同类别加权。
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.detach().float())
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = float(alpha)

    def forward(self, logits, targets):
        """计算 focal loss；targets 必须是 0 到 num_classes-1 的连续标签。"""
        logProbs = F.log_softmax(logits, dim=1)
        probs = torch.exp(logProbs)
        targetsOneHot = F.one_hot(targets, num_classes=logits.size(1)).float()

        focalWeight = (1 - probs) ** self.gamma
        loss = -targetsOneHot * focalWeight * logProbs

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            else:
                alpha = torch.full(
                    (logits.size(1),),
                    float(self.alpha),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            loss = loss * alpha

        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# =========================
# 6. GNN 模型定义
# =========================

class TrafficGNN(nn.Module):
    """
    基于 GATv2 的 IoT-DFIG 图分类模型。

    输入:
        x: 节点特征，形状 [num_nodes, inputDim]
        edgeIndex: 边索引，形状 [2, num_edges]
        edgeAttr: 边特征，统一为 [num_edges, 2]
        batch: PyG batch 向量，标明每个节点属于哪张图

    输出:
        logits: 图级分类得分，形状 [batch_size, numClasses]

    模型主干:
        4 层 GATv2Conv -> mean pooling + attention pooling -> graphClassifier。
        patternFeature 单独进入 patternClassifier，最终做 residual logit fusion。
    """
    def __init__(
        self,
        inputDim: int,
        numClasses: int = 7,
        dropout: float = 0.3,
        usePatternFeature: bool = True,
        patternFeatureDim: int = None,
        patternLambda: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.numClasses = int(numClasses)
        self.graphEmbeddingDim = 110
        self.usePatternFeature = True
        self.patternFeatureDim = (
            int(patternFeatureDim)
            if patternFeatureDim is not None
            else self.numClasses * 2 + 2
        )
        self.patternLambda = float(patternLambda)
        self.lastExpertWeights = None
        self.lastWGraph = None
        self.lastWPattern = None
        self.lastPatternContribution = None
        logger.info(
            "Initialize TrafficGNN: fusionMode=conservativeDenseGate, graphEmbeddingDim=%s, patternFeatureDim=%s, patternLambda=%.6f",
            self.graphEmbeddingDim,
            self.patternFeatureDim,
            self.patternLambda,
        )

        # 统一使用 LeakyReLU，避免 ReLU 在负值区间完全截断特征。
        self.act = nn.LeakyReLU(0.1)

        # 每层 GAT 输出维度不同，因此分别配置 LayerNorm。
        self.norm1 = nn.LayerNorm(22 * 8)
        self.norm2 = nn.LayerNorm(11 * 11)
        self.norm3 = nn.LayerNorm(11 * 8)
        self.norm4 = nn.LayerNorm(11 * 5)

        # GATv2 使用边特征 edge_attr，因此 edge_dim 固定为 2。
        self.gatConv1 = GATv2Conv(in_channels=inputDim, out_channels=22, heads=8, edge_dim=2)
        self.gatConv2 = GATv2Conv(in_channels=22 * 8, out_channels=11, heads=11, edge_dim=2)
        self.gatConv3 = GATv2Conv(in_channels=11 * 11, out_channels=11, heads=8, edge_dim=2)
        self.gatConv4 = GATv2Conv(in_channels=11 * 8, out_channels=11, heads=5, edge_dim=2)

        # 注意力池化让模型自动学习哪些 flow 节点对整张图更重要。
        self.attPool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(55, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1)
            )
        )
        self.dropPathRate = 0.1

        # 两个独立 logits 分支：graphClassifier 只看图表示，patternClassifier 只看 patternFeature。
        self.graphClassifier = nn.Sequential(
            nn.Linear(self.graphEmbeddingDim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, self.numClasses),
        )
        self.patternClassifier = nn.Sequential(
            nn.Linear(self.patternFeatureDim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, self.numClasses),
        )
        self.gateMlp = nn.Sequential(
            nn.Linear(self.patternFeatureDim, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )


    @staticmethod
    def _prepareEdgeAttr(edgeIndex, edgeAttr, device, dtype=torch.float32):
        """把不同来源的边特征统一成 GATv2Conv 需要的 [num_edges, 2]。"""
        numEdges = edgeIndex.size(1)
        if edgeAttr is None:
            return torch.zeros((numEdges, 2), device=device, dtype=dtype)
        edgeAttr = edgeAttr.to(device=device, dtype=dtype)
        if edgeAttr.dim() == 1:
            if edgeAttr.numel() == 0:
                return torch.empty((0, 2), device=device, dtype=dtype)
            if edgeAttr.numel() == numEdges * 2:
                edgeAttr = edgeAttr.view(-1, 2)
            else:
                edgeAttr = edgeAttr.unsqueeze(-1)
        if edgeAttr.size(1) == 1:
            pad = torch.zeros(edgeAttr.size(0), 1, device=edgeAttr.device, dtype=edgeAttr.dtype)
            edgeAttr = torch.cat([edgeAttr, pad], dim=1)
        return edgeAttr[:, :2]

    @staticmethod
    def _preparePatternFeature(patternFeature, numGraphs, patternFeatureDim, device, dtype=torch.float32):
        """Validate patternFeature as [batchSize, patternFeatureDim]."""
        if patternFeature is None:
            raise ValueError("batch 缺少 patternFeature 字段，无法执行 conservativeDenseGate。")
        patternFeature = patternFeature.to(device=device, dtype=dtype)
        if patternFeature.dim() != 2:
            raise ValueError(
                f"patternFeature shape mismatch: got {tuple(patternFeature.shape)}, "
                f"expected [{numGraphs}, {patternFeatureDim}]"
            )
        if patternFeature.size(0) != numGraphs or patternFeature.size(1) != patternFeatureDim:
            raise ValueError(
                f"patternFeature shape mismatch: got {tuple(patternFeature.shape)}, "
                f"expected [{numGraphs}, {patternFeatureDim}]"
            )
        return patternFeature

    def getGraphEmbedding(self, x, edgeIndex, edgeAttr, batch):
        """提取 mean pooling + attention pooling 得到的图级表示。"""
        edgeAttr = self._prepareEdgeAttr(edgeIndex, edgeAttr, x.device, x.dtype)
        
        # 4 层 GATv2 逐步建模 flow 节点之间的时间依赖关系。
        h1 = self.gatConv1(x, edgeIndex, edgeAttr)
        h1 = self.act(self.norm1(h1))

        h2 = self.gatConv2(h1, edgeIndex, edgeAttr)
        h2 = self.act(self.norm2(h2)) 

        h3 = self.gatConv3(h2, edgeIndex, edgeAttr)
        h3 = self.act(self.norm3(h3))

        h4 = self.gatConv4(h3, edgeIndex, edgeAttr)
        h4 = self.act(self.norm4(h4)) 

        # 节点级表示 -> 图级表示：平均池化看整体行为，注意力池化突出关键 flow。
        gMean = globalMeanPool(h4, batch)
        gAtt = self.attPool(h4, batch)
        graphEmbedding = torch.cat([gMean, gAtt], dim=1)

        return graphEmbedding

    def computeExpertWeights(self, patternFeature):
        """Compute conservative dense gate expert weights from patternFeature."""
        return F.softmax(self.gateMlp(patternFeature), dim=1)

    def forward(self, x, edgeIndex, edgeAttr, batch, patternFeature=None):
        """返回 conservativeDenseGate 后的 finalLogits。"""
        logger.debug(f"GNN前向传播 - 输入节点数: {x.shape[0]}")

        graphEmbedding = self.getGraphEmbedding(x, edgeIndex, edgeAttr, batch)
        graphLogits = self.graphClassifier(graphEmbedding)
        patternFeature = self._preparePatternFeature(
            patternFeature,
            numGraphs=graphEmbedding.size(0),
            patternFeatureDim=self.patternFeatureDim,
            device=graphEmbedding.device,
            dtype=graphEmbedding.dtype,
        )
        patternLogits = self.patternClassifier(patternFeature)
        expertWeights = self.computeExpertWeights(patternFeature)
        wGraph = expertWeights[:, 0].unsqueeze(1)
        wPattern = expertWeights[:, 1].unsqueeze(1)
        patternContribution = self.patternLambda * wPattern * patternLogits
        finalLogits = graphLogits + patternContribution

        self.lastExpertWeights = expertWeights.detach()
        self.lastWGraph = wGraph.detach()
        self.lastWPattern = wPattern.detach()
        self.lastPatternContribution = patternContribution.detach()

        return finalLogits



# =========================
# 7. 训练与评估
# =========================

def trainOneEpoch(gnn, trainLoader, gnnOptimizer, lossFn, device):
    """
    训练一个 epoch。

    输入:
        trainLoader: 训练集 DataLoader，每个 batch 是多个图的 PyG Batch。
        lossFn: FocalLoss，使用 batch.y_mapped 计算损失。
    输出:
        当前 epoch 的平均训练 loss。
    """
    gnn.train()
    trainLoss = 0
    totalGradNorm = 0
    numBatches = len(trainLoader)
    wGraphValueList = []
    wPatternValueList = []
    patternContributionNormList = []
    if numBatches == 0:
        raise ValueError("训练集 DataLoader 为空，无法训练。")

    for batch in trainLoader:
        batch = batch.to(device)
        gnnOptimizer.zero_grad(set_to_none=True)

        # 前向传播输出图级分类 logits，形状为 [batch_size, num_classes]。
        if not hasattr(batch, "patternFeature"):
            raise ValueError("batch 缺少 patternFeature 字段，无法执行 conservativeDenseGate 训练。")
        patternFeature = batch.patternFeature
        finalLogits = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch, patternFeature)
        if getattr(gnn, "lastWGraph", None) is not None:
            wGraphValueList.append(gnn.lastWGraph.detach().float().view(-1).cpu())
        if getattr(gnn, "lastWPattern", None) is not None:
            wPatternValueList.append(gnn.lastWPattern.detach().float().view(-1).cpu())
        if getattr(gnn, "lastPatternContribution", None) is not None:
            patternContributionNormList.append(
                gnn.lastPatternContribution.detach().float().norm(dim=1).cpu()
            )

        # 训练时使用连续标签 y_mapped；原始标签 y 保留给评估和日志展示。
        if not hasattr(batch, "y_mapped"):
            raise ValueError("batch 缺少 y_mapped 字段，请先执行类别过滤与标签映射步骤。")
        loss = lossFn(finalLogits, batch.y_mapped.view(-1))
        loss.backward()

        # 裁剪梯度，减少梯度爆炸风险；同时记录梯度范数观察训练稳定性。
        batchGradNorm = torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0).item()
        totalGradNorm += batchGradNorm

        gnnOptimizer.step()
        trainLoss += loss.item()

    avgGradNorm = totalGradNorm / numBatches
    logger.info(f"🔍 本轮训练平均梯度范数 (grad_norm): {avgGradNorm:.6f}")

    if wGraphValueList and wPatternValueList:
        allWGraphValues = torch.cat(wGraphValueList)
        allWPatternValues = torch.cat(wPatternValueList)
        logger.info(
            "[ConservativeDenseGate][train] wGraphMean=%.6f | wGraphStd=%.6f | wGraphMin=%.6f | wGraphMax=%.6f | wPatternMean=%.6f | wPatternStd=%.6f | wPatternMin=%.6f | wPatternMax=%.6f",
            float(allWGraphValues.mean().item()),
            float(allWGraphValues.std(unbiased=False).item()),
            float(allWGraphValues.min().item()),
            float(allWGraphValues.max().item()),
            float(allWPatternValues.mean().item()),
            float(allWPatternValues.std(unbiased=False).item()),
            float(allWPatternValues.min().item()),
            float(allWPatternValues.max().item()),
        )
    if patternContributionNormList:
        allPatternContributionNorms = torch.cat(patternContributionNormList)
        logger.info(
            "[ConservativeDenseGate][train] patternContributionNormMean=%.6f | patternContributionNormMax=%.6f",
            float(allPatternContributionNorms.mean().item()),
            float(allPatternContributionNorms.max().item()),
        )
    
    return trainLoss / numBatches


def evaluateModel(
    model,
    loader,
    device,
    keptClassLabels,
    rejectionThreshold=0.0,
    useAdaptiveRejectThreshold: bool = False,
    rejectThresholdByClass=None,
    rejectPercentile=None,
    minRejectThreshold=None,
    maxRejectThreshold=None,
    maxRejectRatePerClass=1.0,
    rejectMode=None,
    splitName=None,
):
    """
    在训练集、验证集或测试集上评估模型。

    评估时会把模型输出的连续类别索引映射回原始设备类别标签。
    global 和 adaptive 模式都使用预测类别的 softmaxProbability 判断拒绝。

    输出:
        metrics 字典，包含不含拒绝类的 accuracy、micro/macro/weighted F1、
        逐类别指标、混淆矩阵、拒绝率和逐样本预测明细。
    """
    model.eval()
    allPreds, allLabels = [], []
    predictionRows = []
    sampleRows = []
    predClassSampleIndices = defaultdict(list)
    rawRejectedSampleIndicesByPredClass = defaultdict(list)
    wGraphValueList = []
    wPatternValueList = []
    wPatternByClass = defaultdict(list)
    indexToLabel = {idx: label for idx, label in enumerate(keptClassLabels)}
    numClasses = len(keptClassLabels)
    rejectionClass = -1  # 使用 -1 表示拒绝类，避免和真实类别编号冲突。
    rejectMode = getRejectMode(rejectMode, useAdaptiveRejectThreshold)
    useAdaptiveRejectThreshold = rejectMode == "adaptive"
    rejectScoreSource = "softmaxProbability"
    if maxRejectRatePerClass is None:
        maxRejectRatePerClass = 1.0
    maxRejectRatePerClass = float(maxRejectRatePerClass)
    if maxRejectRatePerClass < 0.0 or maxRejectRatePerClass > 1.0:
        raise ValueError(
            f"maxRejectRatePerClass={maxRejectRatePerClass} must be between 0.0 and 1.0."
        )
    if rejectMode == "adaptive":
        if rejectThresholdByClass is None:
            raise ValueError("rejectThresholdByClass is required when rejectMode='adaptive'.")
        if len(rejectThresholdByClass) != numClasses:
            raise ValueError(
                f"rejectThresholdByClass length mismatch: got {len(rejectThresholdByClass)}, expected {numClasses}."
            )
        rejectThresholdByClass = [float(value) for value in rejectThresholdByClass]
    else:
        rejectThresholdByClass = None

    with torch.no_grad():
        for batchIdx, batch in enumerate(loader):
            batch = batch.to(device)
            if not hasattr(batch, "patternFeature"):
                raise ValueError("batch 缺少 patternFeature 字段，无法执行 conservativeDenseGate 评估。")
            patternFeature = batch.patternFeature
            finalLogits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, patternFeature)
            probs = F.softmax(finalLogits, dim=1)
            maxProbs, predIdx = probs.max(dim=1)
            rejectScores = maxProbs

            predIdx = predIdx.cpu()
            maxProbs = maxProbs.cpu()
            rejectScores = rejectScores.cpu()
            labelsCpu = batch.y.view(-1).cpu()
            if getattr(model, "lastWGraph", None) is not None:
                wGraphCpu = model.lastWGraph.detach().float().view(-1).cpu()
                wGraphValueList.append(wGraphCpu)
            if getattr(model, "lastWPattern", None) is not None:
                wPatternCpu = model.lastWPattern.detach().float().view(-1).cpu()
                wPatternValueList.append(wPatternCpu)
                for wPatternValue, trueLabelValue in zip(wPatternCpu.tolist(), labelsCpu.tolist()):
                    wPatternByClass[int(trueLabelValue)].append(float(wPatternValue))

            for sampleIdx, (pIdx, pProb, rejectScore, trueLabel) in enumerate(
                zip(predIdx.tolist(), maxProbs.tolist(), rejectScores.tolist(), labelsCpu.tolist())
            ):
                pIdx = int(pIdx)
                trueLabel = int(trueLabel)
                if rejectMode == "adaptive":
                    sampleRejectThreshold = float(rejectThresholdByClass[pIdx])
                    rawRejected = float(rejectScore) < sampleRejectThreshold
                elif rejectMode == "global":
                    sampleRejectThreshold = float(rejectionThreshold)
                    rawRejected = float(rejectScore) < sampleRejectThreshold
                else:
                    sampleRejectThreshold = 0.0
                    rawRejected = False

                rowIndex = len(sampleRows)
                sampleRows.append({
                    "batch_idx": batchIdx,
                    "sample_idx": sampleIdx,
                    "true_label": trueLabel,
                    "pred_index": pIdx,
                    "confidence": float(pProb),
                    "reject_score": float(rejectScore),
                    "reject_threshold": float(sampleRejectThreshold),
                    "raw_rejected": bool(rawRejected),
                })
                predClassSampleIndices[pIdx].append(rowIndex)
                if rawRejected:
                    rawRejectedSampleIndicesByPredClass[pIdx].append(rowIndex)

    finalRejectedIndices = set()
    if rejectMode == "adaptive":
        for predClassIndex, rawRejectedIndices in rawRejectedSampleIndicesByPredClass.items():
            classTotalCount = len(predClassSampleIndices[predClassIndex])
            maxRejectedCount = int(np.floor(classTotalCount * maxRejectRatePerClass))
            if len(rawRejectedIndices) > maxRejectedCount:
                sortedRejectedIndices = sorted(
                    rawRejectedIndices,
                    key=lambda rowIndex: sampleRows[rowIndex]["confidence"],
                )
                finalRejectedIndices.update(sortedRejectedIndices[:maxRejectedCount])
            else:
                finalRejectedIndices.update(rawRejectedIndices)
    elif rejectMode == "global":
        for rawRejectedIndices in rawRejectedSampleIndicesByPredClass.values():
            finalRejectedIndices.update(rawRejectedIndices)

    predClassTotalCount = {}
    predClassRawRejectedCount = {}
    predClassCappedRejectedCount = {}
    predClassFinalRejectionRate = {}
    for classIndex in range(numClasses):
        classLabel = indexToLabel[classIndex]
        classSampleIndices = predClassSampleIndices.get(classIndex, [])
        classTotalCount = len(classSampleIndices)
        rawRejectedCount = len(rawRejectedSampleIndicesByPredClass.get(classIndex, []))
        cappedRejectedCount = sum(1 for rowIndex in classSampleIndices if rowIndex in finalRejectedIndices)
        predClassTotalCount[classLabel] = int(classTotalCount)
        predClassRawRejectedCount[classLabel] = int(rawRejectedCount)
        predClassCappedRejectedCount[classLabel] = int(cappedRejectedCount)
        predClassFinalRejectionRate[classLabel] = (
            cappedRejectedCount / classTotalCount if classTotalCount > 0 else 0.0
        )

    for rowIndex, sampleRow in enumerate(sampleRows):
        predLabel = (
            rejectionClass
            if rowIndex in finalRejectedIndices
            else indexToLabel[sampleRow["pred_index"]]
        )
        allPreds.append(predLabel)
        allLabels.append(int(sampleRow["true_label"]))
        predictionRows.append({
            "batch_idx": sampleRow["batch_idx"],
            "sample_idx": sampleRow["sample_idx"],
            "true_label": int(sampleRow["true_label"]),
            "pred_label": predLabel,
            "pred_index": int(sampleRow["pred_index"]),
            "confidence": float(sampleRow["confidence"]),
            "rejected": bool(predLabel == rejectionClass),
        })

    # 完整混淆矩阵保留拒绝类列，只用于分析哪些样本被拒绝。
    allClassLabels = list(keptClassLabels) + [rejectionClass]
    cm = confusionMatrix(allLabels, allPreds, labels=allClassLabels)

    # 主指标不包含拒绝识别类：过滤掉 pred_label == -1 的样本后再计算。
    evalPairs = [
        (trueLabel, predLabel)
        for trueLabel, predLabel in zip(allLabels, allPreds)
        if predLabel != rejectionClass
    ]
    evalLabels = [trueLabel for trueLabel, _ in evalPairs]
    evalPreds = [predLabel for _, predLabel in evalPairs]
    evalCm = confusionMatrix(evalLabels, evalPreds, labels=list(keptClassLabels))
    evalTotal = evalCm.sum()

    metrics = {
        'accuracy': (np.diag(evalCm).sum() / evalTotal) if evalTotal > 0 else 0.0,
        'per_class_metrics': {},
        'confusion_matrix': cm,
        'class_labels': allClassLabels,
        'eval_confusion_matrix': evalCm,
        'eval_class_labels': list(keptClassLabels),
        'predictions': predictionRows,
    }
    
    # 手动计算每个真实类别的 precision、recall、F1 和 support。
    # 这里使用 evalCm，因此拒绝样本不进入 TP/FP/FN/support。
    for i, classLabel in enumerate(keptClassLabels):
        tp = evalCm[i, i]
        fp = evalCm[:, i].sum() - tp 
        fn = evalCm[i, :].sum() - tp 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class_metrics'][classLabel] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': evalCm[i, :].sum()
        }

    # 使用过滤后的 evalLabels/evalPreds 计算整体平均指标；拒绝类不参与。
    microP, microR, microF1, _ = precisionRecallFscoreSupport(
        evalLabels, evalPreds, labels=list(keptClassLabels),
        average='micro', zero_division=0
    )
    macroP, macroR, macroF1, _ = precisionRecallFscoreSupport(
        evalLabels, evalPreds, labels=list(keptClassLabels),
        average='macro', zero_division=0
    )
    weightedP, weightedR, weightedF1, _ = precisionRecallFscoreSupport(
        evalLabels, evalPreds, labels=list(keptClassLabels),
        average='weighted', zero_division=0
    )
    metrics['micro_precision'] = microP
    metrics['micro_recall'] = microR
    metrics['micro_f1'] = microF1
    metrics['macro_precision'] = macroP
    metrics['macro_recall'] = macroR
    metrics['macro_f1'] = macroF1
    metrics['weighted_precision'] = weightedP
    metrics['weighted_recall'] = weightedR
    metrics['weighted_f1'] = weightedF1
    metrics['confusion_matrix'] = cm
    metrics['num_eval_samples'] = len(evalPairs)

    # 拒绝率用于观察阈值 rejectionThreshold 对预测覆盖率的影响。
    rejectionCount = (np.array(allPreds) == rejectionClass).sum()
    rejectionRate = rejectionCount / len(allPreds) if allPreds else 0.0
    classSampleCount = {classLabel: 0 for classLabel in keptClassLabels}
    classRejectedCount = {classLabel: 0 for classLabel in keptClassLabels}
    for trueLabel, predLabel in zip(allLabels, allPreds):
        if trueLabel not in classSampleCount:
            continue
        classSampleCount[trueLabel] += 1
        if predLabel == rejectionClass:
            classRejectedCount[trueLabel] += 1
    classRejectionRate = {
        classLabel: (
            classRejectedCount[classLabel] / classSampleCount[classLabel]
            if classSampleCount[classLabel] > 0 else 0.0
        )
        for classLabel in keptClassLabels
    }
    metrics['rejection_rate'] = rejectionRate
    metrics['rejectMode'] = rejectMode
    metrics['rejectScoreSource'] = rejectScoreSource
    metrics['useAdaptiveRejectThreshold'] = bool(useAdaptiveRejectThreshold)
    metrics['rejectionThreshold'] = float(rejectionThreshold)
    metrics['rejectThresholdByClass'] = rejectThresholdByClass if useAdaptiveRejectThreshold else None
    metrics['rejectThresholdByClassLabel'] = (
        {
            classLabel: rejectThresholdByClass[classIndex]
            for classIndex, classLabel in enumerate(keptClassLabels)
        }
        if useAdaptiveRejectThreshold else None
    )
    metrics['rejectPercentile'] = rejectPercentile
    metrics['minRejectThreshold'] = minRejectThreshold
    metrics['maxRejectThreshold'] = maxRejectThreshold
    metrics['maxRejectRatePerClass'] = maxRejectRatePerClass
    metrics['overallRejectionRate'] = rejectionRate
    metrics['coverage'] = 1.0 - rejectionRate
    metrics['predClassTotalCount'] = predClassTotalCount
    metrics['predClassRawRejectedCount'] = predClassRawRejectedCount
    metrics['predClassCappedRejectedCount'] = predClassCappedRejectedCount
    metrics['predClassFinalRejectionRate'] = predClassFinalRejectionRate
    metrics['classSampleCount'] = classSampleCount
    metrics['classRejectedCount'] = classRejectedCount
    metrics['classRejectionRate'] = classRejectionRate
    metrics['num_samples'] = len(allPreds)
    if wGraphValueList and wPatternValueList:
        allWGraphValues = torch.cat(wGraphValueList)
        allWPatternValues = torch.cat(wPatternValueList)
        wPatternByClassMean = {
            classLabel: (
                float(np.mean(wPatternByClass[classLabel]))
                if wPatternByClass.get(classLabel) else None
            )
            for classLabel in keptClassLabels
        }
        metrics['wGraphMean'] = float(allWGraphValues.mean().item())
        metrics['wGraphStd'] = float(allWGraphValues.std(unbiased=False).item())
        metrics['wGraphMin'] = float(allWGraphValues.min().item())
        metrics['wGraphMax'] = float(allWGraphValues.max().item())
        metrics['wPatternMean'] = float(allWPatternValues.mean().item())
        metrics['wPatternStd'] = float(allWPatternValues.std(unbiased=False).item())
        metrics['wPatternMin'] = float(allWPatternValues.min().item())
        metrics['wPatternMax'] = float(allWPatternValues.max().item())
        metrics['wPatternByClassMean'] = wPatternByClassMean
        if splitName is not None:
            logger.info(
                "[ConservativeDenseGate][%s] wGraphMean=%.6f | wGraphStd=%.6f | wPatternMean=%.6f | wPatternStd=%.6f | wPatternByClassMean=%s",
                splitName,
                metrics['wGraphMean'],
                metrics['wGraphStd'],
                metrics['wPatternMean'],
                metrics['wPatternStd'],
                wPatternByClassMean,
            )

    return metrics


def buildAdaptiveRejectThresholds(
    model,
    valLoader,
    device,
    numClasses: int,
    rejectPercentile: float,
    minRejectThreshold: float,
    maxRejectThreshold: float,
    fallbackRejectThreshold: float,
) -> List[float]:
    """Estimate class-wise reject thresholds from validation-set predicted-class probabilities."""
    if float(minRejectThreshold) > float(maxRejectThreshold):
        raise ValueError(
            f"minRejectThreshold={minRejectThreshold} cannot be greater than maxRejectThreshold={maxRejectThreshold}."
        )
    model.eval()
    classScoreList = {classIndex: [] for classIndex in range(numClasses)}

    with torch.no_grad():
        for batch in valLoader:
            batch = batch.to(device)
            if not hasattr(batch, "patternFeature"):
                raise ValueError("batch is missing patternFeature, cannot build adaptive reject thresholds.")
            patternFeature = batch.patternFeature

            finalLogits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, patternFeature)
            scoreTensor = F.softmax(finalLogits, dim=1).detach().cpu()
            predConfidenceTensor, predClassTensor = scoreTensor.max(dim=1)

            for predClassValue, predConfidenceValue in zip(
                predClassTensor.tolist(),
                predConfidenceTensor.tolist(),
            ):
                predClassIndex = int(predClassValue)
                if 0 <= predClassIndex < numClasses:
                    classScoreList[predClassIndex].append(float(predConfidenceValue))

    rejectThresholdByClass = []
    for classIndex in range(numClasses):
        scoreList = classScoreList[classIndex]
        if not scoreList:
            threshold = max(float(fallbackRejectThreshold), float(minRejectThreshold))
            threshold = min(threshold, float(maxRejectThreshold))
            logger.warning(
                "Adaptive reject threshold fallback: classIndex=%s has no validation samples; use global threshold %.6f",
                classIndex,
                threshold,
            )
        else:
            rawThreshold = float(np.percentile(np.asarray(scoreList, dtype=np.float64), float(rejectPercentile)))
            threshold = max(rawThreshold, float(minRejectThreshold))
            threshold = min(threshold, float(maxRejectThreshold))
            logger.info(
                "Adaptive reject threshold: classIndex=%s | sampleCount=%s | rawPercentile=%.6f | clipped=%.6f",
                classIndex,
                len(scoreList),
                rawThreshold,
                threshold,
            )
        rejectThresholdByClass.append(float(threshold))

    return rejectThresholdByClass


def logRejectSummary(splitName: str, metrics: Dict[str, Any], keptClassLabels: List[int]) -> None:
    """Log reject mode, thresholds, per-class rejected counts, and coverage."""
    logger.info(
        "[%s][Reject] mode=%s | scoreSource=%s | globalThreshold=%.6f | maxRejectRatePerClass=%s | overallRejectionRate=%.4f | coverage=%.4f",
        splitName,
        metrics.get("rejectMode", "global"),
        metrics.get("rejectScoreSource", "softmaxProbability"),
        float(metrics.get("rejectionThreshold", 0.0)),
        metrics.get("maxRejectRatePerClass"),
        float(metrics.get("overallRejectionRate", metrics.get("rejection_rate", 0.0))),
        float(metrics.get("coverage", 1.0 - metrics.get("rejection_rate", 0.0))),
    )
    if metrics.get("useAdaptiveRejectThreshold", False):
        logger.info(
            "[%s][Reject] rejectPercentile=%s | minRejectThreshold=%s | maxRejectThreshold=%s | rejectThresholdByClass=%s | rejectThresholdByClassLabel=%s",
            splitName,
            metrics.get("rejectPercentile"),
            metrics.get("minRejectThreshold"),
            metrics.get("maxRejectThreshold"),
            metrics.get("rejectThresholdByClass"),
            metrics.get("rejectThresholdByClassLabel"),
        )
    predClassTotalCount = metrics.get("predClassTotalCount", {})
    predClassRawRejectedCount = metrics.get("predClassRawRejectedCount", {})
    predClassCappedRejectedCount = metrics.get("predClassCappedRejectedCount", {})
    predClassFinalRejectionRate = metrics.get("predClassFinalRejectionRate", {})
    for classLabel in keptClassLabels:
        logger.info(
            "[%s][Reject][PredClass] class=%s | classTotalCount=%s | rawRejectedCount=%s | cappedRejectedCount=%s | finalRejectionRate=%.4f",
            splitName,
            classLabel,
            int(predClassTotalCount.get(classLabel, 0)),
            int(predClassRawRejectedCount.get(classLabel, 0)),
            int(predClassCappedRejectedCount.get(classLabel, 0)),
            float(predClassFinalRejectionRate.get(classLabel, 0.0)),
        )
    classRejectedCount = metrics.get("classRejectedCount", {})
    classRejectionRate = metrics.get("classRejectionRate", {})
    classSampleCount = metrics.get("classSampleCount", {})
    for classLabel in keptClassLabels:
        logger.info(
            "[%s][Reject][TrueClass] class=%s | sampleCount=%s | rejectedCount=%s | classRejectionRate=%.4f",
            splitName,
            classLabel,
            int(classSampleCount.get(classLabel, 0)),
            int(classRejectedCount.get(classLabel, 0)),
            float(classRejectionRate.get(classLabel, 0.0)),
        )
    
def runTrainAndValModel(model, trainLoader, valLoader, optimizer, scheduler, lossFn, device, keptClassLabels, epochs=100, patience=20, rejectionThreshold=0.0, rejectMode="global"):
    """
    完整训练循环：训练、验证、学习率调度和 EarlyStopping。

    重要约定:
        最佳模型选择、学习率调度和 EarlyStopping 只使用验证集非拒绝 Accuracy。
        F1 仍会记录到日志和 history 中，但不参与训练控制。

    输出:
        加载了最佳验证集权重的 model。
    """
    bestAcc = 0
    noImproveEpochs = 0
    bestState = None
    bestEpoch = 0
    bestValMetrics = None
    history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\n[Epoch {epoch}]")
        # 打印当前学习率，方便观察 ReduceLROnPlateau 是否生效。
        logger.info(f"当前学习率 (Current LR): {optimizer.param_groups[0]['lr']:.6f}")


        trainLoss = trainOneEpoch(model, trainLoader, optimizer, lossFn, device)
        trainMetrics = evaluateModel(
            model,
            trainLoader,
            device,
            keptClassLabels,
            rejectionThreshold=rejectionThreshold,
            rejectMode=rejectMode,
            splitName="train",
        )
        logger.info(f"训练集 -> 非拒绝准确率: {trainMetrics['accuracy']:.4f} | 非拒绝微平均F1: {trainMetrics['micro_f1']:.4f}")


        valMetrics = evaluateModel(
            model,
            valLoader,
            device,
            keptClassLabels,
            rejectionThreshold=rejectionThreshold,
            rejectMode=rejectMode,
            splitName="val",
        )
        logger.info(f"训练损失: {trainLoss:.4f}")
        logger.info(f"验证集 -> 非拒绝准确率: {valMetrics['accuracy']:.4f} | 非拒绝微平均F1: {valMetrics['micro_f1']:.4f}")
        
        # 逐类别指标能帮助定位某些设备类别是否明显难识别。
        for classId, classMetrics in valMetrics['per_class_metrics'].items():
            logger.info(
                f"类别 {classId}: 精度={classMetrics['precision']:.4f}, "
                f"召回率={classMetrics['recall']:.4f}, F1={classMetrics['f1']:.4f}, "
                f"支持样本数={classMetrics['support']}"
            )

        currentAcc = valMetrics['accuracy']
        currentF1 = valMetrics['micro_f1']
        # 训练控制指标仅使用验证集非拒绝 Accuracy；F1 只作为辅助观察指标记录。
        scheduler.step(currentAcc)

        # history 会在训练结束后保存到 JSON，便于画训练曲线或复现实验。
        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
            "train_loss": trainLoss,
            "train_accuracy": trainMetrics['accuracy'],
            "train_micro_f1": trainMetrics['micro_f1'],
            "val_accuracy": currentAcc,
            "val_micro_f1": currentF1,
            "val_macro_f1": valMetrics['macro_f1'],
            "val_rejection_rate": valMetrics['rejection_rate'],
        })

        # 只要非拒绝 Accuracy 创新高，就保存当前模型权重。
        if bestState is None or currentAcc > bestAcc:
            bestAcc = currentAcc
            bestEpoch = epoch
            bestState = copy.deepcopy(model.state_dict())
            bestValMetrics = {
                key: value for key, value in valMetrics.items()
                if key not in {"predictions"}
            }
            noImproveEpochs = 0
            logger.info(f"✅ 验证集非拒绝Accuracy提升: Accuracy={currentAcc:.4f}")
            logger.info(f"   辅助指标: Micro-F1={currentF1:.4f}")
        else:
            noImproveEpochs += 1
            logger.info(f"⚠️ 非拒绝Accuracy未提升: Accuracy={currentAcc:.4f} | 连续无提升轮数: {noImproveEpochs}")
            logger.info(f"   辅助指标: Micro-F1={currentF1:.4f}")
        if noImproveEpochs >= patience:
            logger.info("⛔ 触发EarlyStopping: 验证集非拒绝 Accuracy 连续未提升。")
            break

    # 训练结束后恢复到验证集非拒绝 Accuracy 最好的那一轮。
    if bestState:
        model.load_state_dict(bestState)
        logger.info(f"已加载第 {bestEpoch} 轮最佳模型，其验证集非拒绝准确率为: {bestAcc:.4f}")

    model.trainingHistory = history
    model.bestEpoch = bestEpoch
    model.bestValMetrics = bestValMetrics
    return model


# =========================
# 8. 类别过滤与数据划分
# =========================

def filterExtremeImbalanceClasses(
    dataList,
    imbalanceRatioThreshold: float = 200.0,
    minCountThreshold: int = 30
) -> Tuple[list, list, dict]:
    """
    过滤极度不平衡或样本过少的类别。

    执行位置:
        构图完成后、训练/验证/测试划分之前。

    规则:
        1. 如果最大类别数 / 最小类别数超过 imbalanceRatioThreshold，
           删除样本数小于 max_count / imbalanceRatioThreshold 的类别；
        2. 无论第 1 条是否触发，都删除样本数小于 minCountThreshold 的类别；
        3. 不修改原始标签 data.y，只新增训练用标签 data.y_mapped。

    输出:
        filteredDataList: 过滤后的图列表；
        keptClasses: 保留下来的原始类别标签；
        stats: 过滤前后的类别统计。
    """
    originalCounts = Counter(getLabelValue(data.y) for data in dataList)
    if not originalCounts:
        raise ValueError("数据集为空，无法进行类别过滤。")

    maxCount = max(originalCounts.values())
    minCount = min(originalCounts.values())
    ratio = (maxCount / minCount) if minCount > 0 else float("inf")

    removedByRatio = set()
    ratioGate = ratio > imbalanceRatioThreshold
    ratioCountThreshold = maxCount / imbalanceRatioThreshold
    if ratioGate:
        removedByRatio = {
            cls for cls, cnt in originalCounts.items()
            if cnt < ratioCountThreshold
        }

    removedByMinCount = {
        cls for cls, cnt in originalCounts.items()
        if cnt < minCountThreshold
    }
    removedClasses = removedByRatio.union(removedByMinCount)
    keptClasses = sorted([cls for cls in originalCounts.keys() if cls not in removedClasses])

    if not keptClasses:
        raise ValueError("过滤后没有剩余类别，请调整阈值。")

    filteredDataList = [data for data in dataList if getLabelValue(data.y) in keptClasses]

    # 模型训练需要 0 到 numClasses-1 的连续标签；评估时仍使用原始标签。
    labelToMapped = {label: idx for idx, label in enumerate(keptClasses)}
    for data in filteredDataList:
        mappedIdx = labelToMapped[getLabelValue(data.y)]
        data.y_mapped = torch.tensor([mappedIdx], dtype=torch.long)

    stats = {
        "original_counts": originalCounts,
        "kept_classes": keptClasses,
        "removed_classes": sorted(list(removedClasses)),
        "ratio_gate_triggered": ratioGate,
        "max_count": maxCount,
        "min_count": minCount,
        "ratio": ratio,
        "ratio_count_threshold": ratioCountThreshold,
        "removed_by_ratio": sorted(list(removedByRatio)),
        "removed_by_min_count": sorted(list(removedByMinCount)),
        "kept_counts": Counter(getLabelValue(data.y) for data in filteredDataList),
    }
    return filteredDataList, keptClasses, stats


def stratifiedSplit(dataList, testRatio=0.2, seed=42):
    """
    按类别分层划分图数据。

    输入:
        dataList: PyG Data 图列表；
        testRatio: 第二个返回集合所占比例；
        seed: 随机种子。

    输出:
        trainData, testData。

    注意:
        这个函数也用于 trainVal -> val 的二次划分，因此第二个返回值不一定总是测试集。
    """
    random.seed(seed)
    classBuckets = defaultdict(list)

    # 先按原始标签分桶，保证每个类别都按比例拆分。
    for data in dataList:
        label = getLabelValue(data.y)
        classBuckets[label].append(data)

    trainData, testData = [], []

    for label, samples in classBuckets.items():
        random.shuffle(samples)
        testSize = int(len(samples) * testRatio)
        testData.extend(samples[:testSize])
        trainData.extend(samples[testSize:])

    # 拆分后再次打乱，避免类别顺序影响 DataLoader。
    random.shuffle(trainData)
    random.shuffle(testData)

    return trainData, testData

def filterClasses(dataList, keepClasses):
    """旧版辅助函数：只保留指定类别，并直接改写 data.y 为连续编号。主流程当前不使用。"""
    filteredList = []
    labelMap = {old: new for new, old in enumerate(keepClasses)}

    for data in dataList:
        label = int(data.y.item())
        if label in keepClasses:
            data.y = torch.tensor([labelMap[label]], dtype=torch.long)  # 重新编号
            filteredList.append(data)

    return filteredList


# =========================
# 9. 数据加载与图结构校验
# =========================

def normalizeEdgeAttr(data):
    """
    统一图的 edge_attr 形状。

    GATv2Conv 在本脚本中要求 edge_attr 为 [num_edges, 2]。
    如果图没有 edge_attr，则按边数补零；如果只有 1 维，则补一列 0。
    """
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        numEdges = data.edge_index.size(1) if hasattr(data, "edge_index") else 0
        data.edge_attr = torch.zeros((numEdges, 2), dtype=torch.float32)
        return data

    edgeAttr = data.edge_attr
    if not isinstance(edgeAttr, torch.Tensor):
        edgeAttr = torch.as_tensor(edgeAttr, dtype=torch.float32)
    if edgeAttr.dim() == 1:
        if edgeAttr.numel() == 0:
            edgeAttr = torch.empty((0, 2), dtype=edgeAttr.dtype, device=edgeAttr.device)
        elif edgeAttr.numel() == data.edge_index.size(1) * 2:
            edgeAttr = edgeAttr.view(-1, 2)
        else:
            edgeAttr = edgeAttr.view(-1, 1)
    if edgeAttr.size(1) == 1:
        pad = torch.zeros(edgeAttr.size(0), 1, dtype=edgeAttr.dtype, device=edgeAttr.device)
        edgeAttr = torch.cat([edgeAttr, pad], dim=1)
    data.edge_attr = edgeAttr[:, :2].float()
    return data


def validateAndPrepareGraphs(dataList: List[Any]) -> List[Any]:
    """
    检查并整理图列表，过滤无法训练的图。

    每个有效图至少需要:
        data.x: 节点特征；
        data.edge_index: 边索引；
        data.y: 图标签。

    输出:
        cleanList: 可以送入 DataLoader 的图列表。
    """
    cleanList = []
    skipped = []
    for idx, data in enumerate(dataList):
        if not hasattr(data, "x") or data.x is None:
            skipped.append((idx, "missing_x"))
            continue
        if not hasattr(data, "edge_index") or data.edge_index is None:
            skipped.append((idx, "missing_edge_index"))
            continue
        if not hasattr(data, "y"):
            skipped.append((idx, "missing_y"))
            continue
        if data.x.dim() != 2 or data.x.numel() == 0:
            skipped.append((idx, "bad_x_shape"))
            continue
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        cleanList.append(normalizeEdgeAttr(data))

    if skipped:
        logger.warning(f"数据校验跳过 {len(skipped)} 个图，前10个原因: {skipped[:10]}")
    if not cleanList:
        raise ValueError("没有可用于训练的有效图数据。")
    return cleanList


def fitNodeFeatureStandardizer(trainDataset, epsilon: float = 1e-6) -> Dict[str, Any]:
    """仅使用训练集节点拟合逐特征均值和标准差。

    使用 float64 累加，减少大量纲流量特征在统计时的数值误差。
    标准差过小的常量特征将使用 1 作为缩放因子，避免除零。
    """
    if not trainDataset:
        raise ValueError("训练集为空，无法拟合节点特征标准化参数。")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(f"nodeFeatureStdEpsilon must be positive, got {epsilon}.")

    featureDim = int(trainDataset[0].x.size(1))
    featureSum = torch.zeros(featureDim, dtype=torch.float64)
    featureSquareSum = torch.zeros(featureDim, dtype=torch.float64)
    nodeCount = 0

    for dataIndex, data in enumerate(trainDataset):
        x = data.x.detach().to(dtype=torch.float64, device="cpu")
        if x.dim() != 2 or int(x.size(1)) != featureDim:
            raise ValueError(
                f"train dataset sample {dataIndex} x shape mismatch: "
                f"got {tuple(x.shape)}, expected [num_nodes, {featureDim}]."
            )
        if not torch.isfinite(x).all():
            raise ValueError(
                f"train dataset sample {dataIndex} contains NaN or Inf node features."
            )
        featureSum += x.sum(dim=0)
        featureSquareSum += (x * x).sum(dim=0)
        nodeCount += int(x.size(0))

    if nodeCount <= 0:
        raise ValueError("训练集不包含节点，无法拟合节点特征标准化参数。")

    mean = featureSum / float(nodeCount)
    variance = (featureSquareSum / float(nodeCount)) - mean.square()
    rawStd = torch.sqrt(torch.clamp(variance, min=0.0))
    constantMask = rawStd < epsilon
    scale = torch.where(constantMask, torch.ones_like(rawStd), rawStd)

    return {
        "mean": mean.to(dtype=torch.float32),
        "scale": scale.to(dtype=torch.float32),
        "rawStd": rawStd.to(dtype=torch.float32),
        "constantFeatureIndices": torch.nonzero(constantMask, as_tuple=False).view(-1).tolist(),
        "featureDim": featureDim,
        "nodeCount": nodeCount,
        "epsilon": epsilon,
    }


def applyNodeFeatureStandardizer(
    dataset,
    standardizer: Dict[str, Any],
    splitName: str,
) -> None:
    """将训练集拟合的同一组参数应用到指定数据集。"""
    featureDim = int(standardizer["featureDim"])
    meanCpu = standardizer["mean"]
    scaleCpu = standardizer["scale"]

    for dataIndex, data in enumerate(dataset):
        if data.x.dim() != 2 or int(data.x.size(1)) != featureDim:
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} x shape mismatch: "
                f"got {tuple(data.x.shape)}, expected [num_nodes, {featureDim}]."
            )
        mean = meanCpu.to(device=data.x.device, dtype=data.x.dtype)
        scale = scaleCpu.to(device=data.x.device, dtype=data.x.dtype)
        standardizedX = (data.x - mean) / scale
        if not torch.isfinite(standardizedX).all():
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} produced NaN or Inf after standardization."
            )
        data.x = standardizedX


def standardizeNodeFeatureDatasets(
    trainDataset,
    valDataset,
    testDataset,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """仅在 train 上拟合，再对 train/val/test 应用节点特征标准化。"""
    standardizer = fitNodeFeatureStandardizer(trainDataset, epsilon=epsilon)
    applyNodeFeatureStandardizer(trainDataset, standardizer, "train")
    applyNodeFeatureStandardizer(valDataset, standardizer, "val")
    applyNodeFeatureStandardizer(testDataset, standardizer, "test")

    logger.info(
        "Node feature standardization fitted on train only: nodeCount=%s | featureDim=%s | constantFeatureIndices=%s",
        standardizer["nodeCount"],
        standardizer["featureDim"],
        standardizer["constantFeatureIndices"],
    )
    return {
        "fitSplit": "train",
        "featureDim": int(standardizer["featureDim"]),
        "nodeCount": int(standardizer["nodeCount"]),
        "epsilon": float(standardizer["epsilon"]),
        "mean": standardizer["mean"].tolist(),
        "scale": standardizer["scale"].tolist(),
        "constantFeatureIndices": list(standardizer["constantFeatureIndices"]),
    }


def loadGraphDataset(config: Dict[str, Any]) -> List[Any]:
    """
    根据配置加载图数据。

    dataSource = "builder":
        调用 TrafficGraphBuild 从原始流量数据构建图。
    dataSource = "processed":
        从 datasetPath/processed/*.pt 读取已保存的 PyG 图。

    输出:
        经过 validateAndPrepareGraphs 清洗后的图列表。
    """
    datasetPath = config["datasetPath"]
    if config.get("dataSource", "builder") == "processed":
        dataList = loadProcessedGraphs(datasetPath)
    else:
        builder = TrafficGraphBuild(datasetPath)
        dataList, _, _ = builder.loadData()
    dataList = validateAndPrepareGraphs(dataList)
    logger.info(f"数据集加载完成: {datasetPath}, 有效图数量={len(dataList)}")
    logger.info(f"原始标签分布: {Counter(getLabelValue(data.y) for data in dataList)}")
    return dataList


def getNumWorkers(config: Dict[str, Any]) -> int:
    """根据配置或 CPU 核心数决定 DataLoader 的 numWorkers。"""
    if config.get("numWorkers") is not None:
        return int(config["numWorkers"])
    cpuCount = os.cpu_count() or 2
    return max(0, min(8, cpuCount // 2))


def makeDataLoader(dataset, batchSize, shuffle, numWorkers, pinMemory):
    """创建 PyG DataLoader，把多个图打包成一个 Batch。"""
    return DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
        pin_memory=pinMemory,
    )


def attachPatternFeatureToDataset(
    dataset,
    patternLibrary,
    appPortSet,
    ngramLengths,
    scoreMode,
    numClasses: int,
    splitName: str,
    portSequenceField: str,
) -> None:
    """Compute and store patternFeature on each Data object as [1, 2 * numClasses + 2]."""
    patternFeatureDim = numClasses * 2 + 2
    for dataIndex, data in enumerate(dataset):
        if not hasattr(data, portSequenceField):
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} is missing {portSequenceField}. "
                f"Please rebuild graphs with {portSequenceField} before fusion training."
            )
        if not hasattr(data, "y_mapped"):
            raise ValueError(f"{splitName} dataset sample {dataIndex} is missing y_mapped.")

        labelValue = data.y_mapped
        if hasattr(labelValue, "view"):
            classIndex = int(labelValue.view(-1)[0].item())
        else:
            classIndex = int(labelValue)
        if classIndex < 0 or classIndex >= numClasses:
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} y_mapped={classIndex} "
                f"is outside numClasses={numClasses}."
            )

        patternFeatureDict = computePatternFeature(
            getattr(data, portSequenceField),
            patternLibrary,
            appPortSet,
            ngramLengths,
            scoreMode,
            numClasses,
        )
        patternFeature = patternFeatureDict["patternFeature"]
        if len(patternFeature) != patternFeatureDim:
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} patternFeature length mismatch: "
                f"got {len(patternFeature)}, expected {patternFeatureDim}."
            )

        featureTensor = torch.tensor([patternFeature], dtype=torch.float32)
        if tuple(featureTensor.shape) != (1, patternFeatureDim):
            raise ValueError(
                f"{splitName} dataset sample {dataIndex} patternFeature shape mismatch: "
                f"got {tuple(featureTensor.shape)}, expected (1, {patternFeatureDim})."
            )
        data.patternFeature = featureTensor


def logFormattedConfusionMatrix(loggerObj, confusionMatrix, classLabels, logPerClassMetrics: bool = True) -> None:
    """Log a readable confusion matrix and per-class metrics."""
    cm = np.asarray(confusionMatrix)
    labels = list(classLabels)
    if cm.shape != (len(labels), len(labels)):
        raise ValueError(
            f"confusionMatrix shape {cm.shape} does not match classLabels length {len(labels)}"
        )

    maxValueLength = max([len(str(int(value))) for value in cm.flatten()] + [1])
    maxLabelLength = max([len(str(label)) for label in labels] + [1])
    cellWidth = max(6, maxValueLength + 2, maxLabelLength + 2)

    loggerObj.info("(行: 真实标签, 列: 预测标签)")
    header = "      " + "".join(f"{str(label):<{cellWidth}}" for label in labels)
    loggerObj.info(header)
    loggerObj.info("    " + "-" * max(1, len(header) - 4))
    for label, row in zip(labels, cm):
        rowText = f" {str(label):<4} |" + "".join(f"{int(value):<{cellWidth}}" for value in row)
        loggerObj.info(rowText)

    if not logPerClassMetrics:
        return

    loggerObj.info("-" * 20)
    for classIndex, classLabel in enumerate(labels):
        truePositive = float(cm[classIndex, classIndex])
        falsePositive = float(cm[:, classIndex].sum() - cm[classIndex, classIndex])
        falseNegative = float(cm[classIndex, :].sum() - cm[classIndex, classIndex])
        support = int(cm[classIndex, :].sum())
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1Score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        loggerObj.info(
            "     类别 %s: 精度=%.4f, 召回率=%.4f, F1=%.4f, 支持样本数=%s",
            classLabel,
            precision,
            recall,
            f1Score,
            support,
        )


def logFinalTestSummary(config: Dict[str, Any], keptClassLabels: List[int], testMetrics: Dict[str, Any]) -> None:
    """Log final non-rejected test metrics, full confusion matrix, and kept-class metrics."""
    logger.info(
        "测试集 -> 非拒绝准确率: %.4f | 非拒绝微平均F1: %.4f | 非拒绝宏平均F1: %.4f",
        testMetrics["accuracy"],
        testMetrics["micro_f1"],
        testMetrics["macro_f1"],
    )
    logger.info(
        "配置信息: epochs=%s, lr=%s, patience=%s, numClasses=%s, keptClassLabels=%s",
        config["epochs"],
        config["lr"],
        config["patience"],
        len(keptClassLabels),
        keptClassLabels,
    )
    logRejectSummary("test", testMetrics, keptClassLabels)
    logFormattedConfusionMatrix(
        logger,
        testMetrics["confusion_matrix"],
        testMetrics["class_labels"],
        logPerClassMetrics=False,
    )

    logger.info("-" * 20)
    for classLabel in keptClassLabels:
        if classLabel not in testMetrics["per_class_metrics"]:
            raise KeyError(f"testMetrics missing per_class_metrics for class {classLabel}")
        classMetrics = testMetrics["per_class_metrics"][classLabel]
        logger.info(
            "     类别 %s: 精度=%.4f, 召回率=%.4f, F1=%.4f, 支持样本数=%s",
            classLabel,
            classMetrics["precision"],
            classMetrics["recall"],
            classMetrics["f1"],
            int(classMetrics["support"]),
        )


# =========================
# 10. 结果保存
# =========================

def saveConfusionMatrixCsv(path: str, metrics: Dict[str, Any]) -> None:
    """把混淆矩阵保存成 CSV，行是真实标签，列是预测标签。"""
    labels = metrics["class_labels"]
    cm = np.asarray(metrics["confusion_matrix"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row.tolist())


def saveClassMetricsCsv(path: str, metrics: Dict[str, Any]) -> None:
    """保存每个类别的 precision、recall、F1 和 support。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_label", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for classLabel, classMetrics in metrics["per_class_metrics"].items():
            writer.writerow({
                "class_label": classLabel,
                "precision": classMetrics["precision"],
                "recall": classMetrics["recall"],
                "f1": classMetrics["f1"],
                "support": classMetrics["support"],
            })


def savePredictionsCsv(path: str, predictions: List[Dict[str, Any]]) -> None:
    """保存测试集中每个样本的真实标签、预测标签、置信度和是否被拒绝。"""
    fieldnames = ["batch_idx", "sample_idx", "true_label", "pred_label", "pred_index", "confidence", "rejected"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def saveRunArtifacts(
    runDir: str,
    checkpointDir: str,
    model,
    optimizer,
    config: Dict[str, Any],
    keptClassLabels: List[int],
    inputDim: int,
    testMetrics: Dict[str, Any],
    filterStats: Dict[str, Any],
) -> None:
    """
    保存一次训练运行的全部产物。

    输出目录:
        results/gnn_iot_<时间>/
            config.json
            filter_stats.json
            training_history.json
            test_metrics.json
            confusion_matrix.csv
            per_class_metrics.csv
            test_predictions.csv

        checkpoints/traffic_gnn_<时间>.pt
            模型权重、优化器状态、类别映射和关键指标。
    """
    os.makedirs(runDir, exist_ok=True)
    os.makedirs(checkpointDir, exist_ok=True)

    saveJson(os.path.join(runDir, "config.json"), config)
    saveJson(os.path.join(runDir, "filter_stats.json"), filterStats)
    saveJson(os.path.join(runDir, "training_history.json"), {"history": getattr(model, "trainingHistory", [])})
    saveJson(os.path.join(runDir, "test_metrics.json"), testMetrics)
    saveConfusionMatrixCsv(os.path.join(runDir, "confusion_matrix.csv"), testMetrics)
    saveClassMetricsCsv(os.path.join(runDir, "per_class_metrics.csv"), testMetrics)
    savePredictionsCsv(os.path.join(runDir, "test_predictions.csv"), testMetrics.get("predictions", []))

    experimentName = config.get("experimentName")
    checkpointSuffix = f"_{experimentName}" if experimentName else ""
    checkpointPath = os.path.join(checkpointDir, f"traffic_gnn_{currentTime}{checkpointSuffix}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "input_dim": inputDim,
        "num_classes": len(keptClassLabels),
        "kept_class_labels": keptClassLabels,
        "label_to_index": {label: idx for idx, label in enumerate(keptClassLabels)},
        "best_epoch": getattr(model, "bestEpoch", 0),
        "best_val_metrics": getattr(model, "bestValMetrics", None),
        "test_metrics": {
            key: value for key, value in testMetrics.items()
            if key != "predictions"
        },
    }, checkpointPath)
    logger.info(f"最佳模型 checkpoint 已保存到: {checkpointPath}")


def isNoisyGraph(data, identicalTol: float = 1e-6, zeroNeg1RatioThreshold: float = 1.0):
    """
    判断 processed/*.pt 中的图是否明显不可用。

    判定为噪声图的典型情况:
        1. 大量节点特征全 0 或全 -1；
        2. 所有节点特征几乎完全相同。

    输出:
        isNoisy: 是否噪声图；
        info: 噪声原因和统计信息。
    """
    # 基本健壮性检查：没有节点特征或形状异常时直接视为不可用。
    if not hasattr(data, "x") or data.x is None:
        return True, {"reason": "missing_x"}
    X = data.x

    # 转到 CPU float32，避免 dtype 或设备差异影响后续判断。
    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
        except Exception:
            return True, {"reason": "x_not_tensor_and_convert_failed"}
    if X.ndimension() != 2 or X.numel() == 0:
        return True, {"reason": "bad_x_shape"}
    X = X.detach().to(dtype=torch.float32, device="cpu")  # [N, F]

    N = X.size(0)

    # 统计全 0 节点和全 -1 节点。
    rowAllZero = (X == 0).all(dim=1)
    rowAllNeg1  = (X == -1).all(dim=1)

    zeroIdx = torch.nonzero(rowAllZero, as_tuple=False).view(-1).tolist()
    neg1Idx = torch.nonzero(rowAllNeg1,  as_tuple=False).view(-1).tolist()
    zeroCnt = len(zeroIdx)
    neg1Cnt = len(neg1Idx)

    # 检查所有节点是否几乎一样；如果一样，图没有区分性的结构信息。
    ref = X[0].unsqueeze(0)
    maxAbsDiffPerRow = (X - ref).abs().max(dim=1).values
    allNodesIdentical = bool((maxAbsDiffPerRow <= identicalTol).all()) if N > 0 else True

    isNoisy = False
    info = {
        "zero_nodes": zeroCnt,
        "neg1_nodes": neg1Cnt,
        "zero_idx": zeroIdx,
        "neg1_idx": neg1Idx,
        "all_nodes_identical": allNodesIdentical,
    }

    if zeroCnt + neg1Cnt > N * zeroNeg1RatioThreshold:
        isNoisy = True
        info["reason"] = "zero_or_neg1_nodes_exceed_half"
        print("⚠️ 图包含超过一半节点特征为全零或全-1，跳过该图。")
        print("节点特征：")
        print(X)
    
    elif allNodesIdentical:
        isNoisy = True
        info["reason"] = "all_nodes_identical"
        print("⚠️ 图包含所有节点特征相同，跳过该图。")
        print("节点特征：")
        print(X)

    return isNoisy, info





def loadProcessedGraphs(datasetDir: str):
    """
    从当前构图版本对应的 processed 目录读取 PyG 图文件。

    输入:
        datasetDir: 数据集根目录。

    输出:
        dataList: 通过字段校验和噪声过滤的图列表。
    """

    processedDir = os.path.join(datasetDir, graphCacheDirName)
    if not os.path.exists(processedDir):
        raise FileNotFoundError(f"❌ 路径不存在: {processedDir}")

    dataList = []
    badFiles = []

    for fname in sorted(os.listdir(processedDir)):
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(processedDir, fname)
        try:
            obj = torch.load(fpath)
        except Exception as e:
            logging.warning(f"⚠️ 读取失败（无法反序列化）: {fpath} -> {type(e).__name__}: {e}")
            badFiles.append((fpath, "unpickle_error"))
            continue

        # 只保留带有节点特征、边索引和图标签的 PyG Data 对象。
        if not hasattr(obj, "x") or not hasattr(obj, "edge_index"):
            logging.warning(f"⚠️ 非 Graph 对象（缺少 x/edge_index）: {fpath} -> type={type(obj)}")
            badFiles.append((fpath, f"type={type(obj)}"))
            continue
        if not hasattr(obj, "y"):
            logging.warning(f"⚠️ 缺少 y 字段: {fpath}")
            badFiles.append((fpath, "missing_y"))
            continue

        # 噪声图会污染训练信号，因此在加载阶段直接跳过。
        isNoisy, info = isNoisyGraph(obj)
        if isNoisy:
            logging.warning(
                f"⚠️ 跳过噪声图: {fpath} -> 全0节点={info.get('zero_nodes',0)}, 全-1节点={info.get('neg1_nodes',0)}"
            )
            badFiles.append(
                (fpath, f"noisy_graph: zero_nodes={info.get('zero_nodes',0)}, neg1_nodes={info.get('neg1_nodes',0)}")
            )
            continue

        dataList.append(obj)

    logging.info(f"✅ 已加载 {len(dataList)} 个图数据，自 {processedDir}")
    if badFiles:
        logging.warning("⚠️ 以下 .pt 文件内容异常（已跳过）：")
        for path, reason in badFiles:
            logging.warning(f"  - {path}  ({reason})")

    return dataList


# =========================
# 11. 主流程入口
# =========================

def runSingleExperiment(
    config: Dict[str, Any],
    trainDataset,
    valDataset,
    testDataset,
    keptClassLabels: List[int],
    filterStats: Dict[str, Any],
    device,
    numWorkers: int,
    pinMemory: bool,
    runDir: str,
) -> Dict[str, Any]:
    """Train and evaluate one conservativeDenseGate setting."""

    setRandomSeed(config["seed"])
    os.makedirs(runDir, exist_ok=True)

    trainLoader = makeDataLoader(trainDataset, config["batchSize"], True, numWorkers, pinMemory)
    valLoader = makeDataLoader(valDataset, config["batchSize"], False, numWorkers, pinMemory)
    testLoader = makeDataLoader(testDataset, config["batchSize"], False, numWorkers, pinMemory)

    epochs = config["epochs"]
    lr = config["lr"]
    patience = config["patience"]
    rejectionThreshold = config["rejectionThreshold"]
    checkpointDir = config["checkpointDir"]
    numClasses = len(keptClassLabels)
    inputDim = trainDataset[0].x.shape[1]
    usePatternFeature = True
    fusionMode = config.get("fusionMode", "conservativeDenseGate")
    if fusionMode != "conservativeDenseGate":
        raise ValueError(
            f"Unsupported fusionMode={fusionMode}; "
            "this file only implements conservativeDenseGate."
        )
    patternFeatureDim = int(config.get("patternFeatureDim", numClasses * 2 + 2))
    patternLambda = float(config.get("patternLambda", 0.2))
    rejectMode = getRejectMode(
        config.get("rejectMode"),
        config.get("useAdaptiveRejectThreshold", False),
    )
    useAdaptiveRejectThreshold = rejectMode == "adaptive"
    rejectPercentile = config.get("rejectPercentile", 5)
    minRejectThreshold = config.get("minRejectThreshold", 0.30)
    maxRejectThreshold = config.get("maxRejectThreshold", 0.65)
    maxRejectRatePerClass = config.get("maxRejectRatePerClass", 0.7)
    rejectScoreSource = config.get("rejectScoreSource", "softmaxProbability")
    if rejectScoreSource != "softmaxProbability":
        raise ValueError(
            f"Unsupported rejectScoreSource={rejectScoreSource}; this file uses softmaxProbability."
        )
    config["rejectMode"] = rejectMode
    config["rejectScoreSource"] = rejectScoreSource
    config["useAdaptiveRejectThreshold"] = useAdaptiveRejectThreshold

    logger.info("=== Start experiment: %s ===", config.get("experimentName", "singleRun"))
    logger.info(
        "fusionMode=%s | patternLambda=%.6f | usePatternFeature=%s | rejectMode=%s | rejectScoreSource=%s | portSequenceField=%s | patternFeatureDim=%s | numClasses=%s",
        fusionMode,
        patternLambda,
        usePatternFeature,
        rejectMode,
        rejectScoreSource,
        config.get("portSequenceField", "dstPortSequence"),
        patternFeatureDim,
        numClasses,
    )
    logger.info(
        "Reject config: rejectMode=%s | useAdaptiveRejectThreshold=%s | rejectionThreshold=%.6f | rejectPercentile=%s | minRejectThreshold=%s | maxRejectThreshold=%s | maxRejectRatePerClass=%s",
        rejectMode,
        useAdaptiveRejectThreshold,
        float(rejectionThreshold),
        rejectPercentile,
        minRejectThreshold,
        maxRejectThreshold,
        maxRejectRatePerClass,
    )

    gnn = TrafficGNN(
        inputDim=inputDim,
        numClasses=numClasses,
        dropout=config["dropout"],
        usePatternFeature=usePatternFeature,
        patternFeatureDim=patternFeatureDim,
        patternLambda=patternLambda,
    ).to(device)
    gnnOptimizer = Adam(gnn.parameters(), lr=lr, weight_decay=config["weightDecay"])
    scheduler = ReduceLROnPlateau(
        gnnOptimizer,
        mode='max',
        factor=config["schedulerFactor"],
        patience=config["schedulerPatience"],
    )

    labelCounts = Counter(getLabelValue(data.y) for data in trainDataset)
    alpha = torch.tensor([
        1.0 / (labelCounts[originalLabel]**0.5) if labelCounts[originalLabel] > 0 else 0.0
        for originalLabel in keptClassLabels
    ], dtype=torch.float).to(device)
    lossFn = FocalLoss(gamma=2.0, alpha=alpha).to(device)

    trainEvalRejectMode = "global" if rejectMode == "global" else "none"
    model = runTrainAndValModel(
        gnn, trainLoader, valLoader, gnnOptimizer, scheduler, lossFn, device,
        keptClassLabels=keptClassLabels, patience=patience, epochs=epochs,
        rejectionThreshold=rejectionThreshold,
        rejectMode=trainEvalRejectMode,
    )

    rejectThresholdByClass = None
    if useAdaptiveRejectThreshold:
        rejectThresholdByClass = buildAdaptiveRejectThresholds(
            model,
            valLoader,
            device,
            numClasses,
            rejectPercentile=rejectPercentile,
            minRejectThreshold=minRejectThreshold,
            maxRejectThreshold=maxRejectThreshold,
            fallbackRejectThreshold=rejectionThreshold,
        )
        logger.info("Adaptive rejectThresholdByClass=%s", rejectThresholdByClass)
    config["rejectMode"] = rejectMode
    config["rejectScoreSource"] = rejectScoreSource
    config["rejectThresholdByClass"] = rejectThresholdByClass
    config["maxRejectRatePerClass"] = maxRejectRatePerClass

    finalTrainMetrics = evaluateModel(
        model,
        trainLoader,
        device,
        keptClassLabels,
        rejectionThreshold=rejectionThreshold,
        useAdaptiveRejectThreshold=useAdaptiveRejectThreshold,
        rejectThresholdByClass=rejectThresholdByClass,
        rejectPercentile=rejectPercentile,
        minRejectThreshold=minRejectThreshold,
        maxRejectThreshold=maxRejectThreshold,
        maxRejectRatePerClass=maxRejectRatePerClass,
        rejectMode=rejectMode,
        splitName="trainFinal",
    )
    logRejectSummary("train", finalTrainMetrics, keptClassLabels)

    testMetrics = evaluateModel(
        model,
        testLoader,
        device,
        keptClassLabels,
        rejectionThreshold=rejectionThreshold,
        useAdaptiveRejectThreshold=useAdaptiveRejectThreshold,
        rejectThresholdByClass=rejectThresholdByClass,
        rejectPercentile=rejectPercentile,
        minRejectThreshold=minRejectThreshold,
        maxRejectThreshold=maxRejectThreshold,
        maxRejectRatePerClass=maxRejectRatePerClass,
        rejectMode=rejectMode,
        splitName="test",
    )
    logger.info(
        "Experiment %s test -> accuracy=%.4f | macroF1=%.4f | weightedF1=%.4f",
        config.get("experimentName", "singleRun"),
        testMetrics["accuracy"],
        testMetrics["macro_f1"],
        testMetrics["weighted_f1"],
    )
    logFinalTestSummary(config, keptClassLabels, testMetrics)

    saveRunArtifacts(
        runDir,
        checkpointDir,
        model,
        gnnOptimizer,
        config,
        keptClassLabels,
        inputDim,
        testMetrics,
        filterStats,
    )

    return {
        "experimentName": config.get("experimentName", "singleRun"),
        "experimentGroup": config.get("experimentGroup", "singleRun"),
        "fusionMode": fusionMode,
        "usePatternFeature": usePatternFeature,
        "useAdaptiveRejectThreshold": useAdaptiveRejectThreshold,
        "rejectMode": rejectMode,
        "rejectScoreSource": rejectScoreSource,
        "rejectionThreshold": float(rejectionThreshold),
        "rejectThresholdByClass": rejectThresholdByClass,
        "rejectPercentile": rejectPercentile,
        "minRejectThreshold": minRejectThreshold,
        "maxRejectThreshold": maxRejectThreshold,
        "maxRejectRatePerClass": maxRejectRatePerClass,
        "portSequenceField": config.get("portSequenceField", "dstPortSequence"),
        "patternFeatureDim": patternFeatureDim,
        "patternLambda": patternLambda,
        "accuracy": float(testMetrics["accuracy"]),
        "macroF1": float(testMetrics["macro_f1"]),
        "weightedF1": float(testMetrics["weighted_f1"]),
        "microF1": float(testMetrics["micro_f1"]),
        "rejectionRate": float(testMetrics["rejection_rate"]),
        "overallRejectionRate": float(testMetrics["overallRejectionRate"]),
        "coverage": float(testMetrics["coverage"]),
        "bestEpoch": int(getattr(model, "bestEpoch", 0)),
        "runDir": runDir,
    }


def main():
    """
    程序入口，串联完整训练流程。

    主流程:
        1. 读取配置并固定随机种子；
        2. 加载和清洗图数据；
        3. 过滤极度不平衡类别；
        4. 分层划分训练集、验证集、测试集；
        5. 初始化模型、优化器、损失函数和学习率调度器；
        6. 训练并按验证集非拒绝 Accuracy 选择最佳模型；
        7. 在测试集评估；
        8. 保存指标、预测明细和 checkpoint。
    """
    setupLogging()
    logger.info("=== 执行主程序 ===")
    config = copy.deepcopy(trainConfig)
    datasetPathOverride = os.getenv("GNN_DATASET_PATH")
    if datasetPathOverride:
        config["datasetPath"] = datasetPathOverride
        logger.info("Dataset path overridden by GNN_DATASET_PATH: %s", datasetPathOverride)
    config["batchSize"] = resolveBatchSize(
        config["batchSize"],
        os.getenv("GNN_BATCH_SIZE"),
    )
    logger.info("Effective batchSize: %s", config["batchSize"])
    setRandomSeed(config["seed"])

    # Step 1: 加载图数据。每个图对应一个固定时间窗口内的 IoT 流量快照。
    dataList = loadGraphDataset(config)
    if config.get("visualizeFirstGraph", True):
        try:
            visualizeGraph(dataList[0])  # 可视化第一个图
        except Exception as e:
            logger.warning(f"首个图可视化失败，训练继续: {type(e).__name__}: {e}")

    logger.info("***************************************")

    # Step 2: 过滤样本数过少或极度不平衡的类别，并生成 y_mapped。
    dataList, keptClassLabels, filterStats = filterExtremeImbalanceClasses(
        dataList,
        imbalanceRatioThreshold=config["imbalanceRatioThreshold"],
        minCountThreshold=config["minCountThreshold"],
    )
    logger.info(f"原始标签分布: {filterStats['original_counts']}")
    logger.info(
        f"max/min={filterStats['max_count']}/{filterStats['min_count']}="
        f"{filterStats['ratio']:.4f}, 是否触发200倍规则: {filterStats['ratio_gate_triggered']}"
    )
    logger.info(f"200倍规则阈值(样本数): {filterStats['ratio_count_threshold']:.4f}")
    logger.info(f"因200倍规则删除类别: {filterStats['removed_by_ratio']}")
    logger.info(f"因<30规则删除类别: {filterStats['removed_by_min_count']}")
    logger.info(f"最终删除类别: {filterStats['removed_classes']}")
    logger.info(f"保留类别: {keptClassLabels}")
    logger.info(f"过滤后标签分布: {filterStats['kept_counts']}")
    logger.info("***************************************")

    # Step 3: 分层划分数据集，尽量保持每个集合中的类别比例一致。
    trainValDataset, testDataset = stratifiedSplit(
        dataList,
        testRatio=config["testRatio"],
        seed=config["seed"],
    )
    trainDataset, valDataset = stratifiedSplit(
        trainValDataset,
        testRatio=config["valRatioInTrainVal"],
        seed=config["seed"] + 1,
    )

    # 打印划分后的类别分布，便于发现某个集合中类别缺失的问题。
    labelCounts = Counter(getLabelValue(data.y) for data in trainValDataset)
    trainCounts = Counter(getLabelValue(data.y) for data in trainDataset)
    valCounts = Counter(getLabelValue(data.y) for data in valDataset)
    testCounts = Counter(getLabelValue(data.y) for data in testDataset)
    logger.info(f"Train 标签分布（原始）：{labelCounts}")
    logger.info(f"训练集标签分布：{trainCounts}")
    logger.info(f"验证集标签分布：{valCounts}")
    logger.info(f"测试集标签分布：{testCounts}")
    logger.info("***************************************")

    # 节点特征的统计量只由训练集拟合，验证集和测试集仅复用，避免数据泄漏。
    if config.get("standardizeNodeFeatures", True):
        config["nodeFeatureStandardizer"] = standardizeNodeFeatureDatasets(
            trainDataset,
            valDataset,
            testDataset,
            epsilon=config.get("nodeFeatureStdEpsilon", 1e-6),
        )
    else:
        config["nodeFeatureStandardizer"] = None
        logger.warning("Node feature standardization is disabled by configuration.")

    numClasses = len(keptClassLabels)
    patternFeatureDim = numClasses * 2 + 2
    portSequenceField = config.get("portSequenceField", "dstPortSequence")
    ngramLengths = config["ngramLengths"]
    scoreMode = config["scoreMode"]

    config["fusionMode"] = "conservativeDenseGate"
    config["usePatternFeature"] = True
    config["patternLambda"] = float(config.get("patternLambda", 0.2))

    appPortSet = {
        int(port)
        for port in appPortLabel.values()
        if int(port) != 0
    }
    logger.info("=== Build train-only patternLibrary for conservativeDenseGate ===")
    patternLibrary = buildClassPatternLibrary(
        trainDataset=trainDataset,
        numClasses=numClasses,
        ngramLengths=ngramLengths,
        topK=config["topK"],
        minSupport=config["minSupport"],
        appPortSet=appPortSet,
        portSequenceField=portSequenceField,
        maxTopK=config["maxTopK"],
        topKPower=config["topKPower"],
    )
    attachPatternFeatureToDataset(
        trainDataset, patternLibrary, appPortSet, ngramLengths, scoreMode,
        numClasses, "train", portSequenceField
    )
    attachPatternFeatureToDataset(
        valDataset, patternLibrary, appPortSet, ngramLengths, scoreMode,
        numClasses, "val", portSequenceField
    )
    attachPatternFeatureToDataset(
        testDataset, patternLibrary, appPortSet, ngramLengths, scoreMode,
        numClasses, "test", portSequenceField
    )
    logger.info(
        "Attached destination-port PatternFeature: portSequenceField=%s, scoreMode=%s, featureDim=%s",
        portSequenceField,
        scoreMode,
        patternFeatureDim,
    )

    # Step 4: 准备运行设备并执行最终方法的一次完整训练与评估。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pinMemory = bool(config["pinMemory"] and device.type == "cuda")
    numWorkers = getNumWorkers(config)
    config["patternFeatureDim"] = patternFeatureDim
    runDir = makeRunDir(config["resultsDir"])
    runSingleExperiment(
        config,
        trainDataset,
        valDataset,
        testDataset,
        keptClassLabels,
        filterStats,
        device,
        numWorkers,
        pinMemory,
        runDir,
    )
    logger.info("训练完成")

if __name__ == "__main__":
    main()
