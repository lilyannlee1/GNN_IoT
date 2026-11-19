"""
Pattern-only 端口序列模式挖掘工具。

程序设计总览:
1. 每张图在构图阶段保存一个端口序列字段，例如:
   - data.srcPortSequence = [50000, 443, 1883]
   - data.dstPortSequence = [443, 443, 53]
2. 本文件根据 portSequenceField 读取某个端口序列字段。
3. 端口序列先被编码成 token 序列，例如:
   [50000, 443, 1883] -> ["DYNAMIC_1", "443", "1883"]
4. token 序列再被切成连续 n-gram，例如 ngramLengths=[2]:
   ["DYNAMIC_1", "443", "1883"]
   -> ("DYNAMIC_1", "443"), ("443", "1883")
5. 训练阶段只用 trainDataset 统计每个类别的高频 n-gram，得到 patternLibrary。
6. 评估阶段对每张图计算 patternScore，直接用 argmax(patternScore) 做分类。

核心数据结构:
patternLibrary = {
    patternTuple: {
        classIndex: count,
        ...
    },
    ...
}

例子:
patternLibrary = {
    ("DYNAMIC_1", "443"): {0: 12, 2: 3},
    ("1883",): {1: 8},
}

含义:
("DYNAMIC_1", "443") 是一个被保留下来的频繁模式；
它在训练集中 class 0 出现过 12 次，在 class 2 出现过 3 次。
当前加权评分中，count 会用于增强训练集中更常见的类别模式。
"""

from collections import Counter
import logging
import math
from typing import Dict, Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline as makePipeline
from sklearn.preprocessing import StandardScaler

# 复用主训练脚本中的 logging 配置，避免本工具文件单独配置日志。
moduleLogger = logging.getLogger(__name__)


def encodePortToken(portValue, appPortSet) -> str:
    """
    功能:
        将一个端口号编码成字符串 token。

    输入:
        srcPort:
            单个端口号，可以是 int、float、字符串数字等。
            例子: 443, "1883", 50000。
        appPortSet:
            应用端口集合，来自 APP_PORT_LABEL.values() 并排除 0。
            例子: {80, 443, 1883, 8883}。

    输出:
        str:
            编码后的端口 token。
            例子:
            - 443 且 443 在 appPortSet 中 -> "443"
            - 0 -> "UNKNOWN_PORT"
            - 999 -> "WELL_KNOWN_OTHER"
            - 1500 -> "REGISTERED_1"
            - 3000 -> "REGISTERED_2"
            - 40000 -> "REGISTERED_6"
            - 50000 -> "DYNAMIC_1"

    映射优先级与分片规则:
        1. 无效端口（<= 0 或 > 65535）映射为 UNKNOWN_PORT；
        2. appPortSet 中的应用端口保留具体端口号字符串；
        3. 其余端口按以下闭区间映射：
           - [1, 1023]       -> WELL_KNOWN_OTHER
           - [1024, 2047]    -> REGISTERED_1
           - [2048, 4095]    -> REGISTERED_2
           - [4096, 8191]    -> REGISTERED_3
           - [8192, 16383]   -> REGISTERED_4
           - [16384, 32767]  -> REGISTERED_5
           - [32768, 49151]  -> REGISTERED_6
           - [49152, 53247]  -> DYNAMIC_1
           - [53248, 57343]  -> DYNAMIC_2
           - [57344, 61439]  -> DYNAMIC_3
           - [61440, 65535]  -> DYNAMIC_4

    关键变量:
        portFloat:
            先把输入解析为浮点数，以便拒绝 1.5 这类非整数端口。
            例如 srcPort="443.0" 时，portFloat=443.0。
        portInt:
            仅在 portFloat 有限且为整数时转换得到。
            例如 portFloat=443.0 时，portInt=443。

    注意:
        APP_PORT_LABEL["unknown"] = 0 不应被保留为真实端口 token。
        因为 0 会先被 portInt <= 0 捕获，统一映射为 UNKNOWN_PORT。
    """
    if isinstance(portValue, bool):
        return "UNKNOWN_PORT"
    try:
        portFloat = float(portValue)
    except (TypeError, ValueError, OverflowError):
        return "UNKNOWN_PORT"
    if not math.isfinite(portFloat) or not portFloat.is_integer():
        return "UNKNOWN_PORT"
    portInt = int(portFloat)

    if portInt <= 0 or portInt > 65535:
        return "UNKNOWN_PORT"
    if portInt in appPortSet:
        return str(portInt)
    if 1 <= portInt <= 1023:
        return "WELL_KNOWN_OTHER"
    if 1024 <= portInt <= 2047:
        return "REGISTERED_1"
    if 2048 <= portInt <= 4095:
        return "REGISTERED_2"
    if 4096 <= portInt <= 8191:
        return "REGISTERED_3"
    if 8192 <= portInt <= 16383:
        return "REGISTERED_4"
    if 16384 <= portInt <= 32767:
        return "REGISTERED_5"
    if 32768 <= portInt <= 49151:
        return "REGISTERED_6"
    if 49152 <= portInt <= 53247:
        return "DYNAMIC_1"
    if 53248 <= portInt <= 57343:
        return "DYNAMIC_2"
    if 57344 <= portInt <= 61439:
        return "DYNAMIC_3"
    return "DYNAMIC_4"


def encodePortSequence(portSequence, appPortSet) -> List[str]:
    """
    功能:
        将一张图中的端口序列逐个编码成 token 序列。

    输入:
        srcPortSequence:
            一张图的端口序列。历史命名叫 srcPortSequence，但也可以传入 dstPortSequence。
            例子: [50000, 443, 1883]。
        appPortSet:
            应用端口集合。例子: {80, 443, 1883}。

    输出:
        List[str]:
            编码后的 token 序列。
            例子:
            [50000, 443, 1883]
            -> ["DYNAMIC_1", "443", "1883"]

    关键变量:
        srcPort:
            列表推导式中当前正在编码的单个端口。
    """
    if portSequence is None:
        return []
    return [encodePortToken(portValue, appPortSet) for portValue in portSequence]


def extractNgrams(tokenSequence: Iterable[str], ngramLengths) -> Counter:
    """
    功能:
        从 token 序列中提取连续 n-gram，并统计每种 n-gram 出现次数。

    输入:
        tokenSequence:
            已编码的端口 token 序列。
            例子: ["DYNAMIC_1", "443", "1883"]。
        ngramLengths:
            要提取的 n-gram 长度列表。
            例子: [1, 2, 3] 或 [2, 3]。

    输出:
        Counter:
            key 是 patternTuple，value 是出现次数。
            例子:
            tokenSequence=["A", "B", "A"], ngramLengths=[2]
            -> Counter({("A", "B"): 1, ("B", "A"): 1})

    关键变量:
        tokens:
            将输入序列转成 list，方便按下标切片。
        ngramCounter:
            统计 n-gram 出现次数的计数器。
        ngramLength:
            当前正在提取的 n 值。
        startIndex:
            当前 n-gram 在 tokens 中的起始位置。
        patternTuple:
            一个具体的连续 n-gram。
            例子: ("DYNAMIC_1", "443")。
    """
    tokens = list(tokenSequence)
    ngramCounter = Counter()
    for ngramLength in ngramLengths:
        ngramLength = int(ngramLength)
        if ngramLength <= 0 or len(tokens) < ngramLength:
            continue
        for startIndex in range(len(tokens) - ngramLength + 1):
            patternTuple = tuple(tokens[startIndex:startIndex + ngramLength])
            ngramCounter[patternTuple] += 1
    return ngramCounter


def buildClassPatternLibrary(
    trainDataset,
    numClasses: int,
    ngramLengths,
    topK: int,
    minSupport: int,
    appPortSet,
    portSequenceField: str = "srcPortSequence",
    maxTopK: int = None,
    topKPower: float = 0.5,
) -> Dict[tuple, Dict[int, int]]:
    """
    功能:
        只使用训练集构建每个类别的频繁端口序列模式库。

    输入:
        trainDataset:
            训练集 Data 列表。只能传训练集，不能传 val/test/full dataList。
            每个 data 至少需要:
            - data.y_mapped: 过滤后连续类别索引，例如 0, 1, 2。
            - data.<portSequenceField>: 端口序列，例如 data.dstPortSequence。
        numClasses:
            类别数。例子: keptClassLabels=[0,1,2,4] 时 numClasses=4。
        ngramLengths:
            n-gram 长度配置。例子: [2, 3]。
        topK:
            每个类别最多保留多少个高频 pattern。例子: 50。
        minSupport:
            pattern 在某类别中至少出现多少次才保留。例子: 2。
        appPortSet:
            应用端口集合。
        portSequenceField:
            从 data 中读取哪个端口序列字段。
            例子: "srcPortSequence" 或 "dstPortSequence"。

    输出:
        Dict[tuple, Dict[int, int]]:
            patternLibrary，结构为:
            {
                patternTuple: {
                    classIndex: count
                }
            }

            例子:
            {
                ("DYNAMIC_1", "443"): {0: 12, 2: 3},
                ("1883",): {1: 8}
            }

    关键变量:
        classNgramCounters:
            按类别保存 n-gram 计数的列表。
            classNgramCounters[0] 是 class 0 的 Counter。
        data:
            trainDataset 中的一张图。
        labelValue:
            data.y_mapped，可能是 Tensor，也可能是 int。
        classIndex:
            内部连续类别索引。例子: 原始标签 4 可能被映射为 classIndex=3。
        portSequence:
            当前图读取到的端口序列。
            例子: data.dstPortSequence = [443, 443, 53]。
        tokenSequence:
            portSequence 编码后的 token 序列。
        ngramCounter:
            当前图自己的 n-gram 计数。
        filteredPatterns:
            某个类别经过 minSupport 过滤后的候选 pattern。
        patternLibrary:
            最终反转后的模式库，便于评估时用 graphPattern 直接查命中类别。
    """
    classNgramCounters = [Counter() for _ in range(numClasses)]
    trainClassCount = Counter()
    baseTopK = max(0, int(topK))
    maxTopK = baseTopK if maxTopK is None else max(baseTopK, int(maxTopK))
    topKPower = float(topKPower)

    for data in trainDataset:
        if not hasattr(data, portSequenceField):
            raise ValueError(
                f"Data is missing {portSequenceField}. Please delete the old processed directory "
                "and rebuild graphs before running Pattern-only."
            )
        if not hasattr(data, "y_mapped"):
            raise ValueError("Data is missing y_mapped. Run class filtering before building patternLibrary.")

        labelValue = data.y_mapped
        if hasattr(labelValue, "view"):
            classIndex = int(labelValue.view(-1)[0].item())
        else:
            classIndex = int(labelValue)
        if classIndex < 0 or classIndex >= numClasses:
            raise ValueError(f"classIndex {classIndex} is outside numClasses={numClasses}.")
        trainClassCount[classIndex] += 1

        portSequence = getattr(data, portSequenceField)
        tokenSequence = encodePortSequence(portSequence, appPortSet)
        ngramCounter = extractNgrams(tokenSequence, ngramLengths)
        classNgramCounters[classIndex].update(ngramCounter)

    if not trainClassCount:
        raise ValueError("trainDataset is empty, cannot build patternLibrary.")
    maxClassCount = max(trainClassCount.values())
    classTopKByClass = {}
    for classIndex in range(numClasses):
        classCount = trainClassCount.get(classIndex, 0)
        if classCount <= 0:
            classTopKByClass[classIndex] = baseTopK
        else:
            classTopK = int(baseTopK * ((float(maxClassCount) / float(classCount)) ** topKPower))
            classTopKByClass[classIndex] = max(baseTopK, min(classTopK, maxTopK))
    moduleLogger.debug("[Pattern-only] trainClassCount=%s", dict(trainClassCount))
    moduleLogger.debug("[Pattern-only] topKPower=%s", topKPower)
    moduleLogger.debug("[Pattern-only] classTopKByClass=%s", classTopKByClass)

    patternLibrary = {}
    for classIndex, classNgramCounter in enumerate(classNgramCounters):
        # 先在每个类别内部过滤和排序，保证 topK/minSupport 是按类别生效。
        filteredPatterns = [
            (patternTuple, count)
            for patternTuple, count in classNgramCounter.items()
            if count >= minSupport
        ]
        filteredPatterns.sort(key=lambda item: (-item[1], item[0]))
        for patternTuple, count in filteredPatterns[:classTopKByClass[classIndex]]:
            patternLibrary.setdefault(patternTuple, {})[classIndex] = int(count)

    return patternLibrary


def computePatternScore(
    portSequence,
    patternLibrary: Dict[tuple, Dict[int, int]],
    appPortSet,
    ngramLengths,
    scoreMode: str,
    numClasses: int,
) -> List[float]:
    """
    功能:
        计算一张图对每个类别模式库的匹配分数 patternScore。

    输入:
        srcPortSequence:
            一张图的端口序列。历史命名叫 srcPortSequence，但可以传入 dstPortSequence。
            例子: [50000, 443, 1883]。
        patternLibrary:
            训练集构建出的频繁模式库。
        appPortSet:
            应用端口集合。
        ngramLengths:
            n-gram 长度配置。
        scoreMode:
            当前支持 "weightedHitRatio" 和 "tfIdfHitRatio"。
        numClasses:
            类别数，也就是 patternScore 的长度。

    输出:
        List[float]:
            patternScore，长度为 numClasses。
            patternScore[i] 表示当前图与 class i 的模式匹配比例。

    weightedHitRatio 定义:
        score[classIndex] =
            sum(graphCount * log(1 + trainCount))
            /
            当前图全部 n-gram 出现次数

    tfIdfHitRatio 定义:
        idf = log((numClasses + 1) / (classCount + 1)) + 1
        score[classIndex] =
            sum(graphCount * log(1 + trainCount) * idf)
            /
            当前图全部 n-gram 出现次数

    例子:
        graphNgramCounter = {
            ("DYNAMIC_1", "443"): 2,
            ("1883",): 1,
            ("443",): 1
        }
        patternLibrary = {
            ("DYNAMIC_1", "443"): {0: 12, 2: 3},
            ("1883",): {1: 8}
        }

        totalNgramCount = 4
        weightedHitRatio 中 class 0 的分数为:
        2 * log(1 + 12) / 4

    关键变量:
        tokenSequence:
            端口序列编码后的 token 序列。
        ngramCounter:
            当前图的 n-gram 计数。
        totalNgramCount:
            当前图全部 n-gram 出现次数总和，是归一化分母。
        patternScore:
            每个类别的累计匹配分数。
        graphPattern:
            当前图中的某个 n-gram。
        graphCount:
            graphPattern 在当前图中出现的次数。
        classValueDict:
            patternLibrary[graphPattern]，表示该 pattern 属于哪些类别。
            在 weightedHitRatio 中 trainCount 会通过 log(1 + trainCount) 加权；
            在 tfIdfHitRatio 中还会乘以 idf，降低共享 pattern 权重。
    """
    if scoreMode not in ("weightedHitRatio", "tfIdfHitRatio"):
        raise ValueError(f"Unsupported scoreMode: {scoreMode}")

    tokenSequence = encodePortSequence(portSequence, appPortSet)
    ngramCounter = extractNgrams(tokenSequence, ngramLengths)
    totalNgramCount = sum(ngramCounter.values())
    if totalNgramCount == 0:
        return [0.0] * numClasses

    patternScore = [0.0] * numClasses
    useTfIdf = scoreMode == "tfIdfHitRatio"
    for graphPattern, graphCount in ngramCounter.items():
        classValueDict = patternLibrary.get(graphPattern)
        if not classValueDict:
            continue
        graphCountFloat = float(graphCount)
        idf = 1.0
        if useTfIdf:
            classCount = len(classValueDict)
            idf = math.log((float(numClasses) + 1.0) / (float(classCount) + 1.0)) + 1.0
        for classIndex, trainCount in classValueDict.items():
            if 0 <= classIndex < numClasses:
                trainWeight = math.log1p(float(trainCount))
                patternScore[classIndex] += graphCountFloat * trainWeight * idf

    patternScore = [score / float(totalNgramCount) for score in patternScore]

    return patternScore


def computePatternFeature(
    portSequence,
    patternLibrary: Dict[tuple, Dict[int, int]],
    appPortSet,
    ngramLengths,
    scoreMode: str,
    numClasses: int,
) -> Dict[str, object]:
    """
    为一张图生成轻量分类器输入特征。

    patternFeature =
        patternScore
        + uniquePatternHitRatioByClass
        + [sharedPatternHitRatio, zeroHitFlag]
    """
    patternScore = computePatternScore(
        portSequence,
        patternLibrary,
        appPortSet,
        ngramLengths,
        scoreMode,
        numClasses,
    )
    tokenSequence = encodePortSequence(portSequence, appPortSet)
    graphNgramCounter = extractNgrams(tokenSequence, ngramLengths)

    totalPatternHitCount = 0
    sharedPatternHitCount = 0
    uniquePatternHitCountByClass = [0 for _ in range(numClasses)]
    for graphPattern, graphCount in graphNgramCounter.items():
        classValueDict = patternLibrary.get(graphPattern)
        if not classValueDict:
            continue
        validClassIndexes = [
            classIndex
            for classIndex in classValueDict.keys()
            if 0 <= classIndex < numClasses
        ]
        if not validClassIndexes:
            continue
        graphCount = int(graphCount)
        totalPatternHitCount += graphCount
        if len(validClassIndexes) >= 2:
            sharedPatternHitCount += graphCount
        elif len(validClassIndexes) == 1:
            uniquePatternHitCountByClass[validClassIndexes[0]] += graphCount

    uniquePatternHitRatioByClass = [
        float(uniquePatternHitCount) / float(totalPatternHitCount)
        if totalPatternHitCount > 0 else 0.0
        for uniquePatternHitCount in uniquePatternHitCountByClass
    ]
    sharedPatternHitRatio = (
        float(sharedPatternHitCount) / float(totalPatternHitCount)
        if totalPatternHitCount > 0 else 0.0
    )
    zeroHitFlag = 1.0 if all(score == 0.0 for score in patternScore) else 0.0
    patternFeature = (
        list(patternScore)
        + uniquePatternHitRatioByClass
        + [sharedPatternHitRatio, zeroHitFlag]
    )

    return {
        "patternScore": patternScore,
        "uniquePatternHitRatioByClass": uniquePatternHitRatioByClass,
        "sharedPatternHitRatio": sharedPatternHitRatio,
        "zeroHitFlag": zeroHitFlag,
        "patternFeature": patternFeature,
        "totalPatternHitCount": int(totalPatternHitCount),
        "sharedPatternHitCount": int(sharedPatternHitCount),
    }


def buildPatternFeatureDataset(
    dataset,
    patternLibrary: Dict[tuple, Dict[int, int]],
    appPortSet,
    ngramLengths,
    scoreMode: str,
    numClasses: int,
    portSequenceField: str = "srcPortSequence",
) -> Dict[str, object]:
    """对一个 split 生成 LogisticRegression 使用的 patternFeature 矩阵。"""
    patternFeatureRows = []
    patternLabelValues = []
    featureDim = numClasses * 2 + 2

    for data in dataset:
        if not hasattr(data, portSequenceField):
            raise ValueError(
                f"Data is missing {portSequenceField}. Please delete the old processed directory "
                "and rebuild graphs before running Pattern-only."
            )
        if not hasattr(data, "y_mapped"):
            raise ValueError("Data is missing y_mapped. Run class filtering before building patternFeature.")

        portSequence = getattr(data, portSequenceField)
        patternFeatureDict = computePatternFeature(
            portSequence,
            patternLibrary,
            appPortSet,
            ngramLengths,
            scoreMode,
            numClasses,
        )
        patternFeature = patternFeatureDict["patternFeature"]
        if len(patternFeature) != featureDim:
            raise ValueError(
                f"patternFeature length mismatch: got {len(patternFeature)}, expected {featureDim}."
            )

        labelValue = data.y_mapped
        if hasattr(labelValue, "view"):
            classIndex = int(labelValue.view(-1)[0].item())
        else:
            classIndex = int(labelValue)
        patternFeatureRows.append(patternFeature)
        patternLabelValues.append(classIndex)

    patternFeatureMatrix = (
        np.asarray(patternFeatureRows, dtype=np.float32)
        if patternFeatureRows else np.empty((0, featureDim), dtype=np.float32)
    )
    patternLabelArray = np.asarray(patternLabelValues, dtype=np.int64)
    return {
        "patternFeatureMatrix": patternFeatureMatrix,
        "patternLabelArray": patternLabelArray,
        "featureDim": int(featureDim),
        "numSamples": int(len(patternLabelValues)),
    }


def trainPatternFeatureClassifier(
    trainPatternFeatureMatrix,
    trainPatternLabelArray,
    logisticMaxIter: int = 2000,
    logisticClassWeight="balanced",
    randomSeed: int = 42,
):
    """只在 trainPatternFeature 上训练 LogisticRegression。"""
    if len(trainPatternLabelArray) == 0:
        raise ValueError("trainPatternLabelArray is empty, cannot train PatternFeatureClassifier.")
    patternFeatureClassifier = makePipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(logisticMaxIter),
            class_weight=logisticClassWeight,
            random_state=int(randomSeed),
        ),
    )
    patternFeatureClassifier.fit(trainPatternFeatureMatrix, trainPatternLabelArray)
    return patternFeatureClassifier


def evaluatePatternFeatureClassifier(
    patternFeatureClassifier,
    patternFeatureMatrix,
    patternLabelArray,
    numClasses: int,
    splitName: str,
) -> Dict[str, object]:
    """评估 PatternFeatureClassifier，并输出与 baseline 一致的核心指标。"""
    if len(patternLabelArray) == 0:
        patternPredArray = np.asarray([], dtype=np.int64)
    else:
        patternPredArray = np.asarray(patternFeatureClassifier.predict(patternFeatureMatrix), dtype=np.int64)

    confusionMatrix = np.zeros((numClasses, numClasses), dtype=np.int64)
    for trueIndex, predIndex in zip(patternLabelArray, patternPredArray):
        trueIndex = int(trueIndex)
        predIndex = int(predIndex)
        if 0 <= trueIndex < numClasses and 0 <= predIndex < numClasses:
            confusionMatrix[trueIndex, predIndex] += 1

    accuracy = (
        float((patternPredArray == patternLabelArray).mean())
        if len(patternLabelArray) > 0 else 0.0
    )
    f1Scores = []
    supports = []
    for classIndex in range(numClasses):
        truePositive = float(confusionMatrix[classIndex, classIndex])
        falsePositive = float(confusionMatrix[:, classIndex].sum() - confusionMatrix[classIndex, classIndex])
        falseNegative = float(confusionMatrix[classIndex, :].sum() - confusionMatrix[classIndex, classIndex])
        support = float(confusionMatrix[classIndex, :].sum())
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1Score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1Scores.append(f1Score)
        supports.append(support)
    macroF1 = float(np.mean(f1Scores)) if f1Scores else 0.0
    totalSupport = float(sum(supports))
    weightedF1 = (
        float(sum(f1Score * support for f1Score, support in zip(f1Scores, supports)) / totalSupport)
        if totalSupport > 0 else 0.0
    )
    predLabelCounter = Counter(int(predIndex) for predIndex in patternPredArray.tolist())
    predLabelCounter = {
        classIndex: int(predLabelCounter.get(classIndex, 0))
        for classIndex in range(numClasses)
    }
    moduleLogger.debug("[PatternFeatureClassifier][%s] predLabelCounter=%s", splitName, predLabelCounter)

    return {
        "splitName": splitName,
        "accuracy": accuracy,
        "macroF1": float(macroF1),
        "weightedF1": float(weightedF1),
        "confusionMatrix": confusionMatrix,
        "predLabelCounter": predLabelCounter,
        "numSamples": int(len(patternLabelArray)),
    }


def getPatternClassCountDistribution(patternLibrary: Dict[tuple, Dict[int, int]], numClasses: int) -> Dict[int, int]:
    patternClassCountDistribution = Counter()
    for classValueDict in patternLibrary.values():
        classCount = sum(1 for classIndex in classValueDict.keys() if 0 <= classIndex < numClasses)
        if classCount > 0:
            patternClassCountDistribution[classCount] += 1
    return {
        classCount: int(patternClassCountDistribution[classCount])
        for classCount in sorted(patternClassCountDistribution.keys())
    }


def getTopSharedPatterns(
    patternLibrary: Dict[tuple, Dict[int, int]],
    numClasses: int,
    topSharedPatternCount: int,
) -> List[Dict[str, object]]:
    sharedPatternRows = []
    for patternTuple, classValueDict in patternLibrary.items():
        classCounts = {
            int(classIndex): int(count)
            for classIndex, count in classValueDict.items()
            if 0 <= classIndex < numClasses
        }
        classCount = len(classCounts)
        if classCount < 2:
            continue
        totalTrainCount = sum(classCounts.values())
        sharedPatternRows.append({
            "patternTuple": patternTuple,
            "classIndexes": sorted(classCounts.keys()),
            "classCounts": classCounts,
            "classCount": classCount,
            "totalTrainCount": int(totalTrainCount),
        })
    sharedPatternRows.sort(
        key=lambda row: (-row["classCount"], -row["totalTrainCount"], str(row["patternTuple"]))
    )
    return sharedPatternRows[:max(0, int(topSharedPatternCount))]


def analyzeConfusionPairSharedPatterns(
    patternLibrary: Dict[tuple, Dict[int, int]],
    confusionPairList,
    numClasses: int,
) -> Dict[str, Dict[str, float]]:
    classPatternSets = [set() for _ in range(numClasses)]
    patternClassIndexes = {}
    for patternTuple, classValueDict in patternLibrary.items():
        validClassIndexes = {
            classIndex
            for classIndex in classValueDict.keys()
            if 0 <= classIndex < numClasses
        }
        patternClassIndexes[patternTuple] = validClassIndexes
        for classIndex in validClassIndexes:
            classPatternSets[classIndex].add(patternTuple)

    pairAnalysis = {}
    for classA, classB in confusionPairList or []:
        classA = int(classA)
        classB = int(classB)
        pairKey = f"{classA}-{classB}"
        if not (0 <= classA < numClasses and 0 <= classB < numClasses):
            pairAnalysis[pairKey] = {"error": "classIndex outside numClasses"}
            continue
        classAPatterns = classPatternSets[classA]
        classBPatterns = classPatternSets[classB]
        pairSharedPatterns = classAPatterns & classBPatterns
        classAUniquePatterns = {
            patternTuple
            for patternTuple in classAPatterns
            if patternClassIndexes.get(patternTuple) == {classA}
        }
        classBUniquePatterns = {
            patternTuple
            for patternTuple in classBPatterns
            if patternClassIndexes.get(patternTuple) == {classB}
        }
        pairPatternUnionCount = len(classAPatterns | classBPatterns)
        pairSharedPatternRatio = (
            float(len(pairSharedPatterns)) / float(pairPatternUnionCount)
            if pairPatternUnionCount > 0 else 0.0
        )
        pairAnalysis[pairKey] = {
            "sharedPatternCount": int(len(pairSharedPatterns)),
            "classAUniquePatternCount": int(len(classAUniquePatterns)),
            "classBUniquePatternCount": int(len(classBUniquePatterns)),
            "pairSharedPatternCount": int(len(pairSharedPatterns)),
            "pairSharedPatternRatio": pairSharedPatternRatio,
        }
    return pairAnalysis


def summarizePatternLibrary(patternLibrary: Dict[tuple, Dict[int, int]], numClasses: int) -> Dict[str, object]:
    """
    功能:
        统计 patternLibrary 中有多少模式是类别独有的，有多少模式被多个类别共享。

    输入:
        patternLibrary:
            训练集构建出的模式库。
        numClasses:
            类别数。

    输出:
        Dict[str, object]:
            {
                "totalPatternCount": 总 pattern 数,
                "sharedPatternCount": 被两个及以上类别共享的 pattern 数,
                "uniquePatternCount": 只属于一个类别的 pattern 数,
                "retainedPatternCountByClass": 每类保留下来的 pattern 数,
                "uniquePatternCountByClass": 每类独有 pattern 数,
            }

    例子:
        patternLibrary = {
            ("A",): {0: 5, 1: 2},
            ("B",): {1: 8}
        }
        则 ("A",) 是 shared pattern，("B",) 是 class 1 的 unique pattern。

    关键变量:
        retainedPatternCountByClass:
            每个类别被保留下来的 pattern 数，包含共享 pattern。
        uniquePatternCountByClass:
            每个类别独有的 pattern 数。
        sharedPatternCount:
            出现在两个及以上类别中的 pattern 数。
        uniquePatternCount:
            只出现在一个类别中的 pattern 数。
        validClassIndexes:
            某个 pattern 对应的合法类别索引列表。
    """
    retainedPatternCountByClass = {classIndex: 0 for classIndex in range(numClasses)}
    uniquePatternCountByClass = {classIndex: 0 for classIndex in range(numClasses)}
    sharedPatternCount = 0
    uniquePatternCount = 0

    for classValueDict in patternLibrary.values():
        validClassIndexes = [
            classIndex
            for classIndex in classValueDict.keys()
            if 0 <= classIndex < numClasses
        ]
        for classIndex in validClassIndexes:
            retainedPatternCountByClass[classIndex] += 1
        if len(validClassIndexes) >= 2:
            sharedPatternCount += 1
        elif len(validClassIndexes) == 1:
            uniquePatternCount += 1
            uniquePatternCountByClass[validClassIndexes[0]] += 1

    return {
        "totalPatternCount": len(patternLibrary),
        "sharedPatternCount": sharedPatternCount,
        "sharedPatternRatio": (
            float(sharedPatternCount) / float(len(patternLibrary))
            if len(patternLibrary) > 0 else 0.0
        ),
        "uniquePatternCount": uniquePatternCount,
        "patternClassCountDistribution": getPatternClassCountDistribution(patternLibrary, numClasses),
        "retainedPatternCountByClass": retainedPatternCountByClass,
        "uniquePatternCountByClass": uniquePatternCountByClass,
    }


def evaluatePatternOnly(
    dataset,
    patternLibrary: Dict[tuple, Dict[int, int]],
    appPortSet,
    ngramLengths,
    scoreMode: str,
    numClasses: int,
    splitName: str,
    portSequenceField: str = "srcPortSequence",
    enableSharedPatternDiagnostics: bool = False,
) -> Dict[str, object]:
    """
    功能:
        对一个数据集执行 Pattern-only 分类评估。

    输入:
        dataset:
            待评估的数据集，可以是 trainDataset、valDataset 或 testDataset。
        patternLibrary:
            只能由 trainDataset 构建出的模式库。
        appPortSet:
            应用端口集合。
        ngramLengths:
            n-gram 长度配置。
        scoreMode:
            当前支持 "weightedHitRatio" 和 "tfIdfHitRatio"。
        numClasses:
            类别数。
        splitName:
            当前评估集名称，用于日志显示。
            例子: "train", "val", "test"。
        portSequenceField:
            读取 data 的哪个端口序列字段。
            例子: "srcPortSequence" 或 "dstPortSequence"。

    输出:
        Dict[str, object]:
            包含:
            - accuracy: 准确率
            - macroF1: 宏平均 F1
            - weightedF1: 加权平均 F1
            - confusionMatrix: 混淆矩阵，形状 [numClasses, numClasses]
            - predLabelCounter: 每个类别被预测了多少次
            - tieMaxCount/tieMaxRate: 最大分并列样本数量和比例
            - avgPatternScoreByTrueClass: 每个真实类别的平均 patternScore
            - patternLibrarySummary: 模式库共享/独有模式统计
            - zeroHitCount/zeroHitRate: 全 0 patternScore 样本数量和比例

    分类规则:
        对每张图:
        1. 计算 patternScore
        2. predIndex = argmax(patternScore)
        3. 与 trueIndex = data.y_mapped 比较

    关键变量:
        allPreds:
            保存所有预测类别索引。
        allLabels:
            保存所有真实类别索引。
        predLabelCounter:
            统计每个类别被预测的次数。
        zeroHitCount:
            patternScore 全为 0 的样本数。
        tieMaxCount:
            patternScore 中最大值被两个及以上类别并列获得的样本数。
        scoreSumByTrueClass:
            按真实类别累计 patternScore，用于计算平均分数向量。
        scoreCountByTrueClass:
            每个真实类别有多少样本。
        confusionMatrix:
            行是真实类别，列是预测类别。
        f1Scores/supports:
            逐类 F1 和样本数，用于计算 macroF1 与 weightedF1。

    例子:
        如果某样本 patternScore=[0.5, 0.25, 0.5]，
        np.argmax 会返回 0，也就是并列最大时选择最小类别索引。
        这种样本会被 tieMaxCount 统计到。
    """
    allPreds = []
    allLabels = []
    predLabelCounter = Counter()
    zeroHitCount = 0
    tieMaxCount = 0
    scoreSumByTrueClass = {
        classIndex: np.zeros(numClasses, dtype=np.float64)
        for classIndex in range(numClasses)
    }
    scoreCountByTrueClass = Counter()
    sharedPatternHitCount = 0
    totalPatternHitCount = 0
    uniquePatternHitCountByTrueClass = Counter()
    totalPatternHitCountByTrueClass = Counter()
    patternClassIndexes = {}
    if enableSharedPatternDiagnostics:
        patternClassIndexes = {
            patternTuple: [
                classIndex
                for classIndex in classValueDict.keys()
                if 0 <= classIndex < numClasses
            ]
            for patternTuple, classValueDict in patternLibrary.items()
        }

    for data in dataset:
        if not hasattr(data, portSequenceField):
            raise ValueError(
                f"Data is missing {portSequenceField}. Please delete the old processed directory "
                "and rebuild graphs before running Pattern-only."
            )
        if not hasattr(data, "y_mapped"):
            raise ValueError("Data is missing y_mapped. Run class filtering before Pattern-only evaluation.")

        portSequence = getattr(data, portSequenceField)
        patternScore = computePatternScore(
            portSequence,
            patternLibrary,
            appPortSet,
            ngramLengths,
            scoreMode,
            numClasses,
        )
        if len(patternScore) != numClasses:
            raise ValueError(
                f"patternScore length mismatch: got {len(patternScore)}, expected {numClasses}."
            )
        scoreArray = np.asarray(patternScore, dtype=np.float32)
        if all(score == 0.0 for score in patternScore):
            zeroHitCount += 1
        maxScore = float(scoreArray.max()) if scoreArray.size > 0 else 0.0
        if int(np.isclose(scoreArray, maxScore).sum()) >= 2:
            tieMaxCount += 1

        predIndex = int(np.argmax(scoreArray))
        labelValue = data.y_mapped
        if hasattr(labelValue, "view"):
            trueIndex = int(labelValue.view(-1)[0].item())
        else:
            trueIndex = int(labelValue)

        if enableSharedPatternDiagnostics and 0 <= trueIndex < numClasses:
            tokenSequence = encodePortSequence(portSequence, appPortSet)
            graphNgramCounter = extractNgrams(tokenSequence, ngramLengths)
            for graphPattern, graphCount in graphNgramCounter.items():
                classIndexes = patternClassIndexes.get(graphPattern)
                if not classIndexes:
                    continue
                graphCount = int(graphCount)
                totalPatternHitCount += graphCount
                totalPatternHitCountByTrueClass[trueIndex] += graphCount
                if len(classIndexes) >= 2:
                    sharedPatternHitCount += graphCount
                elif len(classIndexes) == 1 and classIndexes[0] == trueIndex:
                    uniquePatternHitCountByTrueClass[trueIndex] += graphCount

        allPreds.append(predIndex)
        allLabels.append(trueIndex)
        predLabelCounter[predIndex] += 1
        if 0 <= trueIndex < numClasses:
            scoreSumByTrueClass[trueIndex] += np.asarray(patternScore, dtype=np.float64)
            scoreCountByTrueClass[trueIndex] += 1

    confusionMatrix = np.zeros((numClasses, numClasses), dtype=np.int64)
    for trueIndex, predIndex in zip(allLabels, allPreds):
        if 0 <= trueIndex < numClasses and 0 <= predIndex < numClasses:
            confusionMatrix[trueIndex, predIndex] += 1
    accuracy = float((np.asarray(allPreds) == np.asarray(allLabels)).mean()) if allLabels else 0.0
    f1Scores = []
    supports = []
    for classIndex in range(numClasses):
        truePositive = float(confusionMatrix[classIndex, classIndex])
        falsePositive = float(confusionMatrix[:, classIndex].sum() - confusionMatrix[classIndex, classIndex])
        falseNegative = float(confusionMatrix[classIndex, :].sum() - confusionMatrix[classIndex, classIndex])
        support = float(confusionMatrix[classIndex, :].sum())
        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1Score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1Scores.append(f1Score)
        supports.append(support)
    macroF1 = float(np.mean(f1Scores)) if f1Scores else 0.0
    totalSupport = float(sum(supports))
    weightedF1 = (
        float(sum(f1Score * support for f1Score, support in zip(f1Scores, supports)) / totalSupport)
        if totalSupport > 0 else 0.0
    )
    zeroHitRate = float(zeroHitCount) / float(len(allLabels)) if allLabels else 0.0
    tieMaxRate = float(tieMaxCount) / float(len(allLabels)) if allLabels else 0.0
    predLabelCounter = {
        classIndex: int(predLabelCounter.get(classIndex, 0))
        for classIndex in range(numClasses)
    }
    avgPatternScoreByTrueClass = {}
    for classIndex in range(numClasses):
        classSampleCount = scoreCountByTrueClass.get(classIndex, 0)
        if classSampleCount > 0:
            avgScore = scoreSumByTrueClass[classIndex] / float(classSampleCount)
            avgPatternScoreByTrueClass[classIndex] = avgScore.tolist()
        else:
            avgPatternScoreByTrueClass[classIndex] = [0.0] * numClasses
    patternLibrarySummary = summarizePatternLibrary(patternLibrary, numClasses)
    sharedPatternHitRatio = (
        float(sharedPatternHitCount) / float(totalPatternHitCount)
        if totalPatternHitCount > 0 else 0.0
    )
    uniquePatternHitRatioByTrueClass = {}
    for classIndex in range(numClasses):
        classHitCount = totalPatternHitCountByTrueClass.get(classIndex, 0)
        uniquePatternHitRatioByTrueClass[classIndex] = (
            float(uniquePatternHitCountByTrueClass.get(classIndex, 0)) / float(classHitCount)
            if classHitCount > 0 else 0.0
        )

    moduleLogger.debug("[Pattern-only][%s] predLabelCounter=%s", splitName, predLabelCounter)
    moduleLogger.debug(
        "[Pattern-only][%s] tieMaxCount=%s | tieMaxRate=%.4f",
        splitName,
        tieMaxCount,
        tieMaxRate,
    )
    for classIndex, avgScore in avgPatternScoreByTrueClass.items():
        moduleLogger.debug("[Pattern-only][%s] trueClass=%s meanScore=%s", splitName, classIndex, avgScore)
    moduleLogger.debug("[Pattern-only][%s] patternLibrarySummary=%s", splitName, patternLibrarySummary)
    if enableSharedPatternDiagnostics:
        moduleLogger.debug(
            "[Pattern-only][%s] sharedPatternHitRatio=%.4f | sharedPatternHitCount=%s | totalPatternHitCount=%s",
            splitName,
            sharedPatternHitRatio,
            sharedPatternHitCount,
            totalPatternHitCount,
        )
        moduleLogger.debug(
            "[Pattern-only][%s] uniquePatternHitRatioByTrueClass=%s",
            splitName,
            uniquePatternHitRatioByTrueClass,
        )

    return {
        "splitName": splitName,
        "accuracy": accuracy,
        "macroF1": float(macroF1),
        "weightedF1": float(weightedF1),
        "confusionMatrix": confusionMatrix,
        "predLabelCounter": predLabelCounter,
        "tieMaxCount": int(tieMaxCount),
        "tieMaxRate": tieMaxRate,
        "avgPatternScoreByTrueClass": avgPatternScoreByTrueClass,
        "patternLibrarySummary": patternLibrarySummary,
        "sharedPatternHitRatio": sharedPatternHitRatio,
        "sharedPatternHitCount": int(sharedPatternHitCount),
        "totalPatternHitCount": int(totalPatternHitCount),
        "uniquePatternHitRatioByTrueClass": uniquePatternHitRatioByTrueClass,
        "zeroHitCount": int(zeroHitCount),
        "zeroHitRate": zeroHitRate,
        "numSamples": len(allLabels),
    }
