"""
IoT-DFIG 图构建脚本。

本文件负责把 CSV 中的 flow-level 流量记录转换为 PyTorch Geometric Data 图:
1. 每条 flow 拆成正向和反向两个节点；
2. 按设备 MAC 分组，并按固定时间窗口切分为子图；
3. 根据相邻 flow 的开始时间差和时间区间重叠关系构造边；
4. edge_attr 使用 [timeDiff, overlap] 表示开始时间差和重叠标记；
5. 保存带构图版本的 processed 缓存、data_<id>.pt 以及端口统计库。
"""

import ast
import logging
import math
import os
import pickle
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.multiprocessing
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

# 多进程保存/传递 Tensor 时使用文件系统共享策略，降低共享内存不足的概率。
torch.multiprocessing.set_sharing_strategy('file_system')

# 构图语义变更后使用新缓存目录，避免误读修复前的反向端口和节点特征。
# 保留旧 processed 目录，既不破坏历史实验，也保证新实验会重新构图。
GRAPH_CACHE_DIR_NAME = "processed_v4_time_overlap_edges"

NODE_FEATURE_NAMES = [
    "flow_duration",
    "flow_byts_s",
    "pkts_s",
    "tot_pkts",
    "pkts_len_total",
    "node_aligned_packet_log_ratio",
    "pld_len_total",
    "pld_len_max",
    "pld_len_min",
    "pld_len_mean",
    "pld_len_var",
    "pld_len_median",
    "pld_len_mode",
    "inter_arrTime_total",
    "inter_arrTime_max",
    "inter_arrTime_min",
    "inter_arrTime_mean",
    "inter_arrTime_var",
    "inter_arrTime_median",
    "inter_arrTime_mode",
    "app_pld_len_max",
    "app_pld_len_mean",
    "node_direction",
    "opposite_present",
    "paired_direction_delay_log",
    "paired_direction_delay_valid",
    "in_degree",
    "out_degree",
]

# =========================
# 1. 标签映射与常量
# =========================
# UNKNOWN_DEVICE_TYPES 用于过滤无法确定设备类型的 flow。
UNKNOWN_DEVICE_TYPES = ('', 'unknow', 'unknown', 'Unknown')

# 攻击/流量大类映射，当前文件保留该函数接口，主构图流程主要使用设备类型标签。
TRAFFIC_MULTI_TYPE_LABEL = {
    'Benign': 0,
    'DDoS': 1,
    'Brute_Force': 2,
    'DoS': 3,
    'Spoofing': 4,
    'Recon': 5,
    'Web-based': 6,
    'Mirai': 7,
}

# 二分类流量类型映射，当前主构图流程未直接使用，保留给其他实验调用。
TRAFFIC_BI_TYPE_LABEL = {
    "Benign": 0,
    "Anomaly": 1,
}

# IoT 设备类型 -> 数字类别。csv2srcGraph 会用它生成图标签 y。
DEVICE_TYPE_LABEL = {
    'Audio': 0,
    'Camera': 1,
    'Hub': 2,
    'Power_Outlet': 3,
    'Home_Automation': 4,
    'Lighting': 5,
    'NextGen': 6,
}

# 应用协议名 -> 常见端口号。当前节点特征中相关调用被注释，保留给特征扩展使用。
APP_PORT_LABEL = {
    "HTTP": 80,
    "HTTPS": 443,
    "HTTP Alternate": 8080,
    "HTTPS Alternate": 8443,
    "FTP": 21,
    "FTPS": 990,
    "SFTP": 22,
    "SSH": 22,
    "Telnet": 23,
    "SMTP": 25,
    "SMTPS": 465,
    "IMAP": 143,
    "IMAPS": 993,
    "POP3": 110,
    "POP3S": 995,
    "DNS": 53,
    "mDNS": 5353,
    "DHCP": 67,
    "DHCP Client": 68,
    "NTP": 123,
    "SNMP": 161,
    "LDAP": 389,
    "LDAPS": 636,
    "TFTP": 69,
    "SIP": 5060,
    "RDP": 3389,
    "MQTT": 1883,
    "MQTT Secure": 8883,
    "CoAP": 5683,
    "SMB": 445,
    "Kerberos": 88,
    "MySQL": 3306,
    "PostgreSQL": 5432,
    "Redis": 6379,
    "MongoDB": 27017,
    "unknown": 0,
}

# 当前构图流程实际读取的 CSV 列。只读取这些列可以降低大 CSV 的 pandas 内存占用。
REQUIRED_CSV_COLUMNS = [
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "flow_macPair",
    "flow_type",
    "flow_mac_type",
    "fwd_flow_first_Tstamp",
    "bwd_flow_first_Tstamp",
    "fwd_flow_last_Tstamp",
    "bwd_flow_last_Tstamp",
    "fwd_flow_duration",
    "bwd_flow_duration",
    "fwd_flow_byts_s",
    "bwd_flow_byts_s",
    "fwd_pkts_s",
    "bwd_pkts_s",
    "fwd_tot_pkts",
    "bwd_tot_pkts",
    "fwd_pkts_len_total",
    "bwd_pkts_len_total",
    "fwd_pld_len_total",
    "fwd_pld_len_max",
    "fwd_pld_len_min",
    "fwd_pld_len_mean",
    "fwd_pld_len_var",
    "fwd_pld_len_median",
    "fwd_pld_len_mode",
    "bwd_pld_len_total",
    "bwd_pld_len_max",
    "bwd_pld_len_min",
    "bwd_pld_len_mean",
    "bwd_pld_len_var",
    "bwd_pld_len_median",
    "bwd_pld_len_mode",
    "fwd_app_pld_len_max",
    "bwd_app_pld_len_max",
    "fwd_app_pld_len_mean",
    "bwd_app_pld_len_mean",
    "fwd_inter_arrTime_total",
    "fwd_inter_arrTime_max",
    "fwd_inter_arrTime_min",
    "fwd_inter_arrTime_mean",
    "fwd_inter_arrTime_var",
    "fwd_inter_arrTime_median",
    "fwd_inter_arrTime_mode",
    "bwd_inter_arrTime_total",
    "bwd_inter_arrTime_max",
    "bwd_inter_arrTime_min",
    "bwd_inter_arrTime_mean",
    "bwd_inter_arrTime_var",
    "bwd_inter_arrTime_median",
    "bwd_inter_arrTime_mode",
]


def _is_finite_nonnegative(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def getTrafficMultiType(traffic_category: str):
    """
    将流量攻击大类映射为数字标签。

    输入:
        traffic_category: 例如 Benign、DDoS、DoS、Mirai。
    输出:
        对应的整数标签。
    """
    return TRAFFIC_MULTI_TYPE_LABEL[traffic_category]

def getTrafficBiType(traffic_type: str):
    """
    将二分类流量类型映射为数字标签。

    输入:
        traffic_type: "Benign" 或 "Anomaly"。
    输出:
        0 或 1。
    """
    return TRAFFIC_BI_TYPE_LABEL[traffic_type]

def getDeviceType(deviceType: str):
    """
    将 IoT 设备类型映射为图分类标签。

    输入:
        deviceType: 例如 Audio、Camera、Hub。
    输出:
        对应的整数类别。
    """
    return DEVICE_TYPE_LABEL[deviceType]

class Node:
    """保留的简单节点容器；当前主构图流程未直接依赖它。"""

    def __init__(self, node_index, node_type):
        self.node_index = node_index
        self.node_type = node_type

# =========================
# 2. CSV 处理辅助函数
# =========================

def _global_process_worker(csvPath, builder):
    """
    多进程处理单个 CSV 的 worker。

    为什么放在类外:
        ProcessPoolExecutor 需要可 pickle 的顶层函数，放在类方法里容易出现序列化问题。

    输入:
        csvPath: CSV 文件路径。
        builder: TrafficGraphBuild 实例。
    输出:
        builder.csv2srcGraph(rawData) 的返回值。
    """
    csvPath = Path(csvPath)
    workerPid = os.getpid()
    try:
        rawData = pd.read_csv(csvPath, usecols=REQUIRED_CSV_COLUMNS)
        # 调用 builder 内部的方法处理单个 dataframe
        return builder.csv2srcGraph(rawData)
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Worker failed. pid={workerPid}, csv={csvPath.name}, error={type(e).__name__}: {e}\n{tb}"
        ) from e

def getAppPorto(app_porto: str)->int:
    """
    将应用层协议名映射为端口号。

    输入:
        app_porto: 例如 HTTP、HTTPS、DNS。
    输出:
        端口号；未知协议返回 -1。
    """
    try:
        return APP_PORT_LABEL[app_porto]
    except KeyError:
        # 捕获到 KeyError 后，跳过当前包
        return -1  # 返回错误，在调用这个的地方进行排除


def formatPortStats(counter: Counter, numNodes: int) -> str:
    """
    将端口计数转换为原有端口库字符串格式。

    输出格式保持不变:
        "port[prob] port[prob] -1 -2"
    """
    parts = []
    for port, count in counter.items():
        if port is None:
            continue
        prob = float(count) / float(numNodes)
        parts.append(f"{int(port)}[{prob:.6f}]")
    return " ".join(parts) + " -1 -2"


def processFileAndQueueResults(csvFilePath, builder, queue, chunkSize=50000):
    """
    分块处理 CSV，并将每块转换出的图数据放入队列。

    当前主流程使用 _global_process_worker 处理完整 CSV；该函数保留给流式处理场景。
    """
    try:
        chunkIterator = pd.read_csv(csvFilePath, chunksize=chunkSize)
        
        for chunk in chunkIterator:
            graphsFromChunk = builder.csv2srcGraph(chunk)
            # 每个 chunk 的构图结果立刻放入队列，避免一次性占用太多内存。
            if graphsFromChunk:
                queue.put(graphsFromChunk)

    except Exception as e:
        logging.error(f"Error in stream worker for {csvFilePath.name}: {type(e).__name__}: {e}", exc_info=True)
    finally:
        # 关键：无论成功还是失败，都放入一个'None'作为此文件处理结束的信号
        queue.put(None)


class TrafficGraphBuild(Dataset):
    """
    将 CSV 流量记录转换为 PyG Data 图的数据集类。

    外部主要调用:
        builder = TrafficGraphBuild(root)
        dataList, srcPortLib, dstPortLib = builder.loadData()

    输出保持不变:
        dataList: List[Data]
        srcPortLib: graphId -> 源端口统计字符串
        dstPortLib: graphId -> 目的端口统计字符串
    """
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        """
        初始化数据集对象。

        输入:
            root: 数据集根目录。本版本假设 CSV 直接放在 root 下。
            transform / pre_transform / pre_filter: PyG Dataset 标准参数。

        处理:
            创建 raw_dir 和 processed_dir，并初始化兼容旧代码的成员变量。
        """
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)

        # super().__init__ 后才能安全访问 self.raw_dir / self.processed_dir。
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        logging.info(f"Dataset root = {self.root}")

        # 以下成员保留用于兼容旧代码；当前主要数据会直接写入 Data 对象和 processed 目录。
        self.nodeFeatures = []
        self.edgeIndex = []
        self.edgeAttr = []
        self.nodeLabels = []

        self.srcHashtable = defaultdict(list)
        self.dstHashtable = defaultdict(list)
        self.macTypeMap = {}

        self.dataList: List[Data] = []
        self.portStatsList: List[Dict] = []



    def loadData(self) -> Tuple[List[Data], Dict[int, str], Dict[int, str]]:
        """
        数据加载入口。

        输入:
            无，使用初始化时传入的 root。

        输出:
            (dataList, srcPortLib, dstPortLib)
            dataList: Data 对象列表
            srcPortLib: graphId -> 源端口统计字符串
            dstPortLib: graphId -> 目的端口统计字符串

        处理:
            检查 processed_dir 下是否存在端口库和完成标记:
            存在 → _load_from_disk()
            不存在 → _process_and_save()
        """
        src_mapping_path = os.path.join(self.processed_dir, "src_port_lib.pkl")
        dst_mapping_path = os.path.join(self.processed_dir, "dst_port_lib.pkl")
        flag_path = os.path.join(self.processed_dir, "processing_complete.flag")

        if os.path.exists(src_mapping_path) and os.path.exists(dst_mapping_path) and os.path.exists(flag_path):
            logging.info(f"检测到已处理的数据，正在从磁盘加载: {self.processed_dir}")
            return self._load_from_disk()
        else:
            logging.info(f"未检测到处理数据，开始处理 CSV 文件: {self.root}")
            return self._process_and_save()

    # =========================
    # 3. CSV 处理与磁盘缓存
    # =========================
    def _process_and_save(self) -> Tuple[List[Data], Dict[int, str], Dict[int, str]]:
        """
        扫描 root 下所有 CSV，构图并写入 processed 目录。

        输入:
            无，内部扫描 root/*.csv。

        输出:
            同 loadData。

        处理流程：
            找到 CSV 列表；
            默认串行处理；设置 GNN_CSV_WORKERS>1 时使用 ProcessPoolExecutor 并行；
            收集每个 CSV 的返回: batchData, batchSrcPortDicts, batchDstPortDicts;
            对每个子图样本分配全局 graphId: 
            torch.save(data, processed_dir/data_{graphId}.pt);
            srcPortLib[graphId] = srcPortInfo;
            dstPortLib[graphId] = dstPortInfo;
        处理结束后：
            保存 src_port_lib.pkl 和 dst_port_lib.pkl;
            写入 processing_complete.flag 作为完成标记；
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        csvFiles = list(Path(self.root).glob('*.csv'))
        
        if not csvFiles:
            raise FileNotFoundError(f"No CSV files found in {self.root}")

        # 默认串行更稳；仅当显式配置 GNN_CSV_WORKERS>1 时才启用并行。
        maxWorkers = 1
        envWorkers = os.getenv("GNN_CSV_WORKERS")
        if envWorkers is not None:
            try:
                maxWorkers = max(1, min(int(envWorkers), len(csvFiles)))
            except ValueError:
                logging.warning(
                    f"Invalid GNN_CSV_WORKERS='{envWorkers}', fallback to default serial mode."
                )
                maxWorkers = 1
        
        # 端口库和 data_<id>.pt 使用同一个 graphId 对齐。
        srcPortLib: Dict[int, str] = {}  # graphId -> "port[prob] ... -1 -2"
        dstPortLib: Dict[int, str] = {}  # graphId -> "port[prob] ... -1 -2"
        
        graphId = 0 # 全局 ID 计数器

        failedCsvFiles: List[str] = []

        def _persist_batch(batchData, batchSrcPortDicts, batchDstPortDicts):
            """将一个 CSV 生成的一批图保存到磁盘，并同步更新端口库。"""
            nonlocal graphId
            for data, srcDict, dstDict in zip(batchData, batchSrcPortDicts, batchDstPortDicts):
                torch.save(data, os.path.join(self.processed_dir, f"data_{graphId}.pt"))
                srcPortLib[graphId] = srcDict
                dstPortLib[graphId] = dstDict
                graphId += 1
        if maxWorkers == 1:
            logging.info("CSV processing mode: serial (GNN_CSV_WORKERS=1 or unset)")
            progressBar = tqdm(total=len(csvFiles), desc="Processing CSVs (serial)")
            for csvPath in csvFiles:
                csvName = Path(csvPath).name
                try:
                    batchData, batchSrcPortDicts, batchDstPortDicts = _global_process_worker(csvPath, self)
                    logging.info(
                        "CSV processed: %s | graphs=%d",
                        csvName,
                        len(batchData),
                    )
                    _persist_batch(batchData, batchSrcPortDicts, batchDstPortDicts)
                    del batchData, batchSrcPortDicts, batchDstPortDicts
                except Exception as e:
                    failedCsvFiles.append(str(csvPath))
                    logging.error(
                        f"Error processing CSV '{csvName}' in serial mode: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                progressBar.update(1)
            progressBar.close()
        else:
            logging.info(f"CSV processing mode: parallel (GNN_CSV_WORKERS={maxWorkers})")
            with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
                futures = {executor.submit(_global_process_worker, f, self): f for f in csvFiles}
                progressBar = tqdm(total=len(csvFiles), desc="Processing CSVs")
                
                for future in as_completed(futures):
                    csvPath = futures[future]
                    csvName = Path(csvPath).name
                    try:
                        batchData, batchSrcPortDicts, batchDstPortDicts = future.result()
                        logging.info(
                            "CSV processed: %s | graphs=%d",
                            csvName,
                            len(batchData),
                        )
                        _persist_batch(batchData, batchSrcPortDicts, batchDstPortDicts)
                        del batchData, batchSrcPortDicts, batchDstPortDicts
                            
                    except BrokenProcessPool as e:
                        failedCsvFiles.append(str(csvPath))
                        logging.error(
                            f"Process pool broken while processing CSV '{csvName}': {e}",
                            exc_info=True
                        )
                    except Exception as e:
                        failedCsvFiles.append(str(csvPath))
                        logging.error(
                            f"Error processing CSV '{csvName}': {type(e).__name__}: {e}",
                            exc_info=True
                        )
                    
                    progressBar.update(1)
                progressBar.close()
        if failedCsvFiles:
            # 并行模式下某些 CSV 失败时，回退到串行重试，尽量保住可恢复的数据。
            uniqueFailed = list(dict.fromkeys(failedCsvFiles))
            logging.error(
                f"CSV processing failed for {len(uniqueFailed)} file(s): {uniqueFailed}"
            )
            logging.info("Start fallback to sequential processing for failed CSV files...")
            recoveredCount = 0
            stillFailed: List[str] = []
            fallbackBar = tqdm(total=len(uniqueFailed), desc="Fallback CSVs")
            for failedPath in uniqueFailed:
                failedCsv = Path(failedPath)
                try:
                    batchData, batchSrcPortDicts, batchDstPortDicts = _global_process_worker(failedCsv, self)
                    logging.info(
                        "CSV fallback processed: %s | graphs=%d",
                        failedCsv.name,
                        len(batchData),
                    )
                    _persist_batch(batchData, batchSrcPortDicts, batchDstPortDicts)
                    del batchData, batchSrcPortDicts, batchDstPortDicts
                    recoveredCount += 1
                    logging.info(f"Fallback success: {failedCsv.name}")
                except Exception as e:
                    stillFailed.append(str(failedCsv))
                    logging.error(
                        f"Fallback failed for CSV '{failedCsv.name}': {type(e).__name__}: {e}",
                        exc_info=True
                    )
                fallbackBar.update(1)
            fallbackBar.close()
            logging.info(
                f"Fallback completed: recovered {recoveredCount}/{len(uniqueFailed)} failed CSV files."
            )
            if stillFailed:
                raise RuntimeError(
                    f"CSV processing still failed after fallback: {stillFailed}"
                )

        # 保存两个端口库到 .pkl。key 与 data_<graphId>.pt 的 graphId 对齐。
        src_mapping_path = os.path.join(self.processed_dir, "src_port_lib.pkl")
        with open(src_mapping_path, "wb") as f:
            pickle.dump(srcPortLib, f)

        dst_mapping_path = os.path.join(self.processed_dir, "dst_port_lib.pkl")
        with open(dst_mapping_path, "wb") as f:
            pickle.dump(dstPortLib, f)
            
        # 完成标记存在时，下一次 loadData 会直接走 _load_from_disk。
        with open(os.path.join(self.processed_dir, 'processing_complete.flag'), 'w') as f:
            f.write('done')
            
        logging.info(f"处理完成！生成了 {graphId} 个图文件。")
        logging.info(f"源端口库保存于: {src_mapping_path}")
        logging.info(f"目的端口库保存于: {dst_mapping_path}")
    
        return self._load_from_disk()

    def _load_from_disk(self) -> Tuple[List[Data], Dict[int, str], Dict[int, str]]:
        """
        从 processed 目录读取已经构建好的图和端口库。

        输入:
            无。

        输出:
            (dataList, srcPortLib, dstPortLib)

        处理:
            读取 src_port_lib.pkl / dst_port_lib.pkl 得到端口库；
            按 graphId 排序逐个加载 data_{gid}.pt, 组成 dataList。
        """
        # 1. 加载映射字典
        src_mapping_path = os.path.join(self.processed_dir, "src_port_lib.pkl")
        dst_mapping_path = os.path.join(self.processed_dir, "dst_port_lib.pkl")
        with open(src_mapping_path, "rb") as f:
            srcPortLib: Dict[int, str] = pickle.load(f)
        with open(dst_mapping_path, "rb") as f:
            dstPortLib: Dict[int, str] = pickle.load(f)
            
        # 通过端口库中的 graphId 决定加载哪些 data_<id>.pt，保证顺序一致。
        dataList: List[Data] = []
        sorted_ids = sorted(srcPortLib.keys())  
        
        logging.info("正在加载 .pt 文件...")
        for gid in tqdm(sorted_ids, desc="Loading Graphs"):
            pt_path = os.path.join(self.processed_dir, f"data_{gid}.pt")
            if os.path.exists(pt_path):
                # data = torch.load(pt_path)
                data = torch.load(pt_path, weights_only=False)
                dataList.append(data)
            else:
                logging.warning(f"File missing: data_{gid}.pt")
                
        return dataList, srcPortLib, dstPortLib
    # =========================
    # 4. 核心构图逻辑
    # =========================
    def csv2srcGraph(self, rawData: pd.DataFrame, timeThreshold: float = 10.0) -> Tuple[List[Data], List[str], List[str]]:
        """
        将一个 CSV 对应的 DataFrame 转换为多个 IoT-DFIG 子图。

        输入:
            rawData: 一个 CSV 读取出的 DataFrame。
            timeThreshold: 构造边时允许的最大相邻开始时间差，单位为秒。

        输出:
            dataList: PyG Data 图列表，每个 Data 包含 x、edge_index、edge_attr、y。
            srcPortStatsList: 每个子图对应的源端口概率字符串。
            dstPortStatsList: 每个子图对应的目的端口概率字符串。

        构图思路:
            1. 一条 flow 拆成 fwd / bwd 两个节点；
            2. 按源 MAC 聚合节点；
            3. 按时间窗口把同一设备的 flow 节点切成多个子图；
            4. 按开始时间排序，只连接开始时间差不超过阈值的相邻 flow；
            5. 每条边的 edge_attr = [timeDiff, overlap]，重叠为 1，不重叠为 -1。
        """
        nodeFeatures = []  # 全局节点特征表，后续子图用索引引用其中的节点。
        srcHashtable = defaultdict(list)  # srcMac -> 节点元信息列表。
        macLabels = {}  # mac -> 设备类型数值标签，用于生成图标签 y。


        # ---------- 4.1 遍历 flow 行，构造 fwd / bwd 节点 ----------
        for idx, row in rawData.iterrows():
            # 每条 flow 拆成正向节点和反向节点，分别表示两个方向的统计特征。
            fwdFeat, bwdFeat, _, _ = self.flowSplitNode(row)

            if row['flow_mac_type'] in UNKNOWN_DEVICE_TYPES:
                continue
            
            # flow_mac_type 应该是类似 ['Camera', 'Hub'] 的字符串列表。
            try:
                mactypeList = ast.literal_eval(row['flow_mac_type'])
                if not isinstance(mactypeList, list):
                    raise TypeError("Parsed object is not a list")
            except Exception as e:
                error_message = (
                    f"\n\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"!!  数据格式错误 (Data Format Error)  !!\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"在处理 DataFrame 的索引位置 {idx} 时遇到问题。\n"
                    f"原因: 'flow_mac_type' 列的字符串无法被解析成一个列表。\n"
                    f"原始值 (Raw Value): {row['flow_mac_type']}\n"
                    f"Python 异常 (Exception): {type(e).__name__}: {e}\n"
                    f"\n"
                    f"--- 完整的错误行数据 (Full Row Data) ---\n"
                    f"{row.to_string()}\n"
                    f"-----------------------------------------\n"
                )
                raise ValueError(error_message)

            if not mactypeList:
                continue

            if fwdFeat[0] == -1:
                continue
            
            # 获取流节点的 mac 类型。mactypeList 已在上方完成格式校验，这里直接复用。
            srcType = mactypeList[0]
            dstType = mactypeList[-1]

            if srcType in UNKNOWN_DEVICE_TYPES and dstType in UNKNOWN_DEVICE_TYPES:
                continue

            # flow_macPair 记录两个方向的 MAC。这里保持原始 eval 行为，避免改变兼容性。
            macPair = eval(row['flow_macPair'])

            # 添加 fwd 节点：以原始 src_mac 作为设备视角。
            fwdMacSrc = macPair[0]
            fwdMacDst = macPair[-1]
            fwdFirstTime = row['fwd_flow_first_Tstamp']
            fwdLastTime = row['fwd_flow_last_Tstamp']
            fwdSrcIP = row['src_ip']
            fwdDstIP = row['dst_ip']
            fwdSrcPort = row['src_port']
            fwdDstPort = row['dst_port']

            nodeFeatures.append(fwdFeat)
            fwdIndex = len(nodeFeatures) - 1

            if fwdFirstTime not in [None, '', float('nan')] and fwdLastTime not in [None, '', float('nan')] and srcType not in UNKNOWN_DEVICE_TYPES:
                srcHashtable[fwdMacSrc].append(
                    (fwdIndex, fwdFirstTime, fwdLastTime, fwdMacDst, fwdSrcIP, fwdDstIP, fwdSrcPort, fwdDstPort)
                )
            if srcType not in UNKNOWN_DEVICE_TYPES:
                macLabels[fwdMacSrc] = getDeviceType(srcType)

            # 添加 bwd 节点：反向 flow 的源/目的 MAC 与 IP 对调。
            bwdMacSrc = macPair[-1]
            bwdMacDst = macPair[0]
            bwdFirstTime = row['bwd_flow_first_Tstamp']
            bwdLastTime = row['bwd_flow_last_Tstamp']
            bwdSrcIP = row['dst_ip']
            bwdDstIP = row['src_ip']
            # 反向节点的 MAC、IP 和端口必须同时对调。
            # 原 flow 的 dst_port 是反向节点的源端口，src_port 是其目的端口。
            bwdSrcPort = row['dst_port']
            bwdDstPort = row['src_port']

            nodeFeatures.append(bwdFeat)
            bwdIndex = len(nodeFeatures) - 1

            if bwdFirstTime not in [None, '', float('nan')] and bwdLastTime not in [None, '', float('nan')] and dstType not in UNKNOWN_DEVICE_TYPES:
                srcHashtable[bwdMacSrc].append(
                    (bwdIndex, bwdFirstTime, bwdLastTime, bwdMacDst, bwdSrcIP, bwdDstIP, bwdSrcPort, bwdDstPort)
                )
            if dstType not in UNKNOWN_DEVICE_TYPES:
                macLabels[bwdMacSrc] = getDeviceType(dstType)

        # ---------- 4.2 按 MAC 和时间窗口构造 PyG 子图 ----------
        dataList: List[Data] = []
        srcPortStatsList: List[str] = []   # 源端口库: 每个子图一个字符串
        dstPortStatsList: List[str] = []   # 目的端口库: 每个子图一个字符串

        for srcMac, nodes in srcHashtable.items():
            # 同一 srcMac 代表同一设备视角，按 flow 开始时间排序后切窗口。
            sortedNodes = sorted(nodes, key=lambda x: x[1])
            i = 0
            N = len(sortedNodes)

            while i < N:
                # 当前窗口从第 i 个节点开始，窗口宽度由下面的 60 秒条件控制。
                windowNodes = [sortedNodes[i]]
                startTime = sortedNodes[i][1]
                endTime = sortedNodes[i][2]

                j = i + 1
                while j < N and (sortedNodes[j][2] - startTime) <= 60:
                    windowNodes.append(sortedNodes[j])
                    endTime = max(endTime, sortedNodes[j][2])
                    j += 1

                # 将全局节点索引转换为当前子图内部的局部索引。
                nodeGLBIdx = [t[0] for t in windowNodes]
                nodeLocFeats = [nodeFeatures[idx] for idx in nodeGLBIdx]
                nodeLocIdx = {idx: k for k, idx in enumerate(nodeGLBIdx)}

                edgeSources = []
                edgeTargets = []
                edgeAttrs = []

                # 按开始时间排序，只检查相邻 flow。边始终由较早开始的 flow 指向较晚者。
                # 若 earlier.start < later.start < earlier.end，则两个时间区间重叠，标记为 1；
                # 否则标记为 -1。只有后一个 flow 严格晚于前一个 flow，且开始时间差
                # 不超过阈值时才建边；相同开始时间不存在“较早 -> 较晚”的方向，因此跳过。
                wnd_sorted = sorted(windowNodes, key=lambda x: x[1])
                for m in range(len(wnd_sorted) - 1):
                    earlierNode = wnd_sorted[m]
                    laterNode = wnd_sorted[m + 1]
                    timeDiff = laterNode[1] - earlierNode[1]
                    if 0.0 < timeDiff <= timeThreshold:
                        overlap = 1.0 if earlierNode[1] < laterNode[1] < earlierNode[2] else -1.0
                        edgeSources.append(nodeLocIdx[earlierNode[0]])
                        edgeTargets.append(nodeLocIdx[laterNode[0]])
                        edgeAttrs.append([timeDiff, overlap])

                if len(nodeLocFeats) < 3:
                    # 过小的子图结构信息不足，跳过。
                    i = j
                    continue

                # ---------- 4.3 组装 PyG Data ----------
                x = torch.tensor(nodeLocFeats, dtype=torch.float32)
                edgeIndex = torch.tensor([edgeSources, edgeTargets], dtype=torch.int64)
                if len(edgeAttrs) == 0:
                    edgeAttr = torch.empty((0, 2), dtype=torch.float32)
                else:
                    edgeAttr = torch.tensor(edgeAttrs, dtype=torch.float32).view(-1, 2)


                # 将节点入度/出度拼到原始 flow 特征后，作为额外结构特征。
                numNodes = x.size(0)
                outDegree = torch.bincount(edgeIndex[0], minlength=numNodes)
                inDegree = torch.bincount(edgeIndex[1], minlength=numNodes)
                degrees = torch.stack([inDegree, outDegree], dim=1)
                x = torch.cat([x, degrees.float()], dim=1)


                srcPorts = [t[6] for t in windowNodes]
                srcPortSequence = [t[6] for t in wnd_sorted]
                dstPorts = [t[7] for t in windowNodes]
                dstPortSequence = [t[7] for t in wnd_sorted]
                srcPortCounter = Counter(srcPorts)
                dstPortCounter = Counter(dstPorts)

                y = macLabels[srcMac]

                # Data 仅保存 GNN 训练所需字段:
                # x: 节点特征；edge_index: 边；edge_attr: 开始时间差和重叠标记；y: 图标签；
                data = Data(
                    x=x,
                    edge_index=edgeIndex,
                    edge_attr=edgeAttr,
                    y=y,
                    srcPortSequence=srcPortSequence,
                    dstPortSequence=dstPortSequence,
                    # num_src_ips=numSrcIPs,
                    # num_dst_ips=numDstIPs
                )

                dataList.append(data)

                # 端口库字符串格式保持为 "port[prob] ... -1 -2"。
                srcPortStr = formatPortStats(srcPortCounter, numNodes)
                dstPortStr = formatPortStats(dstPortCounter, numNodes)

                srcPortStatsList.append(srcPortStr)
                dstPortStatsList.append(dstPortStr)

                i = j  # 移动到下一个时间窗口。

        return dataList, srcPortStatsList, dstPortStatsList


    def flowSplitNode(self, row: pd.Series) -> Tuple[tuple, tuple, tuple, tuple]:
        """
        将一行 flow 拆成正向节点和反向节点。

        输入:
            row: DataFrame 中的一行 flow 记录。

        输出:
            fwdFeature: 正向 flow 节点特征；
            bwdFeature: 反向 flow 节点特征；
            fwdLabel: 正向五元组 + flow_type；
            bwdLabel: 反向五元组 + flow_type。

        说明:
            特征缺失值会被替换为 -1，保证后续 torch.tensor 可以正常构建。
        """
        # 正向节点特征：使用 fwd_* 统计特征描述 src -> dst 方向的 flow 行为。
        fwdPacketCount = float(row['fwd_tot_pkts']) if _is_finite_nonnegative(row['fwd_tot_pkts']) else 0.0
        bwdPacketCount = float(row['bwd_tot_pkts']) if _is_finite_nonnegative(row['bwd_tot_pkts']) else 0.0
        packetLogRatio = max(-5.0, min(5.0, math.log((fwdPacketCount + 1.0) / (bwdPacketCount + 1.0))))
        pairedDelayValid = (
            fwdPacketCount > 0.0
            and bwdPacketCount > 0.0
            and _is_finite_nonnegative(row['fwd_flow_first_Tstamp'])
            and _is_finite_nonnegative(row['bwd_flow_first_Tstamp'])
        )
        pairedDelayLog = (
            math.log1p(max(0.0, float(row['bwd_flow_first_Tstamp']) - float(row['fwd_flow_first_Tstamp'])))
            if pairedDelayValid
            else 0.0
        )

        fwdFeature = (
            # getAppPorto(row['pkt_appProto']),
            row['fwd_flow_duration'],
            row['fwd_flow_byts_s'],
            row['fwd_pkts_s'],
            row['fwd_tot_pkts'],
            # 与 bwdFeature 同一位置的 bwd_pkts_len_total 保持语义一致。
            row['fwd_pkts_len_total'],
            packetLogRatio,
            # row['fwd_header_len_'],

            # row['fwd_pktLen_max'],
            # row['fwd_pktLen_min'],
            # row['fwd_pktLen_mean'],
            # row['fwd_pktLen_std'],
            # row['fwd_pktLen_var'],
            # row['fwd_pktLen_median'],
            # row['fwd_pktLen_mode'],
            # row['fwd_pktLen_skew'],
            # row['fwd_pktLen_skew2'],
            # row['fwd_pktLen_cov'],

            row['fwd_pld_len_total'],
            row['fwd_pld_len_max'],
            row['fwd_pld_len_min'],
            row['fwd_pld_len_mean'],
            row['fwd_pld_len_var'],
            row['fwd_pld_len_median'],
            row['fwd_pld_len_mode'],

            # row['fwd_app_pld_len_total'],
            # row['fwd_app_pld_len_max'],
            # row['fwd_app_pld_len_min'],
            # row['fwd_app_pld_len_mean'],
            # row['fwd_app_pld_len_std'],
            # row['fwd_app_pld_len_var'],
            # row['fwd_app_pld_len_median'],
            # row['fwd_app_pld_len_mode'],
            # row['fwd_app_pld_len_skew'],
            # row['fwd_app_pld_len_skew2'],
            # row['fwd_app_pld_len_cov'],
            
            row['fwd_inter_arrTime_total'],
            row['fwd_inter_arrTime_max'],
            row['fwd_inter_arrTime_min'],
            row['fwd_inter_arrTime_mean'],
            row['fwd_inter_arrTime_var'],
            row['fwd_inter_arrTime_median'],
            row['fwd_inter_arrTime_mode'],
            row['fwd_app_pld_len_max'],
            row['fwd_app_pld_len_mean'],
            1.0,
            1.0 if bwdPacketCount > 0.0 else 0.0,
            pairedDelayLog,
            1.0 if pairedDelayValid else 0.0,

        )

        # 反向节点特征：使用 bwd_* 统计特征描述 dst -> src 方向的 flow 行为。
        bwdFeature = (
            # getAppPorto(row['pkt_appProto']),
            row['bwd_flow_duration'],
            row['bwd_flow_byts_s'],
            row['bwd_pkts_s'],
            row['bwd_tot_pkts'],
            row['bwd_pkts_len_total'],
            -packetLogRatio,
            # row['bwd_header_len'],
            # row['bwd_pktLen_max'],
            # row['bwd_pktLen_min'],
            # row['bwd_pktLen_mean'],
            # row['bwd_pktLen_std'],
            # row['bwd_pktLen_var'],
            # row['bwd_pktLen_median'],
            # row['bwd_pktLen_mode'],
            # row['bwd_pktLen_skew'],
            # row['bwd_pktLen_skew2'],
            # row['bwd_pktLen_cov'],
            row['bwd_pld_len_total'],
            row['bwd_pld_len_max'],
            row['bwd_pld_len_min'],
            row['bwd_pld_len_mean'],
            row['bwd_pld_len_var'],
            row['bwd_pld_len_median'],
            row['bwd_pld_len_mode'],
            # row['bwd_app_pld_len_total'],
            # row['bwd_app_pld_len_max'],
            # row['bwd_app_pld_len_min'],
            # row['bwd_app_pld_len_mean'],
            # row['bwd_app_pld_len_std'],
            # row['bwd_app_pld_len_var'],
            # row['bwd_app_pld_len_median'],
            # row['bwd_app_pld_len_mode'],
            # row['bwd_app_pld_len_skew'],
            # row['bwd_app_pld_len_skew2'],
            # row['bwd_app_pld_len_cov'],
            row['bwd_inter_arrTime_total'],
            row['bwd_inter_arrTime_max'],
            row['bwd_inter_arrTime_min'],
            row['bwd_inter_arrTime_mean'],
            row['bwd_inter_arrTime_var'],
            row['bwd_inter_arrTime_median'],
            row['bwd_inter_arrTime_mode'],
            row['bwd_app_pld_len_max'],
            row['bwd_app_pld_len_mean'],
            -1.0,
            1.0 if fwdPacketCount > 0.0 else 0.0,
            pairedDelayLog,
            1.0 if pairedDelayValid else 0.0,
        )

        # 正向五元组标签，当前不放入 Data，保留给调试或扩展使用。
        fwdLabel = (
            row["src_ip"], # fwd节点的源IP
            row["src_port"], # fwd节点的源port
            row["dst_ip"], # fwd节点的目的IP
            row["dst_port"], # fwd节点的目的port
            row["protocol"], # fwd节点的协议
            row["flow_type"] # 流的类型（arpspoofing，还是良性）
        )

        # 反向五元组标签，源/目的 IP 和端口对调。
        bwdLabel = (
            row["dst_ip"], # bwd节点的源IP
            row["dst_port"], # bwd节点的源port
            row["src_ip"], # bwd节点的目的IP
            row["src_port"], # bwd节点的目的port
            row["protocol"], # bwd节点的协议
            row["flow_type"] # 流的类型（arpspoofing，还是良性）
        )

        return [(-1 if pd.isna(i) else i) for i in fwdFeature], [(-1 if pd.isna(i) else i) for i in bwdFeature], fwdLabel, bwdLabel

    # =========================
    # 5. PyG Dataset 标准接口
    # =========================
    # 这些属性/方法让 TrafficGraphBuild 兼容 PyTorch Geometric Dataset。
    # 但本文件的实际处理入口是 loadData()，process() 被保留为空实现。
    @property
    def raw_dir(self) -> str:
        """原始 CSV 所在目录。本版本约定 CSV 直接放在 root 下。"""
        return self.root

    @property
    def processed_dir(self) -> str:
        """带构图版本的缓存目录，保存 data_<id>.pt 和端口库文件。"""
        return os.path.join(self.root, GRAPH_CACHE_DIR_NAME)

    @property
    def raw_file_names(self):
        """返回 root 下所有 CSV 文件名。"""
        return [p.name for p in Path(self.raw_dir).glob("*.csv")]



    @property
    def processed_file_names(self):
        """PyG 用于判断 processed 是否完成的文件名。"""
        return ["processing_complete.flag"]



    def download(self):
        """本数据集来自本地 CSV，不需要下载。"""
        pass

    def process(self):
        """不走 PyG 默认 process 流程；统一由 loadData 控制处理与加载。"""
        return

    
    def len(self):
        """返回 processed_dir 中缓存的图数量。"""
        return len([name for name in os.listdir(self.processed_dir) if name.startswith('data_') and name.endswith('.pt')])

    def get(self, idx):
        """按索引加载 data_<idx>.pt 并返回一个 PyG Data 图。"""
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
