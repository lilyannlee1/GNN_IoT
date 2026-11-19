import networkx as nx
import matplotlib.pyplot as plt
import torch

def visualize_graph(data):
    """
    可视化图的函数，适配二维边特征 [时间差, 方向]
    """
    G = nx.DiGraph()

    # 获取节点特征和边的索引信息
    nodes = data.x.numpy()
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()  # shape: [num_edges, 2]

    # 添加节点
    for i in range(len(nodes)):
        G.add_node(i, feature=nodes[i])

    # 添加边，使用 edge_attr[:, 0] 作为 layout 权重；完整 edge_attr 显示为标签
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        time_diff = edge_attr[i][0]
        direction = edge_attr[i][1]
        G.add_edge(src, tgt, weight=time_diff, edge_attr=[time_diff, direction])

    # 使用 spring_layout，仅用 weight 参与 layout 优化
    pos = nx.spring_layout(G, weight='weight')

    node_labels = {i: f"Node {i}" for i in range(len(nodes))}

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # 显示完整边属性 [时间差, 方向] 作为标签
    edge_labels = {
        (u, v): f"[{attrs['edge_attr'][0]:.2f}, {int(attrs['edge_attr'][1])}]"
        for u, v, attrs in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph Visualization with 2D Edge Attributes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("graph_2d.png")
