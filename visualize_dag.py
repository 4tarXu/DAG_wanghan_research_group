import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 读取JSON文件
def load_dag_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 创建DAG图
def create_dag_graph(data):
    G = nx.DiGraph()
    
    # 添加节点及其属性
    for node in data['Nodes']:
        node_id = node['Id']
        # 添加节点标签，包含操作类型和循环数
        label = f"{node['Op']}\n(Id:{node_id})"
        if 'Cycles' in node:
            label += f"\n{node['Cycles']} cycles"
        
        # 为不同类型的操作设置不同的颜色
        op_type = node['Op']
        color_map = {
            'ALLOC': '#FF9999',  # 浅红色
            'COPY_IN': '#99CCFF',  # 浅蓝色
            'COPY_OUT': '#99FF99',  # 浅绿色
            'MOVE': '#FFFF99',  # 浅黄色
            'MATMUL': '#FF99FF',  # 浅紫色
            'FLASH_ATTENTION': '#99FFCC',  # 浅青色
            'CONV': '#FFCC99'  # 浅橙色
        }
        color = color_map.get(op_type, '#CCCCCC')  # 默认灰色
        
        G.add_node(node_id, label=label, color=color, op_type=op_type)
    
    # 添加边
    if 'Edges' in data:
        for edge in data['Edges']:
            source, target = edge
            G.add_edge(source, target)
    
    return G

# 绘制DAG图
def draw_dag_graph(G, output_file=None):
    # 由于节点数量较大，我们需要先进行一些预处理来改善布局效果
    # 尝试使用拓扑排序来创建层次结构
    try:
        # 获取拓扑排序
        topological_order = list(nx.topological_sort(G))
        
        # 创建层次映射，根据拓扑顺序分配层次
        levels = {}
        for i, node in enumerate(topological_order):
            levels[node] = i // 50  # 每50个节点为一层
        
        # 使用multipartite布局
        pos = nx.multipartite_layout(G, subset_key=lambda n: levels[n], align='horizontal')
    except:
        # 如果拓扑排序失败或其他错误，使用spring布局
        pos = nx.spring_layout(G, k=0.1, iterations=50, seed=42)
    
    # 获取节点颜色
    colors = [G.nodes[node]['color'] for node in G.nodes]
    
    # 获取节点标签
    labels = nx.get_node_attributes(G, 'label')
    
    # 统计操作类型数量
    op_counts = {}
    for node in G.nodes:
        op_type = G.nodes[node]['op_type']
        if op_type not in op_counts:
            op_counts[op_type] = 0
        op_counts[op_type] += 1
    
    # 创建画布
    plt.figure(figsize=(20, 15))
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowstyle='->', arrowsize=10)
    
    # 只绘制部分标签以避免拥挤
    if len(G.nodes) > 100:
        # 为每种操作类型选择一些代表性节点显示标签
        selected_labels = {}
        seen_ops = set()
        for node in G.nodes:
            op_type = G.nodes[node]['op_type']
            if op_type not in seen_ops:
                selected_labels[node] = labels[node]
                seen_ops.add(op_type)
        nx.draw_networkx_labels(G, pos, labels=selected_labels, font_size=8)
    else:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # 添加图例
    handles = []
    labels = []
    for op_type, color in {
        'ALLOC': '#FF9999',
        'COPY_IN': '#99CCFF',
        'COPY_OUT': '#99FF99',
        'MOVE': '#FFFF99',
        'MATMUL': '#FF99FF',
        'FLASH_ATTENTION': '#99FFCC',
        'CONV': '#FFCC99'
    }.items():
        if op_type in op_counts:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(f"{op_type} ({op_counts[op_type]})")
    
    plt.legend(handles, labels, title="操作类型", loc='best')
    
    # 设置标题
    plt.title(f"神经网络处理器核内调度计算图 (节点数: {len(G.nodes)}, 边数: {len(G.edges)})")
    plt.axis('off')
    plt.tight_layout()
    
    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图形已保存到 {output_file}")
    else:
        plt.show()

# 分析DAG结构
def analyze_dag_structure(G):
    # 计算入度和出度
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # 找出源节点（入度为0）和汇节点（出度为0）
    source_nodes = [node for node, degree in in_degree.items() if degree == 0]
    sink_nodes = [node for node, degree in out_degree.items() if degree == 0]
    
    # 计算图的高度（最长路径长度）
    try:
        # 使用拓扑排序计算最长路径
        topological_order = list(nx.topological_sort(G))
        longest_path_length = 0
        
        # 初始化距离字典
        distance = {node: 0 for node in G.nodes}
        
        # 按照拓扑顺序更新距离
        for node in topological_order:
            for successor in G.successors(node):
                if distance[successor] < distance[node] + 1:
                    distance[successor] = distance[node] + 1
                longest_path_length = max(longest_path_length, distance[successor])
    except nx.NetworkXUnfeasible:
        longest_path_length = "图中存在环，无法计算最长路径"
    
    # 打印分析结果
    print("DAG结构分析:")
    print(f"节点总数: {len(G.nodes)}")
    print(f"边总数: {len(G.edges)}")
    print(f"源节点数: {len(source_nodes)}")
    print(f"汇节点数: {len(sink_nodes)}")
    print(f"图的高度（最长路径长度）: {longest_path_length}")
    
    # 统计操作类型分布
    op_counts = {}
    for node in G.nodes:
        op_type = G.nodes[node]['op_type']
        if op_type not in op_counts:
            op_counts[op_type] = 0
        op_counts[op_type] += 1
    
    print("操作类型分布:")
    for op_type, count in op_counts.items():
        print(f"  {op_type}: {count} 个节点")

# 主函数
if __name__ == "__main__":
    # JSON文件路径
    json_file = 'Matmul_Case0.json'
    
    # 加载DAG数据
    data = load_dag_from_json(json_file)
    
    # 创建DAG图1
    G = create_dag_graph(data)
    
    # 分析DAG结构
    analyze_dag_structure(G)
    
    # 绘制并保存DAG图
    output_image = 'Matmul_Case0_dag_visualization.png'
    draw_dag_graph(G, output_image)