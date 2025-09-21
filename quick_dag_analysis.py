import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

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
    
    # 存储操作类型和执行单元的映射，用于统计
    op_types = []
    pipe_types = []
    cycles_list = []
    
    # 添加节点及其属性
    for node in data['Nodes']:
        node_id = node['Id']
        op_type = node['Op']
        op_types.append(op_type)
        
        # 为不同类型的操作设置不同的颜色
        color_map = {
            'ALLOC': '#FF9999',  # 浅红色
            'FREE': '#FF6666',  # 红色
            'COPY_IN': '#99CCFF',  # 浅蓝色
            'COPY_OUT': '#99FF99',  # 浅绿色
            'MOVE': '#FFFF99',  # 浅黄色
            'MATMUL': '#FF99FF',  # 浅紫色
            'FLASH_ATTENTION': '#99FFCC',  # 浅青色
            'CONV': '#FFCC99'  # 浅橙色
        }
        color = color_map.get(op_type, '#CCCCCC')  # 默认灰色
        
        # 存储节点属性
        G.add_node(node_id, color=color, op_type=op_type)
        
        # 收集执行单元和周期信息
        if op_type not in ['ALLOC', 'FREE']:
            if 'Pipe' in node:
                pipe_types.append(node['Pipe'])
            if 'Cycles' in node:
                cycles_list.append(node['Cycles'])
    
    # 添加边
    if 'Edges' in data:
        for edge in data['Edges']:
            source, target = edge
            G.add_edge(source, target)
    
    # 返回图和统计数据
    return G, {
        'op_types': op_types,
        'pipe_types': pipe_types,
        'cycles_list': cycles_list
    }

# 快速分析DAG结构
def quick_analysis(G, stats, case_name):
    print(f"\n=== {case_name} DAG快速分析 ===")
    
    # 基本统计
    print(f"节点总数: {len(G.nodes)}")
    print(f"边总数: {len(G.edges)}")
    
    # 计算入度和出度
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # 找出源节点（入度为0）和汇节点（出度为0）
    source_nodes = [node for node, degree in in_degree.items() if degree == 0]
    sink_nodes = [node for node, degree in out_degree.items() if degree == 0]
    
    print(f"源节点数: {len(source_nodes)}")
    print(f"汇节点数: {len(sink_nodes)}")
    
    # 操作类型统计
    op_counts = Counter(stats['op_types'])
    print("\n操作类型分布:")
    for op_type, count in op_counts.items():
        percentage = (count / len(G.nodes)) * 100
        print(f"  {op_type}: {count}个节点 ({percentage:.2f}%)")
    
    # 执行单元(Pipe)统计
    if stats['pipe_types']:
        pipe_counts = Counter(stats['pipe_types'])
        print("\n执行单元(Pipe)分布:")
        for pipe, count in pipe_counts.items():
            percentage = (count / len(stats['pipe_types'])) * 100
            print(f"  {pipe}: {count}个节点 ({percentage:.2f}%)")
    
    # 时钟周期(Cycles)统计
    if stats['cycles_list']:
        print("\n时钟周期(Cycles)统计:")
        print(f"  平均值: {np.mean(stats['cycles_list']):.2f}")
        print(f"  中位数: {np.median(stats['cycles_list']):.2f}")
        print(f"  最小值: {np.min(stats['cycles_list']):.2f}")
        print(f"  最大值: {np.max(stats['cycles_list']):.2f}")
        print(f"  总和: {np.sum(stats['cycles_list']):.2f}")

# 绘制简化的DAG图（优化了大型图的处理）
def draw_simplified_dag(G, stats, case_name, output_file=None, max_nodes=1000):
    # 如果节点太多，进行采样以提高绘图速度
    if len(G.nodes) > max_nodes:
        print(f"节点数量过多({len(G.nodes)}个)，将采样{max_nodes}个节点进行可视化...")
        # 创建一个全新的图，而不是修改子图
        H = nx.DiGraph()
        
        # 选择前max_nodes个节点
        sampled_nodes = list(G.nodes)[:max_nodes]
        
        # 添加采样节点及其属性
        for node in sampled_nodes:
            # 确保所有节点都有必要的属性
            node_attrs = dict(G.nodes[node])
            op_type = node_attrs.get('op_type', 'UNKNOWN')
            
            # 设置颜色属性
            color_map = {
                'ALLOC': '#FF9999',  # 浅红色
                'FREE': '#FF6666',  # 红色
                'COPY_IN': '#99CCFF',  # 浅蓝色
                'COPY_OUT': '#99FF99',  # 浅绿色
                'MOVE': '#FFFF99',  # 浅黄色
                'MATMUL': '#FF99FF',  # 浅紫色
                'FLASH_ATTENTION': '#99FFCC',  # 浅青色
                'CONV': '#FFCC99',  # 浅橙色
                'UNKNOWN': '#CCCCCC'  # 未知类型默认灰色
            }
            
            # 确保有color和op_type属性
            node_attrs['color'] = color_map.get(op_type, '#CCCCCC')
            node_attrs['op_type'] = op_type
            
            H.add_node(node, **node_attrs)
        
        # 添加这些节点之间的边
        for edge in G.edges:
            if edge[0] in H and edge[1] in H:
                H.add_edge(*edge)
        
        G = H
    
    # 使用spring布局，适合大型图
    pos = nx.spring_layout(G, k=0.1, iterations=30, seed=42)
    
    # 获取节点颜色，确保所有节点都有颜色
    colors = []
    for node in G.nodes:
        # 直接使用get方法获取属性，设置默认值以避免KeyError
        node_attrs = G.nodes[node]
        op_type = node_attrs.get('op_type', 'UNKNOWN')
        
        # 根据操作类型设置颜色
        color_map = {
            'ALLOC': '#FF9999',  # 浅红色
            'FREE': '#FF6666',  # 红色
            'COPY_IN': '#99CCFF',  # 浅蓝色
            'COPY_OUT': '#99FF99',  # 浅绿色
            'MOVE': '#FFFF99',  # 浅黄色
            'MATMUL': '#FF99FF',  # 浅紫色
            'FLASH_ATTENTION': '#99FFCC',  # 浅青色
            'CONV': '#FFCC99',  # 浅橙色
            'UNKNOWN': '#CCCCCC'  # 未知类型默认灰色
        }
        colors.append(color_map.get(op_type, '#CCCCCC'))
    
    # 创建画布
    plt.figure(figsize=(16, 12))
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=True, arrowstyle='->', arrowsize=5)
    
    # 添加图例
    handles = []
    labels = []
    color_map = {
        'ALLOC': '#FF9999',  # 浅红色
        'FREE': '#FF6666',  # 红色
        'COPY_IN': '#99CCFF',  # 浅蓝色
        'COPY_OUT': '#99FF99',  # 浅绿色
        'MOVE': '#FFFF99',  # 浅黄色
        'MATMUL': '#FF99FF',  # 浅紫色
        'FLASH_ATTENTION': '#99FFCC',  # 浅青色
        'CONV': '#FFCC99'  # 浅橙色
    }
    
    # 获取图中实际存在的操作类型
    op_types = set([G.nodes[node]['op_type'] for node in G.nodes])
    
    for op_type, color in color_map.items():
        if op_type in op_types:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(op_type)
    
    plt.legend(handles, labels, title="操作类型", loc='best')
    
    # 设置标题
    plt.title(f"{case_name} - 神经网络处理器核内调度计算图\n" +
              f"节点数: {len(G.nodes)}, 边数: {len(G.edges)}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图形
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"图形已保存到 {output_file}")
    else:
        plt.show()
    
    plt.close()

# 绘制操作类型和执行单元的统计图表
def draw_stat_charts(stats, case_name, output_prefix):
    # 操作类型分布饼图
    plt.figure(figsize=(8, 6))
    op_counts = Counter(stats['op_types'])
    plt.pie(
        op_counts.values(),
        labels=op_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=[
            '#FF9999', '#FF6666', '#99CCFF', '#99FF99', 
            '#FFFF99', '#FF99FF', '#99FFCC', '#FFCC99'
        ][:len(op_counts)]
    )
    plt.axis('equal')
    plt.title(f"{case_name} - 操作类型分布")
    pie_file = f'{output_prefix}_op_distribution.png'
    plt.savefig(pie_file, dpi=200, bbox_inches='tight')
    print(f"操作类型分布饼图已保存到 {pie_file}")
    plt.close()
    
    # 执行单元分布条形图
    if stats['pipe_types']:
        plt.figure(figsize=(8, 6))
        pipe_counts = Counter(stats['pipe_types'])
        pipes = list(pipe_counts.keys())
        counts = list(pipe_counts.values())
        
        plt.bar(pipes, counts, color='#8884d8')
        plt.title(f"{case_name} - 执行单元分布")
        plt.xlabel('执行单元')
        plt.ylabel('节点数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        bar_file = f'{output_prefix}_pipe_distribution.png'
        plt.savefig(bar_file, dpi=200, bbox_inches='tight')
        print(f"执行单元分布条形图已保存到 {bar_file}")
        plt.close()

# 主函数
if __name__ == "__main__":
    # 用户可以在这里修改要分析的文件名
    json_file = 'Matmul_Case0.json'  # 可以替换为其他文件如 'Conv_Case0.json' 或 'FlashAttention_Case0.json'
    
    try:
        print(f"正在处理文件: {json_file}")
        
        # 加载DAG数据
        data = load_dag_from_json(json_file)
        
        # 创建DAG图和统计数据
        G, stats = create_dag_graph(data)
        
        # 案例名称（从文件名提取）
        case_name = json_file.replace('.json', '')
        
        # 快速分析DAG结构
        quick_analysis(G, stats, case_name)
        
        # 绘制并保存简化的DAG图
        output_image = f'{case_name}_quick_visualization.png'
        draw_simplified_dag(G, stats, case_name, output_image)
        
        # 绘制统计图表
        draw_stat_charts(stats, case_name, case_name)
        
        print(f"\n{case_name} 快速分析完成！")
    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {str(e)}")