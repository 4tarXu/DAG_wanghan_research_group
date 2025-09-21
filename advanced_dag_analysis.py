import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict

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
        
        # 根据节点类型构建不同的标签
        if node['Op'] in ['ALLOC', 'FREE']:
            # 缓存管理节点
            label = f"{node['Op']}\nId:{node_id}\nBuf:{node['BufId']}\nSize:{node['Size']}\nType:{node['Type']}"
        else:
            # 操作节点
            label = f"{node['Op']}\nId:{node_id}\nPipe:{node['Pipe']}\nCycles:{node['Cycles']}"
            if 'Bufs' in node:
                label += f"\nBufs:{','.join(map(str, node['Bufs']))}"
        
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
        color = color_map.get(node['Op'], '#CCCCCC')  # 默认灰色
        
        # 存储节点的所有属性
        node_attrs = node.copy()
        node_attrs['label'] = label
        node_attrs['color'] = color
        
        G.add_node(node_id, **node_attrs)
    
    # 添加边
    if 'Edges' in data:
        for edge in data['Edges']:
            source, target = edge
            G.add_edge(source, target)
    
    return G

# 详细分析DAG结构
def detailed_analysis(G, case_name):
    print(f"\n=== {case_name} DAG详细分析 ===")
    
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
    op_types = [G.nodes[node]['Op'] for node in G.nodes]
    op_counts = Counter(op_types)
    print("\n操作类型分布:")
    for op_type, count in op_counts.items():
        percentage = (count / len(G.nodes)) * 100
        print(f"  {op_type}: {count}个节点 ({percentage:.2f}%)")
    
    # 执行单元(Pipe)统计（仅对操作节点）
    pipe_nodes = [node for node in G.nodes if G.nodes[node]['Op'] not in ['ALLOC', 'FREE']]
    if pipe_nodes:
        pipes = [G.nodes[node]['Pipe'] for node in pipe_nodes]
        pipe_counts = Counter(pipes)
        print("\n执行单元(Pipe)分布:")
        for pipe, count in pipe_counts.items():
            percentage = (count / len(pipe_nodes)) * 100
            print(f"  {pipe}: {count}个节点 ({percentage:.2f}%)")
    
    # 时钟周期(Cycles)统计（仅对操作节点）
    if pipe_nodes:
        cycles = [G.nodes[node]['Cycles'] for node in pipe_nodes]
        print("\n时钟周期(Cycles)统计:")
        print(f"  平均值: {np.mean(cycles):.2f}")
        print(f"  中位数: {np.median(cycles):.2f}")
        print(f"  最小值: {np.min(cycles):.2f}")
        print(f"  最大值: {np.max(cycles):.2f}")
        print(f"  总和: {np.sum(cycles):.2f}")
    
    # 缓存类型统计（仅对ALLOC节点）
    alloc_nodes = [node for node in G.nodes if G.nodes[node]['Op'] == 'ALLOC']
    if alloc_nodes:
        cache_types = [G.nodes[node]['Type'] for node in alloc_nodes]
        cache_counts = Counter(cache_types)
        print("\n缓存类型分布:")
        for cache_type, count in cache_counts.items():
            percentage = (count / len(alloc_nodes)) * 100
            print(f"  {cache_type}: {count}个节点 ({percentage:.2f}%)")
    
    # 缓冲区大小统计（仅对ALLOC节点）
    if alloc_nodes:
        buffer_sizes = [G.nodes[node]['Size'] for node in alloc_nodes]
        print("\n缓冲区大小统计:")
        print(f"  平均值: {np.mean(buffer_sizes):.2f}")
        print(f"  中位数: {np.median(buffer_sizes):.2f}")
        print(f"  最小值: {np.min(buffer_sizes):.2f}")
        print(f"  最大值: {np.max(buffer_sizes):.2f}")
        print(f"  总和: {np.sum(buffer_sizes):.2f}")
    
    return {
        'case_name': case_name,
        'total_nodes': len(G.nodes),
        'total_edges': len(G.edges),
        'source_nodes': len(source_nodes),
        'sink_nodes': len(sink_nodes),
        'op_counts': op_counts,
        'pipe_counts': pipe_counts if 'pipe_counts' in locals() else None,
        'cycles_stats': {
            'mean': np.mean(cycles) if 'cycles' in locals() else 0,
            'median': np.median(cycles) if 'cycles' in locals() else 0,
            'min': np.min(cycles) if 'cycles' in locals() else 0,
            'max': np.max(cycles) if 'cycles' in locals() else 0,
            'sum': np.sum(cycles) if 'cycles' in locals() else 0
        } if 'cycles' in locals() else None,
        'cache_counts': cache_counts if 'cache_counts' in locals() else None,
        'buffer_stats': {
            'mean': np.mean(buffer_sizes) if 'buffer_sizes' in locals() else 0,
            'median': np.median(buffer_sizes) if 'buffer_sizes' in locals() else 0,
            'min': np.min(buffer_sizes) if 'buffer_sizes' in locals() else 0,
            'max': np.max(buffer_sizes) if 'buffer_sizes' in locals() else 0,
            'sum': np.sum(buffer_sizes) if 'buffer_sizes' in locals() else 0
        } if 'buffer_sizes' in locals() else None
    }

# 绘制改进的DAG图
def draw_enhanced_dag(G, case_name, output_file=None):
    # 由于节点数量可能很大，我们需要优化布局
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
    
    # 根据节点类型设置大小
    node_sizes = []
    for node in G.nodes:
        if G.nodes[node]['Op'] in ['ALLOC', 'FREE']:
            node_sizes.append(200)
        else:
            # 操作节点大小根据Cycles调整
            cycles = G.nodes[node]['Cycles']
            # 将cycles映射到合理的节点大小范围
            size = min(100 + cycles * 2, 500)
            node_sizes.append(size)
    
    # 创建画布
    plt.figure(figsize=(24, 18))
    
    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowstyle='->', arrowsize=10)
    
    # 如果节点数量不太多，显示标签
    if len(G.nodes) < 200:
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    else:
        # 只显示一些关键节点的标签
        key_ops = ['MATMUL', 'CONV', 'FLASH_ATTENTION']
        selected_labels = {}
        for node in G.nodes:
            if G.nodes[node]['Op'] in key_ops:
                # 只显示部分关键节点的标签以避免拥挤
                if int(node) % 10 == 0:  # 每10个同类型节点显示一个
                    selected_labels[node] = f"{G.nodes[node]['Op']}\nId:{node}"
        nx.draw_networkx_labels(G, pos, labels=selected_labels, font_size=8)
    
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
    op_types = set([G.nodes[node]['Op'] for node in G.nodes])
    
    for op_type, color in color_map.items():
        if op_type in op_types:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(op_type)
    
    plt.legend(handles, labels, title="操作类型", loc='best')
    
    # 设置标题
    plt.title(f"{case_name} - 神经网络处理器核内调度计算图\n" +
              f"节点数: {len(G.nodes)}, 边数: {len(G.edges)}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图形已保存到 {output_file}")
    else:
        plt.show()

# 绘制统计图表
def draw_statistics(analysis_results, output_prefix):
    # 创建一个包含所有分析结果的DataFrame进行比较
    comparison_data = []
    
    for result in analysis_results:
        row = {
            'Case': result['case_name'],
            'Total Nodes': result['total_nodes'],
            'Total Edges': result['total_edges'],
            'Source Nodes': result['source_nodes'],
            'Sink Nodes': result['sink_nodes']
        }
        
        # 添加操作类型计数
        for op_type, count in result['op_counts'].items():
            row[f'{op_type} Count'] = count
        
        # 添加执行单元计数
        if result['pipe_counts']:
            for pipe, count in result['pipe_counts'].items():
                row[f'{pipe} Count'] = count
        
        # 添加周期统计
        if result['cycles_stats']:
            row['Avg Cycles'] = result['cycles_stats']['mean']
            row['Total Cycles'] = result['cycles_stats']['sum']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 保存比较数据到CSV文件
    csv_file = f'{output_prefix}_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"比较数据已保存到 {csv_file}")
    
    # 创建操作类型分布饼图
    plt.figure(figsize=(12, 10))
    
    # 选择有代表性的案例进行显示
    if analysis_results:
        first_case = analysis_results[0]
        plt.pie(
            first_case['op_counts'].values(),
            labels=first_case['op_counts'].keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=[
                '#FF9999', '#FF6666', '#99CCFF', '#99FF99', 
                '#FFFF99', '#FF99FF', '#99FFCC', '#FFCC99'
            ][:len(first_case['op_counts'])]
        )
        plt.axis('equal')
        plt.title(f"{first_case['case_name']} - 操作类型分布", fontsize=16)
        pie_file = f'{output_prefix}_op_distribution.png'
        plt.savefig(pie_file, dpi=300, bbox_inches='tight')
        print(f"操作类型分布饼图已保存到 {pie_file}")
    
    # 创建执行单元分布条形图
    if analysis_results and analysis_results[0]['pipe_counts']:
        plt.figure(figsize=(12, 8))
        pipe_data = []
        case_names = []
        
        for result in analysis_results:
            if result['pipe_counts']:
                case_names.append(result['case_name'])
                row = []
                for pipe in set(p for r in analysis_results if r['pipe_counts'] for p in r['pipe_counts'].keys()):
                    row.append(result['pipe_counts'].get(pipe, 0))
                pipe_data.append(row)
        
        if pipe_data:
            pipes = list(set(p for r in analysis_results if r['pipe_counts'] for p in r['pipe_counts'].keys()))
            df_pipes = pd.DataFrame(pipe_data, index=case_names, columns=pipes)
            df_pipes.plot(kind='bar', stacked=True, figsize=(12, 8))
            plt.title('不同案例的执行单元使用分布', fontsize=16)
            plt.xlabel('案例')
            plt.ylabel('节点数量')
            plt.legend(title='执行单元', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            pipe_file = f'{output_prefix}_pipe_comparison.png'
            plt.savefig(pipe_file, dpi=300, bbox_inches='tight')
            print(f"执行单元比较图已保存到 {pipe_file}")

# 主函数
if __name__ == "__main__":
    # JSON文件列表
    json_files = [
        'Matmul_Case0.json',
        'Matmul_Case1.json', 
        'Conv_Case0.json',
        'Conv_Case1.json',
        'FlashAttention_Case0.json',
        'FlashAttention_Case1.json'
    ]
    
    analysis_results = []
    
    # 处理每个文件
    for json_file in json_files:
        try:
            print(f"\n正在处理文件: {json_file}")
            
            # 加载DAG数据
            data = load_dag_from_json(json_file)
            
            # 创建DAG图
            G = create_dag_graph(data)
            
            # 案例名称（从文件名提取）
            case_name = json_file.replace('.json', '')
            
            # 详细分析DAG结构
            analysis = detailed_analysis(G, case_name)
            analysis_results.append(analysis)
            
            # 绘制并保存增强的DAG图
            output_image = f'{case_name}_detailed_visualization.png'
            draw_enhanced_dag(G, case_name, output_image)
            
            print(f"\n{case_name} 分析完成！")
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    # 如果有多个分析结果，创建比较统计图表
    if len(analysis_results) > 1:
        draw_statistics(analysis_results, 'cases_comparison')
    
    print("\n所有文件分析完成！")