import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import matplotlib as mpl
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置更美观的风格
plt.style.use('seaborn-v0_8-pastel')

# 更美观的配色方案
COLORS = {
    'active_reasoning': '#1f77b4',  # 蓝色
    'random_selection': '#ff7f0e',  # 橙色
    'load_balancing': '#2ca02c'     # 绿色
}

MARKERS = {
    'active_reasoning': 'o',
    'random_selection': 's',
    'load_balancing': '^'
}

LABELS = {
    'active_reasoning': '主动推理',
    'random_selection': '随机选择',
    'load_balancing': '负载均衡'
}

def smooth_data(data, window_length=11, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑数据
    
    参数:
    data - 要平滑的数据
    window_length - 窗口长度，必须是正奇数
    polyorder - 多项式阶数，必须小于window_length
    """
    if len(data) < window_length:
        # 如果数据点太少，降低窗口长度
        window_length = min(len(data) - 2 if len(data) > 2 else 1, 5)
        window_length = window_length if window_length % 2 == 1 else window_length + 1
        polyorder = min(polyorder, window_length - 1)
    
    if window_length > 1 and len(data) > window_length:
        try:
            return savgol_filter(data, window_length, polyorder)
        except:
            return data
    return data

def calculate_weighted_performance(df):
    """
    计算加权性能得分
    
    加权公式：
    准确率 × 100 + 规范化时间得分 + 平均吞吐量
    
    规范化时间得分计算方式：
    1. 将时间规范到0-0.3范围内的占比
    2. 超过0.3的部分算作1
    3. 用1减去这个占比
    4. 乘以100得到时间得分（时间越短，得分越高）
    """
    weighted_scores = []
    
    for _, row in df.iterrows():
        # 准确率部分：直接乘以100
        accuracy_score = row['average_batch_accuracy_score_per_batch'] * 100
        
        # 规范化时间得分部分
        time_value = row['single_batch_time_consumption']
        normalized_time = min(time_value / 0.3, 1.0)  # 规范到0-1，超过0.3的算作1
        time_score = (1 - normalized_time) * 100      # 转化为0-100的分数（时间越短分数越高）
        
        # 吞吐量部分：直接使用
        throughput_score = row['avg_throughput_score_per_batch']
        
        # 计算总加权得分
        total_score = accuracy_score + time_score + throughput_score
        weighted_scores.append(total_score)
    
    return weighted_scores

def create_weighted_performance_chart(csv_file, output_dir):
    """
    创建加权性能对比图表
    
    参数:
    csv_file - 输入的CSV文件路径
    output_dir - 输出图表的目录
    """
    # 加载数据
    df = pd.read_csv(csv_file)
    
    # 计算加权性能得分
    df['weighted_performance'] = calculate_weighted_performance(df)
    
    # 为每种算法创建单独的DataFrame
    algorithms = df['method'].unique()
    algo_data = {}
    
    for algo in algorithms:
        algo_data[algo] = df[df['method'] == algo]
    
    # 设置图表样式
    plt.figure(figsize=(12, 8))
    
    # 绘制每种算法的加权性能曲线
    for algo in algorithms:
        data = algo_data[algo]
        x = data['sequence']
        y = data['weighted_performance']
        
        # 平滑数据
        y_smooth = smooth_data(y, window_length=7, polyorder=2)
        
        # 仅显示部分原始数据点 - 每N个点显示一个
        skip_points = max(1, len(x) // 10)  # 大约显示10个点
        
        # 绘制选择性的原始数据点（半透明）
        plt.scatter(x[::skip_points], y[::skip_points], color=COLORS.get(algo), 
                  marker=MARKERS.get(algo, 'o'), s=50, alpha=0.6)
        
        # 绘制平滑曲线
        plt.plot(x, y_smooth, color=COLORS.get(algo), linewidth=3, 
                label=LABELS.get(algo, algo))
    
    # 设置图表标题和轴标签
    plt.title('不同算法的加权性能对比', fontsize=18, fontweight='bold')
    plt.xlabel('序列', fontsize=15)
    plt.ylabel('加权性能得分', fontsize=15)
    
    # 添加图例
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 美化坐标轴
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 保存图表
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, '算法加权性能对比.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成加权性能对比图表: {output_path}")
    
    # 创建加权性能得分的箱型图
    create_performance_boxplot(df, output_dir)
    
    # 返回带有加权性能得分的DataFrame
    return df

def create_performance_boxplot(df, output_dir):
    """
    创建加权性能得分的箱型图
    
    参数:
    df - 带有加权性能得分的DataFrame
    output_dir - 输出图表的目录
    """
    plt.figure(figsize=(10, 8))
    
    # 准备数据
    plot_data = []
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # 将数据转换为适合箱型图的格式
        for _, row in method_data.iterrows():
            plot_data.append({
                'Algorithm': LABELS.get(method, method),
                'Weighted Performance': row['weighted_performance']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 使用seaborn绘制美观的箱型图
    palette = {LABELS.get(k, k): v for k, v in COLORS.items()}
    
    # 创建箱型图
    ax = sns.boxplot(
        x='Algorithm', 
        y='Weighted Performance', 
        data=plot_df, 
        palette=palette, 
        width=0.6,
        showfliers=False  # 不显示异常值点
    )
    
    # 添加抖动点以显示实际数据分布
    sns.stripplot(
        x='Algorithm', 
        y='Weighted Performance', 
        data=plot_df,
        size=5, 
        alpha=0.5,
        palette=palette,
        jitter=True
    )
    
    # 添加均值点和标签
    for i, algorithm in enumerate(plot_df['Algorithm'].unique()):
        algorithm_data = plot_df[plot_df['Algorithm'] == algorithm]['Weighted Performance']
        mean_val = algorithm_data.mean()
        
        # 绘制均值点
        ax.plot(i, mean_val, 'o', color='red', markersize=8)
        
        # 添加均值标签
        ax.annotate(
            f'均值: {mean_val:.2f}',
            xy=(i, mean_val),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
    
    # 设置图表标题和标签
    plt.title('不同算法加权性能得分分布', fontsize=18, fontweight='bold')
    plt.xlabel('算法', fontsize=15)
    plt.ylabel('加权性能得分', fontsize=15)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存图表
    output_path = os.path.join(output_dir, '加权性能得分箱型图.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成加权性能得分箱型图: {output_path}")
    
    # 创建增强版箱型图（包含统计信息）
    create_enhanced_boxplot(df, output_dir)

def create_enhanced_boxplot(df, output_dir):
    """
    创建增强版箱型图，包含更多统计信息
    
    参数:
    df - 带有加权性能得分的DataFrame
    output_dir - 输出图表的目录
    """
    plt.figure(figsize=(12, 10))
    
    # 计算每种算法的统计数据
    stats_data = []
    for method in df['method'].unique():
        method_data = df[df['method'] == method]['weighted_performance']
        
        stats_data.append({
            'Algorithm': LABELS.get(method, method),
            'Mean': method_data.mean(),
            'Median': method_data.median(),
            'Std Dev': method_data.std(),
            'Min': method_data.min(),
            'Max': method_data.max(),
            'Q1': method_data.quantile(0.25),
            'Q3': method_data.quantile(0.75),
            'IQR': method_data.quantile(0.75) - method_data.quantile(0.25)
        })
    
    # 创建主图和子图
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # 主箱型图
    ax_main = fig.add_subplot(gs[0, 0])
    
    # 准备数据
    plot_data = []
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # 将数据转换为适合箱型图的格式
        for _, row in method_data.iterrows():
            plot_data.append({
                'Algorithm': LABELS.get(method, method),
                'Weighted Performance': row['weighted_performance']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 使用seaborn绘制美观的箱型图
    palette = {LABELS.get(k, k): v for k, v in COLORS.items()}
    
    # 创建箱型图
    sns.boxplot(
        x='Algorithm', 
        y='Weighted Performance', 
        data=plot_df, 
        palette=palette, 
        width=0.6,
        showfliers=False,  # 不显示异常值点
        ax=ax_main
    )
    
    # 添加抖动点以显示实际数据分布
    sns.stripplot(
        x='Algorithm', 
        y='Weighted Performance', 
        data=plot_df,
        size=6, 
        alpha=0.5,
        palette=palette,
        jitter=True,
        ax=ax_main
    )
    
    # 添加均值点和标签
    for i, algorithm in enumerate(plot_df['Algorithm'].unique()):
        algorithm_data = plot_df[plot_df['Algorithm'] == algorithm]['Weighted Performance']
        mean_val = algorithm_data.mean()
        
        # 绘制均值点
        ax_main.plot(i, mean_val, 'o', color='red', markersize=10)
        
        # 添加均值标签
        ax_main.annotate(
            f'均值: {mean_val:.2f}',
            xy=(i, mean_val),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
    
    # 设置图表标题和标签
    ax_main.set_title('不同算法加权性能得分分布', fontsize=18, fontweight='bold')
    ax_main.set_xlabel('算法', fontsize=15)
    ax_main.set_ylabel('加权性能得分', fontsize=15)
    
    # 添加网格线
    ax_main.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 美化坐标轴
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # 添加小提琴图
    ax_violin = fig.add_subplot(gs[0, 1])
    sns.violinplot(
        y='Algorithm', 
        x='Weighted Performance', 
        data=plot_df, 
        palette=palette,
        orient='h',
        inner='quartile',
        ax=ax_violin
    )
    ax_violin.set_title('分布密度', fontsize=14)
    ax_violin.set_xlabel('加权性能得分', fontsize=12)
    ax_violin.set_ylabel('')
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    
    # 添加统计表格
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # 创建表格数据
    table_data = []
    for stat in stats_data:
        table_data.append([
            stat['Algorithm'],
            f"{stat['Mean']:.2f}",
            f"{stat['Median']:.2f}",
            f"{stat['Std Dev']:.2f}",
            f"{stat['Min']:.2f}",
            f"{stat['Max']:.2f}",
            f"{stat['Q1']:.2f}",
            f"{stat['Q3']:.2f}",
            f"{stat['IQR']:.2f}"
        ])
    
    columns = ['算法', '均值', '中位数', '标准差', '最小值', '最大值', '25%分位数', '75%分位数', '四分位距']
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)  # 调整表格大小
    
    # 为表头设置颜色
    for i, key in enumerate(columns):
        cell = table[(0, i)]
        cell.set_facecolor('#D8E9F0')
        cell.set_text_props(weight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, '加权性能得分详细分析.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成加权性能得分详细分析图: {output_path}")

def create_performance_components_chart(df, output_dir):
    """
    创建性能组成部分对比图表，展示每种算法的准确率、时间得分和吞吐量
    
    参数:
    df - 带有加权性能得分的DataFrame
    output_dir - 输出图表的目录
    """
    # 计算各组成部分的得分
    df['accuracy_component'] = df['average_batch_accuracy_score_per_batch'] * 100
    df['time_component'] = df['single_batch_time_consumption'].apply(
        lambda x: (1 - min(x / 0.3, 1.0)) * 100
    )
    df['throughput_component'] = df['avg_throughput_score_per_batch']
    
    # 为每种算法创建单独的DataFrame
    algorithms = df['method'].unique()
    algo_data = {}
    
    for algo in algorithms:
        algo_data[algo] = df[df['method'] == algo]
    
    # 创建包含两个子图的图表（准确率和时间得分）
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    components = ['accuracy_component', 'time_component']
    titles = ['准确率得分 (×100)', '时间得分 (规范化后)']
    
    for i, (component, title) in enumerate(zip(components, titles)):
        ax = axes[i]
        
        for algo in algorithms:
            data = algo_data[algo]
            x = data['sequence']
            y = data[component]
            
            # 平滑数据
            y_smooth = smooth_data(y, window_length=7, polyorder=2)
            
            # 仅显示部分原始数据点 - 每N个点显示一个
            skip_points = max(1, len(x) // 10)  # 大约显示10个点
            
            # 绘制选择性的原始数据点（半透明）
            ax.scatter(x[::skip_points], y[::skip_points], color=COLORS.get(algo), 
                     marker=MARKERS.get(algo, 'o'), s=50, alpha=0.6)
            
            # 绘制平滑曲线
            ax.plot(x, y_smooth, color=COLORS.get(algo), linewidth=3, 
                   label=LABELS.get(algo, algo))
        
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_ylabel('得分', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=12, loc='best', framealpha=0.7)
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 添加x轴标签
    axes[1].set_xlabel('序列', fontsize=14)
    
    # 添加总标题
    fig.suptitle('性能指标对比: 准确率与时间', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图表
    output_path = os.path.join(output_dir, '准确率与时间得分对比.png')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成准确率与时间得分对比图表: {output_path}")
    
    # 创建单独的吞吐量图表
    create_throughput_chart(df, output_dir)

def create_throughput_chart(df, output_dir):
    """
    创建单独的吞吐量对比图表
    
    参数:
    df - 带有加权性能得分的DataFrame
    output_dir - 输出图表的目录
    """
    # 为每种算法创建单独的DataFrame
    algorithms = df['method'].unique()
    algo_data = {}
    
    for algo in algorithms:
        algo_data[algo] = df[df['method'] == algo]
    
    # 设置图表样式
    plt.figure(figsize=(12, 8))
    
    # 绘制每种算法的吞吐量曲线
    for algo in algorithms:
        data = algo_data[algo]
        x = data['sequence']
        y = data['avg_throughput_score_per_batch']
        
        # 平滑数据 - 增加窗口长度使曲线更平滑
        y_smooth = smooth_data(y, window_length=9, polyorder=2)
        
        # 仅显示部分原始数据点 - 每N个点显示一个
        skip_points = max(1, len(x) // 10)  # 大约显示10个点
        
        # 绘制选择性的原始数据点（半透明）
        plt.scatter(x[::skip_points], y[::skip_points], color=COLORS.get(algo), 
                  marker=MARKERS.get(algo, 'o'), s=50, alpha=0.6)
        
        # 绘制平滑曲线
        plt.plot(x, y_smooth, color=COLORS.get(algo), linewidth=3, 
                label=LABELS.get(algo, algo))
    
    # 设置图表标题和轴标签
    plt.title('不同算法的吞吐量得分对比', fontsize=18, fontweight='bold')
    plt.xlabel('序列', fontsize=15)
    plt.ylabel('吞吐量得分', fontsize=15)
    
    # 添加图例
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 美化坐标轴
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 保存图表
    output_path = os.path.join(output_dir, '吞吐量得分对比.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成吞吐量得分对比图表: {output_path}")
    
    # 创建综合比较图 - 所有三个指标在一张图上
    create_combined_metrics_chart(df, output_dir)

def create_combined_metrics_chart(df, output_dir):
    """
    创建三个指标的综合比较图表
    
    参数:
    df - 带有加权性能得分的DataFrame
    output_dir - 输出图表的目录
    """
    # 计算各算法的平均指标值
    summary_data = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        summary_data.append({
            'method': method,
            'accuracy': method_data['accuracy_component'].mean(),
            'time_score': method_data['time_component'].mean(),
            'throughput': method_data['throughput_component'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置条形图的位置
    bar_width = 0.25
    index = np.arange(len(summary_df))
    
    # 绘制三组条形图，使用更鲜明的颜色
    bars1 = ax.bar(index - bar_width, summary_df['accuracy'], bar_width, 
                  label='准确率得分', color='#3274A1', alpha=0.85)
    bars2 = ax.bar(index, summary_df['time_score'], bar_width, 
                  label='时间得分', color='#E1812C', alpha=0.85)
    bars3 = ax.bar(index + bar_width, summary_df['throughput'], bar_width, 
                  label='吞吐量得分', color='#3A923A', alpha=0.85)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3点垂直偏移
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # 设置坐标轴
    ax.set_ylabel('平均得分', fontsize=14)
    ax.set_title('不同算法的性能指标综合对比', fontsize=18, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels([LABELS.get(m, m) for m in summary_df['method']], fontsize=12)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存图表
    output_path = os.path.join(output_dir, '性能指标综合对比.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成性能指标综合对比图表: {output_path}")

# 主程序
if __name__ == "__main__":
    # 定义输入和输出路径
    input_csv = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_17/93/combined_algorithm_data.csv"  # 请替换为您的CSV文件路径
    output_directory = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_17/93/cn"    # 输出目录
    
    # 创建加权性能对比图表
    df_with_scores = create_weighted_performance_chart(input_csv, output_directory)
    
    # 创建性能组成部分对比图表
    create_performance_components_chart(df_with_scores, output_directory)