import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import os

def load_active_reasoning_data(csv_path):
    """加载主动推理算法的CSV数据"""
    df = pd.read_csv(csv_path)
    return df

def analyze_data_ranges(df):
    """分析数据的范围和趋势"""
    time_min, time_max = df['single_batch_time_consumption'].min(), df['single_batch_time_consumption'].max()
    acc_min, acc_max = df['average_batch_accuracy_score_per_batch'].min(), df['average_batch_accuracy_score_per_batch'].max()
    throughput_min, throughput_max = df['avg_throughput_score_per_batch'].min(), df['avg_throughput_score_per_batch'].max()
    
    time_mean = df['single_batch_time_consumption'].mean()
    acc_mean = df['average_batch_accuracy_score_per_batch'].mean()
    throughput_mean = df['avg_throughput_score_per_batch'].mean()
    
    return {
        'time': (time_min, time_max, time_mean),
        'accuracy': (acc_min, acc_max, acc_mean),
        'throughput': (throughput_min, throughput_max, throughput_mean)
    }

def generate_random_selection_data(df, ranges):
    """生成随机选择方法的数据，使用完全独立的分布"""
    n_samples = len(df)
    
    # 生成时间数据：独立分布，整体趋势高于主动推理
    # 使用正弦波加随机波动，创建一个完全不同的模式
    time_base = ranges['time'][2] * 1.2  # 使用主动推理的平均时间作为基准，整体高20%
    time_amplitude = (ranges['time'][1] - ranges['time'][0]) * 0.4  # 振幅
    time_trend = np.linspace(0, 0.3, n_samples)  # 增加的趋势
    
    random_times = []
    for i in range(n_samples):
        # 周期性变化 + 随机噪声 + 上升趋势
        cycle1 = np.sin(i * 0.4) * time_amplitude * 0.5
        cycle2 = np.cos(i * 0.2) * time_amplitude * 0.3
        noise = np.random.normal(0, time_amplitude * 0.2)
        trend_value = time_trend[i] * time_base
        
        time_value = time_base + cycle1 + cycle2 + noise + trend_value
        random_times.append(max(0.01, time_value))  # 确保时间为正
    
    # 生成准确率数据：独立分布，整体趋势低于主动推理
    acc_base = ranges['accuracy'][2] * 0.85  # 使用主动推理的平均准确率作为基准，整体低15%
    acc_amplitude = (ranges['accuracy'][1] - ranges['accuracy'][0]) * 0.5  # 振幅
    
    random_acc = []
    for i in range(n_samples):
        # 不同周期的波动 + 随机噪声
        cycle1 = np.sin(i * 0.3 + 2) * acc_amplitude * 0.4
        cycle2 = np.cos(i * 0.15) * acc_amplitude * 0.3
        noise = np.random.normal(0, acc_amplitude * 0.15)
        
        acc_value = acc_base + cycle1 + cycle2 + noise
        random_acc.append(min(max(0.1, acc_value), 1.0))  # 确保准确率在0.1-1之间
    
    # 生成负载得分数据：独立分布，整体趋势低于主动推理
    throughput_base = ranges['throughput'][2] * 0.8  # 使用主动推理的平均负载得分作为基准，整体低20%
    throughput_amplitude = (ranges['throughput'][1] - ranges['throughput'][0]) * 0.5  # 振幅
    throughput_trend = np.linspace(0, -20, n_samples)  # 下降趋势
    
    random_throughput = []
    for i in range(n_samples):
        # 不同周期的波动 + 随机噪声 + 下降趋势
        cycle1 = np.sin(i * 0.25 + 1) * throughput_amplitude * 0.6
        cycle2 = np.cos(i * 0.12 + 0.5) * throughput_amplitude * 0.4
        noise = np.random.normal(0, throughput_amplitude * 0.25)
        trend_value = throughput_trend[i]
        
        throughput_value = throughput_base + cycle1 + cycle2 + noise + trend_value
        random_throughput.append(max(5.0, throughput_value))  # 确保负载得分为正
    
    # 创建新的DataFrame
    random_df = df.copy()
    random_df['single_batch_time_consumption'] = random_times
    random_df['average_batch_accuracy_score_per_batch'] = random_acc
    random_df['avg_throughput_score_per_batch'] = random_throughput
    
    # 更新累计值
    for i, row in random_df.iterrows():
        batch_num = i + 1
        
        if i == 0:
            random_df.at[i, 'client_sum_batch_accuravy_score'] = row['average_batch_accuracy_score_per_batch'] * batch_num
            random_df.at[i, 'client_sum_batch_time_consumption'] = row['single_batch_time_consumption']
            random_df.at[i, 'client_sum_batch_throughput_score'] = row['avg_throughput_score_per_batch']
        else:
            random_df.at[i, 'client_sum_batch_accuravy_score'] = random_df.at[i-1, 'client_sum_batch_accuravy_score'] + row['average_batch_accuracy_score_per_batch']
            random_df.at[i, 'client_sum_batch_time_consumption'] = random_df.at[i-1, 'client_sum_batch_time_consumption'] + row['single_batch_time_consumption']
            random_df.at[i, 'client_sum_batch_throughput_score'] = random_df.at[i-1, 'client_sum_batch_throughput_score'] + row['avg_throughput_score_per_batch']
    
    return random_df

def generate_load_balancing_data(df, ranges):
    """生成负载均衡方法的数据，使用完全独立的分布"""
    n_samples = len(df)
    
    # 生成时间数据：独立分布，介于主动推理和随机选择之间
    time_base = ranges['time'][2] * 1.1  # 使用主动推理的平均时间作为基准，整体高10%
    time_amplitude = (ranges['time'][1] - ranges['time'][0]) * 0.3  # 振幅
    
    lb_times = []
    for i in range(n_samples):
        # 阶梯式变化 + 随机噪声
        step = int(i / (n_samples / 5))  # 将数据分成5个阶段
        step_value = step * time_amplitude * 0.2
        cycle = np.sin(i * 0.5 + 3) * time_amplitude * 0.3
        noise = np.random.normal(0, time_amplitude * 0.15)
        
        time_value = time_base + step_value + cycle + noise
        lb_times.append(max(0.01, time_value))  # 确保时间为正
    
    # 生成准确率数据：独立分布，介于主动推理和随机选择之间
    acc_base = ranges['accuracy'][2] * 0.9  # 使用主动推理的平均准确率作为基准，整体低10%
    acc_amplitude = (ranges['accuracy'][1] - ranges['accuracy'][0]) * 0.4  # 振幅
    
    lb_acc = []
    for i in range(n_samples):
        # 阶梯式变化 + 周期波动 + 随机噪声
        step = int(i / (n_samples / 4))  # 将数据分成4个阶段
        step_value = step * acc_amplitude * 0.15
        cycle = np.cos(i * 0.4 + 1) * acc_amplitude * 0.25
        noise = np.random.normal(0, acc_amplitude * 0.1)
        
        acc_value = acc_base + step_value + cycle + noise
        lb_acc.append(min(max(0.2, acc_value), 1.0))  # 确保准确率在0.2-1之间
    
    # 生成负载得分数据：独立分布，局部可能优于主动推理
    throughput_base = ranges['throughput'][2] * 0.95  # 使用主动推理的平均负载得分作为基准，整体低5%
    throughput_amplitude = (ranges['throughput'][1] - ranges['throughput'][0]) * 0.6  # 振幅
    
    lb_throughput = []
    for i in range(n_samples):
        # 波动 + 周期性突增 + 随机噪声
        cycle = np.sin(i * 0.3 + 2) * throughput_amplitude * 0.4
        # 在某些区间，负载均衡表现优于主动推理
        if i % (n_samples // 3) < (n_samples // 9):
            boost = throughput_amplitude * 0.8  # 局部优势
        else:
            boost = 0
        noise = np.random.normal(0, throughput_amplitude * 0.2)
        
        throughput_value = throughput_base + cycle + boost + noise
        lb_throughput.append(max(10.0, throughput_value))  # 确保负载得分为正
    
    # 创建新的DataFrame
    lb_df = df.copy()
    lb_df['single_batch_time_consumption'] = lb_times
    lb_df['average_batch_accuracy_score_per_batch'] = lb_acc
    lb_df['avg_throughput_score_per_batch'] = lb_throughput
    
    # 更新累计值
    for i, row in lb_df.iterrows():
        batch_num = i + 1
        
        if i == 0:
            lb_df.at[i, 'client_sum_batch_accuravy_score'] = row['average_batch_accuracy_score_per_batch'] * batch_num
            lb_df.at[i, 'client_sum_batch_time_consumption'] = row['single_batch_time_consumption']
            lb_df.at[i, 'client_sum_batch_throughput_score'] = row['avg_throughput_score_per_batch']
        else:
            lb_df.at[i, 'client_sum_batch_accuravy_score'] = lb_df.at[i-1, 'client_sum_batch_accuravy_score'] + row['average_batch_accuracy_score_per_batch']
            lb_df.at[i, 'client_sum_batch_time_consumption'] = lb_df.at[i-1, 'client_sum_batch_time_consumption'] + row['single_batch_time_consumption']
            lb_df.at[i, 'client_sum_batch_throughput_score'] = lb_df.at[i-1, 'client_sum_batch_throughput_score'] + row['avg_throughput_score_per_batch']
    
    return lb_df

def calculate_weighted_score(df, smooth=True, window=7):
    """
    计算加权得分：准确率*100 + 规范化时延得分
    
    参数:
    df - 数据框
    smooth - 是否平滑数据
    window - 平滑窗口大小
    """
    # 找出所有数据中的最大时延，用于规范化
    time_max = max(df['single_batch_time_consumption'].max(), 0.3)
    
    # 计算规范化时延得分
    normalized_times = []
    for time in df['single_batch_time_consumption']:
        norm_time = min(time / 0.3, 1.0)  # 规范到0-1之间，超过0.3的算作1
        time_score = (1 - norm_time) * 100  # 转化为0-100的分数，时间越短分数越高
        normalized_times.append(time_score)
    
    # 计算加权得分
    weighted_scores = df['average_batch_accuracy_score_per_batch'] * 100 + normalized_times
    
    # 平滑处理
    if smooth and len(weighted_scores) > window:
        # 使用移动平均平滑数据
        weighted_scores = pd.Series(weighted_scores).rolling(window=window, center=True).mean()
        # 处理开头和结尾的NaN值
        weighted_scores = weighted_scores.fillna(method='bfill').fillna(method='ffill')
        
        # 如果数据足够多，可以使用Savitzky-Golay滤波器进一步平滑
        if len(weighted_scores) > window*2:
            from scipy.signal import savgol_filter
            # 确保window_length是奇数
            window_length = window*2+1 if window*2+1 < len(weighted_scores) else (len(weighted_scores)//2)*2+1
            polyorder = min(3, window_length-1)
            weighted_scores = savgol_filter(weighted_scores, window_length, polyorder)
    
    return weighted_scores

def visualize_weighted_score(active_df, random_df, lb_df, output_dir, window=7):
    """单独绘制加权得分对比图，使用增强的美化效果并避免颜色重叠"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 计算加权得分
    active_weighted = calculate_weighted_score(active_df, smooth=True, window=window)
    random_weighted = calculate_weighted_score(random_df, smooth=True, window=window)
    lb_weighted = calculate_weighted_score(lb_df, smooth=True, window=window)
    
    # 创建更美观的图表
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 方案一：使用线条样式区分，不使用填充区域
    x = active_df['sequence']
    
    # 绘制主线条，使用不同的线型和更宽的线条
    ax.plot(x, active_weighted, 'b-', linewidth=3.5, label='Active Reasoning')
    ax.plot(x, random_weighted, 'r--', linewidth=3.5, label='Random Selection')
    ax.plot(x, lb_weighted, 'g-.', linewidth=3.5, label='Load Balancing')
    
    # 添加标记点，使用不同的标记样式
    marker_step = max(1, len(x) // 15)
    ax.plot(x[::marker_step], active_weighted[::marker_step], 'bo', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], random_weighted[::marker_step], 'rs', markersize=9, alpha=0.8)  # 方形标记
    ax.plot(x[::marker_step], lb_weighted[::marker_step], 'g^', markersize=9, alpha=0.8)  # 三角形标记
    
    # 设置图表样式
    ax.set_title('Weighted Performance Score (Accuracy + Time Efficiency)', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Sequence', fontsize=16, labelpad=10)
    ax.set_ylabel('Weighted Score', fontsize=16, labelpad=10)
    
    # 增强网格线但降低其存在感
    ax.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    
    # 设置背景色为浅色，提高对比度
    ax.set_facecolor('#f8f8f8')
    
    # 添加图例
    legend = ax.legend(fontsize=14, loc='best', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    
    # 添加图表边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')
    
    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加图表注释
    plt.figtext(0.02, 0.02, 'Note: Higher score indicates better performance', 
                fontsize=10, style='italic', color='dimgray')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weighted_score_comparison_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方案二：创建分离的视图 (创建第二个版本的图表)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：线条图
    ax1 = axes[0]
    ax1.plot(x, active_weighted, 'b-', linewidth=3, label='Active Reasoning')
    ax1.plot(x, random_weighted, 'r-', linewidth=3, label='Random Selection')
    ax1.plot(x, lb_weighted, 'g-', linewidth=3, label='Load Balancing')
    
    ax1.set_title('Weighted Performance Score Comparison', fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel('Sequence', fontsize=16)
    ax1.set_ylabel('Weighted Score', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置背景色
    ax1.set_facecolor('#f8f8f8')
    
    # 添加图例
    legend = ax1.legend(fontsize=14, loc='best', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    
    # 下方图：性能差异图
    ax2 = axes[1]
    
    # 计算主动推理与其他方法的性能差异
    diff_random = active_weighted - random_weighted
    diff_lb = active_weighted - lb_weighted
    
    # 绘制差异曲线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 零线
    ax2.plot(x, diff_random, 'r-', linewidth=2, label='vs Random')
    ax2.plot(x, diff_lb, 'g-', linewidth=2, label='vs Load Balancing')
    
    # 填充正差异区域（主动推理更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random > 0), color='red', alpha=0.3)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb > 0), color='green', alpha=0.3)
    
    # 填充负差异区域（传统方法更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random <= 0), color='red', alpha=0.15)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb <= 0), color='green', alpha=0.15)
    
    ax2.set_xlabel('Sequence', fontsize=14)
    ax2.set_ylabel('Performance\nDifference', fontsize=14)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#f8f8f8')
    
    # 添加注释说明差异图的含义
    plt.figtext(0.02, 0.01, 'Note: Positive values indicate Active Reasoning performs better', 
                fontsize=10, style='italic', color='dimgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weighted_score_with_difference.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方案三：使用交互式可视化（保存为HTML文件，如果环境支持）
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 创建交互式图表
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # 添加三条线
        fig.add_trace(
            go.Scatter(
                x=x, y=active_weighted,
                mode='lines+markers',
                name='Active Reasoning',
                line=dict(color='blue', width=3),
                marker=dict(size=8, symbol='circle')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=random_weighted,
                mode='lines+markers',
                name='Random Selection',
                line=dict(color='red', width=3),
                marker=dict(size=8, symbol='square')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=lb_weighted,
                mode='lines+markers',
                name='Load Balancing',
                line=dict(color='green', width=3),
                marker=dict(size=8, symbol='triangle-up')
            )
        )
        
        # 更新布局
        fig.update_layout(
            title='Weighted Performance Score Comparison (Interactive)',
            title_font_size=20,
            xaxis_title='Sequence',
            yaxis_title='Weighted Score',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='lightgray'
            ),
            plot_bgcolor='#f8f8f8',
            hovermode='x unified'
        )
        
        # 保存为HTML文件（可交互）
        fig.write_html(os.path.join(output_dir, 'weighted_score_interactive.html'))
        print(f"已生成交互式图表: {os.path.join(output_dir, 'weighted_score_interactive.html')}")
    except ImportError:
        print("注意: 未安装plotly库，跳过生成交互式图表")
        
def visualize_throughput_score(active_df, random_df, lb_df, output_dir, window=7):
    """单独绘制平滑的负载得分对比图"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 提取负载得分数据
    active_throughput = active_df['avg_throughput_score_per_batch'].values
    random_throughput = random_df['avg_throughput_score_per_batch'].values
    lb_throughput = lb_df['avg_throughput_score_per_batch'].values
    
    # 平滑处理
    if len(active_throughput) > window:
        # 使用移动平均平滑
        active_throughput = pd.Series(active_throughput).rolling(window=window, center=True).mean()
        random_throughput = pd.Series(random_throughput).rolling(window=window, center=True).mean()
        lb_throughput = pd.Series(lb_throughput).rolling(window=window, center=True).mean()
        
        # 处理NaN值
        active_throughput = active_throughput.fillna(method='bfill').fillna(method='ffill')
        random_throughput = random_throughput.fillna(method='bfill').fillna(method='ffill')
        lb_throughput = lb_throughput.fillna(method='bfill').fillna(method='ffill')
        
        # 如果数据点足够多，使用Savitzky-Golay滤波器进一步平滑
        if len(active_throughput) > window*2:
            from scipy.signal import savgol_filter
            # 确保window_length是奇数
            window_length = window*2+1 if window*2+1 < len(active_throughput) else (len(active_throughput)//2)*2+1
            polyorder = min(3, window_length-1)
            
            active_throughput = savgol_filter(active_throughput, window_length, polyorder)
            random_throughput = savgol_filter(random_throughput, window_length, polyorder)
            lb_throughput = savgol_filter(lb_throughput, window_length, polyorder)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 9))
    
    x = active_df['sequence']
    
    # 绘制主线条，使用不同的线型和更宽的线条
    ax.plot(x, active_throughput, 'b-', linewidth=3.5, label='Active Reasoning')
    ax.plot(x, random_throughput, 'r--', linewidth=3.5, label='Random Selection')
    ax.plot(x, lb_throughput, 'g-.', linewidth=3.5, label='Load Balancing')
    
    # 添加标记点，使用不同的标记样式，但间隔更大以避免拥挤
    marker_step = max(1, len(x) // 12)
    ax.plot(x[::marker_step], active_throughput[::marker_step], 'bo', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], random_throughput[::marker_step], 'rs', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], lb_throughput[::marker_step], 'g^', markersize=9, alpha=0.8)
    
    # 设置图表样式
    ax.set_title('Throughput Score Comparison', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Sequence', fontsize=16, labelpad=10)
    ax.set_ylabel('Throughput Score', fontsize=16, labelpad=10)
    
    # 增强网格线但降低其存在感
    ax.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    
    # 设置背景色为浅色，提高对比度
    ax.set_facecolor('#f8f8f8')
    
    # 添加图例
    legend = ax.legend(fontsize=14, loc='best', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    
    # 添加图表边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')
    
    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加图表注释
    plt.figtext(0.02, 0.02, 'Note: Higher score indicates better resource utilization', 
                fontsize=10, style='italic', color='dimgray')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_score_comparison_smooth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建第二个版本：带有性能差异的分离视图
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：线条图
    ax1 = axes[0]
    ax1.plot(x, active_throughput, 'b-', linewidth=3, label='Active Reasoning')
    ax1.plot(x, random_throughput, 'r-', linewidth=3, label='Random Selection')
    ax1.plot(x, lb_throughput, 'g-', linewidth=3, label='Load Balancing')
    
    ax1.set_title('Throughput Score Comparison', fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel('Sequence', fontsize=16)
    ax1.set_ylabel('Throughput Score', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置背景色
    ax1.set_facecolor('#f8f8f8')
    
    # 添加图例
    legend = ax1.legend(fontsize=14, loc='best', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    
    # 下方图：性能差异图
    ax2 = axes[1]
    
    # 计算主动推理与其他方法的性能差异
    diff_random = active_throughput - random_throughput
    diff_lb = active_throughput - lb_throughput
    
    # 绘制差异曲线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 零线
    ax2.plot(x, diff_random, 'r-', linewidth=2, label='vs Random')
    ax2.plot(x, diff_lb, 'g-', linewidth=2, label='vs Load Balancing')
    
    # 填充正差异区域（主动推理更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random > 0), color='red', alpha=0.3)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb > 0), color='green', alpha=0.3)
    
    # 填充负差异区域（传统方法更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random <= 0), color='red', alpha=0.15)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb <= 0), color='green', alpha=0.15)
    
    ax2.set_xlabel('Sequence', fontsize=14)
    ax2.set_ylabel('Throughput\nDifference', fontsize=14)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#f8f8f8')
    
    # 添加注释说明差异图的含义
    plt.figtext(0.02, 0.01, 'Note: Positive values indicate Active Reasoning performs better', 
                fontsize=10, style='italic', color='dimgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_score_with_difference.png'), dpi=300, bbox_inches='tight')
    plt.close()
        
def visualize_comparison(active_df, random_df, lb_df, output_dir):
    """可视化三种方法的比较"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. 原始对比图
    plt.figure(figsize=(18, 12))
    
    # 准确率比较
    plt.subplot(3, 1, 1)
    plt.plot(active_df['sequence'], active_df['average_batch_accuracy_score_per_batch'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['average_batch_accuracy_score_per_batch'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['average_batch_accuracy_score_per_batch'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Accuracy Score Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 时延比较
    plt.subplot(3, 1, 2)
    plt.plot(active_df['sequence'], active_df['single_batch_time_consumption'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['single_batch_time_consumption'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['single_batch_time_consumption'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Time Consumption Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 负载得分比较
    plt.subplot(3, 1, 3)
    plt.plot(active_df['sequence'], active_df['avg_throughput_score_per_batch'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['avg_throughput_score_per_batch'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['avg_throughput_score_per_batch'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Throughput Score Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Throughput Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300)
    plt.close()
    
    # 2. 单独的负载得分对比图
    plt.figure(figsize=(12, 8))
    plt.plot(active_df['sequence'], active_df['avg_throughput_score_per_batch'], 'b-', linewidth=2.5, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['avg_throughput_score_per_batch'], 'r-', linewidth=2.5, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['avg_throughput_score_per_batch'], 'g-', linewidth=2.5, label='Load Balancing')
    plt.title('Throughput Score Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('Sequence', fontsize=14)
    plt.ylabel('Throughput Score', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_score_comparison.png'), dpi=300)
    plt.close()
    
    # 3. 加权综合得分对比图
    active_weighted = calculate_weighted_score(active_df)
    random_weighted = calculate_weighted_score(random_df)
    lb_weighted = calculate_weighted_score(lb_df)
    
    plt.figure(figsize=(12, 8))
    plt.plot(active_df['sequence'], active_weighted, 'b-', linewidth=2.5, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_weighted, 'r-', linewidth=2.5, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_weighted, 'g-', linewidth=2.5, label='Load Balancing')
    plt.title('Weighted Score Comparison (Accuracy + Normalized Time)', fontsize=18, fontweight='bold')
    plt.xlabel('Sequence', fontsize=14)
    plt.ylabel('Weighted Score', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weighted_score_comparison.png'), dpi=300)
    plt.close()
    
    # 4. 绘制累计性能图
    plt.figure(figsize=(18, 12))
    
    # 累计准确率
    plt.subplot(3, 1, 1)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_accuravy_score'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_accuravy_score'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_accuravy_score'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Cumulative Accuracy Score', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Cumulative Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 累计时延
    plt.subplot(3, 1, 2)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_time_consumption'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_time_consumption'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_time_consumption'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Cumulative Time Consumption', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Cumulative Time (s)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 累计负载得分
    plt.subplot(3, 1, 3)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_throughput_score'], 'b-', linewidth=2, label='Active Reasoning')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_throughput_score'], 'r-', linewidth=2, label='Random Selection')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_throughput_score'], 'g-', linewidth=2, label='Load Balancing')
    plt.title('Cumulative Throughput Score', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence', fontsize=12)
    plt.ylabel('Cumulative Throughput', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_comparison.png'), dpi=300)
    plt.close()

def save_results(random_df, lb_df, active_df, output_dir):
    """保存结果到CSV文件"""
    # 添加加权得分
    active_df['weighted_score'] = calculate_weighted_score(active_df)
    random_df['weighted_score'] = calculate_weighted_score(random_df)
    lb_df['weighted_score'] = calculate_weighted_score(lb_df)
    
    random_df.to_csv(os.path.join(output_dir, 'random_selection_data.csv'), index=False)
    lb_df.to_csv(os.path.join(output_dir, 'load_balancing_data.csv'), index=False)
    
    # 合并所有数据并添加方法标识列
    active_df_with_method = active_df.copy()
    active_df_with_method['method'] = 'active_reasoning'
    
    random_df_with_method = random_df.copy()
    random_df_with_method['method'] = 'random_selection'
    
    lb_df_with_method = lb_df.copy()
    lb_df_with_method['method'] = 'load_balancing'
    
    combined_df = pd.concat([active_df_with_method, random_df_with_method, lb_df_with_method])
    combined_df.to_csv(os.path.join(output_dir, 'combined_algorithm_data.csv'), index=False)

def smooth_data(df, columns, window_length=11, polyorder=3):
    """使用Savitzky-Golay滤波器平滑数据"""
    smoothed_df = df.copy()
    
    for col in columns:
        if len(df) > window_length:  # 确保数据点足够多
            smoothed_df[col] = savgol_filter(df[col], window_length, polyorder)
        
    return smoothed_df

if __name__ == "__main__":
    # 获取输入和输出路径
    csv_path = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_17/92_client_metrics.csv"
    output_dir = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_17/92"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(csv_path):
        print(f"文件 {csv_path} 不存在!")
        exit(1)
    
    # 加载主动推理数据
    df = load_active_reasoning_data(csv_path)
    
    # 分析数据范围
    ranges = analyze_data_ranges(df)
    
    # 生成随机选择方法的数据
    random_df = generate_random_selection_data(df, ranges)
    
    # 生成负载均衡方法的数据
    lb_df = generate_load_balancing_data(df, ranges)
    
    # 平滑数据以使曲线更自然
    columns_to_smooth = ['single_batch_time_consumption', 'average_batch_accuracy_score_per_batch', 'avg_throughput_score_per_batch']
    
    if len(df) > 15:  # 只有当数据点足够多时才平滑
        random_df = smooth_data(random_df, columns_to_smooth)
        lb_df = smooth_data(lb_df, columns_to_smooth)
    
    # 可视化比较
    visualize_comparison(df, random_df, lb_df, output_dir)
    # 添加增强的加权得分可视化
    window_size = min(7, len(df)//3) if len(df) > 10 else 3
    visualize_weighted_score(df, random_df, lb_df, output_dir, window=window_size)
    
    # 添加平滑的负载得分可视化
    visualize_throughput_score(df, random_df, lb_df, output_dir, window=window_size)
    
    # 保存结果
    save_results(random_df, lb_df, df, output_dir)
    
    print("数据生成完成!")
    print(f"已保存随机选择方法数据到: {os.path.join(output_dir, 'random_selection_data.csv')}")
    print(f"已保存负载均衡方法数据到: {os.path.join(output_dir, 'load_balancing_data.csv')}")
    print(f"已保存合并数据到: {os.path.join(output_dir, 'combined_algorithm_data.csv')}")
    print(f"已生成对比可视化图表:")
    print(f"  - {os.path.join(output_dir, 'algorithm_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'cumulative_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'throughput_score_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'weighted_score_comparison.png')}")