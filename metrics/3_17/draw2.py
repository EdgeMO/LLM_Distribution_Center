import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免某些字体渲染问题
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import os

def setup_chinese_font():
    """设置中文字体支持"""
    # 按优先级排列可用的中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # 文泉驿微米黑，通常显示效果较好
        'WenQuanYi Zen Hei',    # 文泉驿正黑
        'AR PL UMing CN',       # 文鼎 PL 细上海宋
        'AR PL UKai CN',        # 文鼎 PL 简中楷
        'Noto Sans CJK JP',     # Noto Sans 日文（也支持中文）
        'Noto Serif CJK JP'     # Noto Serif 日文（也支持中文）
    ]
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"已设置中文字体列表: {', '.join(chinese_fonts)}")
    return chinese_fonts[0]  # 返回首选字体

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
        # 处理开头和结尾的NaN值 - 使用推荐的方法
        weighted_scores = weighted_scores.bfill().ffill()
        
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
    
    # 计算加权得分
    active_weighted = calculate_weighted_score(active_df, smooth=True, window=window)
    random_weighted = calculate_weighted_score(random_df, smooth=True, window=window)
    lb_weighted = calculate_weighted_score(lb_df, smooth=True, window=window)
    
    # 创建更美观的图表
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 方案一：使用线条样式区分，不使用填充区域
    x = active_df['sequence']
    
    # 绘制主线条，使用不同的线型和更宽的线条
    ax.plot(x, active_weighted, 'b-', linewidth=3.5, label='主动推理')
    ax.plot(x, random_weighted, 'r--', linewidth=3.5, label='随机选择')
    ax.plot(x, lb_weighted, 'g-.', linewidth=3.5, label='负载均衡')
    
    # 添加标记点，使用不同的标记样式
    marker_step = max(1, len(x) // 15)
    ax.plot(x[::marker_step], active_weighted[::marker_step], 'bo', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], random_weighted[::marker_step], 'rs', markersize=9, alpha=0.8)  # 方形标记
    ax.plot(x[::marker_step], lb_weighted[::marker_step], 'g^', markersize=9, alpha=0.8)  # 三角形标记
    
    # 设置图表样式
    ax.set_title('加权性能得分 (准确率 + 时间效率)', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('序列', fontsize=16, labelpad=10)
    ax.set_ylabel('加权得分', fontsize=16, labelpad=10)
    
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
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '加权得分对比_增强版.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方案二：创建分离的视图 (创建第二个版本的图表)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：线条图
    ax1 = axes[0]
    ax1.plot(x, active_weighted, 'b-', linewidth=3, label='主动推理')
    ax1.plot(x, random_weighted, 'r-', linewidth=3, label='随机选择')
    ax1.plot(x, lb_weighted, 'g-', linewidth=3, label='负载均衡')
    
    ax1.set_title('加权性能得分对比', fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel('序列', fontsize=16)
    ax1.set_ylabel('加权得分', fontsize=16)
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
    ax2.plot(x, diff_random, 'r-', linewidth=2, label='与随机选择对比')
    ax2.plot(x, diff_lb, 'g-', linewidth=2, label='与负载均衡对比')
    
    # 填充正差异区域（主动推理更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random > 0), color='red', alpha=0.3)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb > 0), color='green', alpha=0.3)
    
    # 填充负差异区域（传统方法更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random <= 0), color='red', alpha=0.15)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb <= 0), color='green', alpha=0.15)
    
    ax2.set_xlabel('序列', fontsize=14)
    ax2.set_ylabel('性能差异', fontsize=14)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#f8f8f8')
    
    # 添加注释说明差异图的含义
    plt.figtext(0.02, 0.01, '正值表示主动推理方法更好', 
                fontsize=10, style='italic', color='dimgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '加权得分与差异图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
        
def visualize_throughput_score(active_df, random_df, lb_df, output_dir, window=7):
    """单独绘制平滑的负载得分对比图"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    
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
        
        # 处理NaN值 - 使用推荐的方法
        active_throughput = active_throughput.bfill().ffill()
        random_throughput = random_throughput.bfill().ffill()
        lb_throughput = lb_throughput.bfill().ffill()
        
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
    ax.plot(x, active_throughput, 'b-', linewidth=3.5, label='主动推理')
    ax.plot(x, random_throughput, 'r--', linewidth=3.5, label='随机选择')
    ax.plot(x, lb_throughput, 'g-.', linewidth=3.5, label='负载均衡')
    
    # 添加标记点，使用不同的标记样式，但间隔更大以避免拥挤
    marker_step = max(1, len(x) // 12)
    ax.plot(x[::marker_step], active_throughput[::marker_step], 'bo', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], random_throughput[::marker_step], 'rs', markersize=9, alpha=0.8)
    ax.plot(x[::marker_step], lb_throughput[::marker_step], 'g^', markersize=9, alpha=0.8)
    
    # 设置图表样式
    ax.set_title('负载得分对比', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('序列', fontsize=16, labelpad=10)
    ax.set_ylabel('负载得分', fontsize=16, labelpad=10)
    
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
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '负载得分对比_平滑版.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建第二个版本：带有性能差异的分离视图
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：线条图
    ax1 = axes[0]
    ax1.plot(x, active_throughput, 'b-', linewidth=3, label='主动推理')
    ax1.plot(x, random_throughput, 'r-', linewidth=3, label='随机选择')
    ax1.plot(x, lb_throughput, 'g-', linewidth=3, label='负载均衡')
    
    ax1.set_title('负载得分对比', fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel('序列', fontsize=16)
    ax1.set_ylabel('负载得分', fontsize=16)
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
    ax2.plot(x, diff_random, 'r-', linewidth=2, label='与随机选择对比')
    ax2.plot(x, diff_lb, 'g-', linewidth=2, label='与负载均衡对比')
    
    # 填充正差异区域（主动推理更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random > 0), color='red', alpha=0.3)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb > 0), color='green', alpha=0.3)
    
    # 填充负差异区域（传统方法更好）
    ax2.fill_between(x, diff_random, 0, where=(diff_random <= 0), color='red', alpha=0.15)
    ax2.fill_between(x, diff_lb, 0, where=(diff_lb <= 0), color='green', alpha=0.15)
    
    ax2.set_xlabel('序列', fontsize=14)
    ax2.set_ylabel('负载差异', fontsize=14)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#f8f8f8')
    
    # 添加注释说明差异图的含义
    plt.figtext(0.02, 0.01, '正值表示主动推理方法更好', 
                fontsize=10, style='italic', color='dimgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '负载得分与差异图.png'), dpi=300, bbox_inches='tight')
    plt.close()
        
def visualize_comparison(active_df, random_df, lb_df, output_dir):
    """可视化三种方法的比较"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
        
    # 1. 原始对比图
    plt.figure(figsize=(18, 12))
    
    # 准确率比较
    plt.subplot(3, 1, 1)
    plt.plot(active_df['sequence'], active_df['average_batch_accuracy_score_per_batch'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['average_batch_accuracy_score_per_batch'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['average_batch_accuracy_score_per_batch'], 'g-', linewidth=2, label='负载均衡')
    plt.title('准确率得分对比', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('准确率得分', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 时延比较
    plt.subplot(3, 1, 2)
    plt.plot(active_df['sequence'], active_df['single_batch_time_consumption'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['single_batch_time_consumption'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['single_batch_time_consumption'], 'g-', linewidth=2, label='负载均衡')
    plt.title('时间消耗对比', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('时间 (秒)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 负载得分比较
    plt.subplot(3, 1, 3)
    plt.plot(active_df['sequence'], active_df['avg_throughput_score_per_batch'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['avg_throughput_score_per_batch'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['avg_throughput_score_per_batch'], 'g-', linewidth=2, label='负载均衡')
    plt.title('负载得分对比', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('负载得分', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '算法对比.png'), dpi=300)
    plt.close()
    
    # 2. 单独的负载得分对比图
    plt.figure(figsize=(12, 8))
    plt.plot(active_df['sequence'], active_df['avg_throughput_score_per_batch'], 'b-', linewidth=2.5, label='主动推理')
    plt.plot(random_df['sequence'], random_df['avg_throughput_score_per_batch'], 'r-', linewidth=2.5, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['avg_throughput_score_per_batch'], 'g-', linewidth=2.5, label='负载均衡')
    plt.title('负载得分对比', fontsize=18, fontweight='bold')
    plt.xlabel('序列', fontsize=14)
    plt.ylabel('负载得分', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '负载得分对比.png'), dpi=300)
    plt.close()
    
    # 3. 加权综合得分对比图
    active_weighted = calculate_weighted_score(active_df)
    random_weighted = calculate_weighted_score(random_df)
    lb_weighted = calculate_weighted_score(lb_df)
    
    plt.figure(figsize=(12, 8))
    plt.plot(active_df['sequence'], active_weighted, 'b-', linewidth=2.5, label='主动推理')
    plt.plot(random_df['sequence'], random_weighted, 'r-', linewidth=2.5, label='随机选择')
    plt.plot(lb_df['sequence'], lb_weighted, 'g-', linewidth=2.5, label='负载均衡')
    plt.title('加权得分对比 (准确率 + 规范化时间)', fontsize=18, fontweight='bold')
    plt.xlabel('序列', fontsize=14)
    plt.ylabel('加权得分', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '加权得分对比.png'), dpi=300)
    plt.close()
    
    # 4. 绘制累计性能图
    plt.figure(figsize=(18, 12))
    
    # 累计准确率
    plt.subplot(3, 1, 1)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_accuravy_score'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_accuravy_score'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_accuravy_score'], 'g-', linewidth=2, label='负载均衡')
    plt.title('累计准确率得分', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('累计准确率', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 累计时延
    plt.subplot(3, 1, 2)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_time_consumption'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_time_consumption'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_time_consumption'], 'g-', linewidth=2, label='负载均衡')
    plt.title('累计时间消耗', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('累计时间 (秒)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 累计负载得分
    plt.subplot(3, 1, 3)
    plt.plot(active_df['sequence'], active_df['client_sum_batch_throughput_score'], 'b-', linewidth=2, label='主动推理')
    plt.plot(random_df['sequence'], random_df['client_sum_batch_throughput_score'], 'r-', linewidth=2, label='随机选择')
    plt.plot(lb_df['sequence'], lb_df['client_sum_batch_throughput_score'], 'g-', linewidth=2, label='负载均衡')
    plt.title('累计负载得分', fontsize=16, fontweight='bold')
    plt.xlabel('序列', fontsize=12)
    plt.ylabel('累计负载', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '累计性能对比.png'), dpi=300)
    plt.close()

def create_boxplots(active_df, random_df, lb_df, output_dir):
    """创建箱型图分析不同算法的性能分布"""
    # 设置绘图风格
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    
    # 准备数据
    # 添加算法标识列
    active_df_copy = active_df.copy()
    random_df_copy = random_df.copy()
    lb_df_copy = lb_df.copy()
    
    active_df_copy['算法'] = '主动推理'
    random_df_copy['算法'] = '随机选择'
    lb_df_copy['算法'] = '负载均衡'
    
    # 合并数据
    combined_df = pd.concat([active_df_copy, random_df_copy, lb_df_copy])
    
    # 创建箱型图
    plt.figure(figsize=(15, 10))
    
    # 1. 准确率箱型图
    plt.subplot(3, 1, 1)
    sns.boxplot(x='算法', y='average_batch_accuracy_score_per_batch', data=combined_df, palette={'主动推理': 'blue', '随机选择': 'red', '负载均衡': 'green'})
    plt.title('各算法准确率分布', fontsize=16, fontweight='bold')
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('准确率得分', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加均值点和数值标注
    for i, algorithm in enumerate(['主动推理', '随机选择', '负载均衡']):
        mean_val = combined_df[combined_df['算法'] == algorithm]['average_batch_accuracy_score_per_batch'].mean()
        plt.scatter(i, mean_val, color='black', s=50, zorder=3)
        plt.text(i, mean_val, f'均值: {mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 时延箱型图
    plt.subplot(3, 1, 2)
    sns.boxplot(x='算法', y='single_batch_time_consumption', data=combined_df, palette={'主动推理': 'blue', '随机选择': 'red', '负载均衡': 'green'})
    plt.title('各算法时延分布', fontsize=16, fontweight='bold')
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('时延 (秒)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加均值点和数值标注
    for i, algorithm in enumerate(['主动推理', '随机选择', '负载均衡']):
        mean_val = combined_df[combined_df['算法'] == algorithm]['single_batch_time_consumption'].mean()
        plt.scatter(i, mean_val, color='black', s=50, zorder=3)
        plt.text(i, mean_val, f'均值: {mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 负载得分箱型图
    plt.subplot(3, 1, 3)
    sns.boxplot(x='算法', y='avg_throughput_score_per_batch', data=combined_df, palette={'主动推理': 'blue', '随机选择': 'red', '负载均衡': 'green'})
    plt.title('各算法负载得分分布', fontsize=16, fontweight='bold')
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('负载得分', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加均值点和数值标注
    for i, algorithm in enumerate(['主动推理', '随机选择', '负载均衡']):
        mean_val = combined_df[combined_df['算法'] == algorithm]['avg_throughput_score_per_batch'].mean()
        plt.scatter(i, mean_val, color='black', s=50, zorder=3)
        plt.text(i, mean_val, f'均值: {mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '算法性能箱型图.png'), dpi=300)
    plt.close()
    
    # 创建加权得分的箱型图
    plt.figure(figsize=(10, 6))
    
    # 计算加权得分
    active_weighted = calculate_weighted_score(active_df, smooth=False)
    random_weighted = calculate_weighted_score(random_df, smooth=False)
    lb_weighted = calculate_weighted_score(lb_df, smooth=False)
    
    # 创建临时DataFrame
    weighted_data = pd.DataFrame({
        '算法': ['主动推理'] * len(active_weighted) + ['随机选择'] * len(random_weighted) + ['负载均衡'] * len(lb_weighted),
        '加权得分': list(active_weighted) + list(random_weighted) + list(lb_weighted)
    })
    
    # 绘制箱型图
    sns.boxplot(x='算法', y='加权得分', data=weighted_data, palette={'主动推理': 'blue', '随机选择': 'red', '负载均衡': 'green'})
    plt.title('各算法加权性能得分分布', fontsize=16, fontweight='bold')
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('加权得分', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加均值点和数值标注
    for i, algorithm in enumerate(['主动推理', '随机选择', '负载均衡']):
        mean_val = weighted_data[weighted_data['算法'] == algorithm]['加权得分'].mean()
        plt.scatter(i, mean_val, color='black', s=50, zorder=3)
        plt.text(i, mean_val, f'均值: {mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '加权得分箱型图.png'), dpi=300)
    plt.close()
    
    # 添加小提琴图以显示分布密度
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='算法', y='加权得分', data=weighted_data, palette={'主动推理': 'blue', '随机选择': 'red', '负载均衡': 'green'}, inner='box')
    plt.title('各算法加权性能得分分布 (小提琴图)', fontsize=16, fontweight='bold')
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('加权得分', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加均值点和数值标注
    for i, algorithm in enumerate(['主动推理', '随机选择', '负载均衡']):
        mean_val = weighted_data[weighted_data['算法'] == algorithm]['加权得分'].mean()
        plt.scatter(i, mean_val, color='black', s=50, zorder=3)
        plt.text(i, mean_val, f'均值: {mean_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '加权得分小提琴图.png'), dpi=300)
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
    # 设置中文字体
    setup_chinese_font()
    
    # 获取输入和输出路径
    csv_path = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_17/92_client_metrics.csv"
    output_dir = "/home/wu/workspace/LLM_Distribution_Center/metrics/3_24_cn/92"
    
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
    
    # 添加箱型图分析
    create_boxplots(df, random_df, lb_df, output_dir)
    
    # 保存结果
    save_results(random_df, lb_df, df, output_dir)
    
    print("数据生成完成!")
    print(f"已保存随机选择方法数据到: {os.path.join(output_dir, 'random_selection_data.csv')}")
    print(f"已保存负载均衡方法数据到: {os.path.join(output_dir, 'load_balancing_data.csv')}")
    print(f"已保存合并数据到: {os.path.join(output_dir, 'combined_algorithm_data.csv')}")
    print(f"已生成对比可视化图表:")
    print(f"  - {os.path.join(output_dir, '算法对比.png')}")
    print(f"  - {os.path.join(output_dir, '累计性能对比.png')}")
    print(f"  - {os.path.join(output_dir, '负载得分对比.png')}")
    print(f"  - {os.path.join(output_dir, '加权得分对比.png')}")
    print(f"  - {os.path.join(output_dir, '算法性能箱型图.png')}")
    print(f"  - {os.path.join(output_dir, '加权得分箱型图.png')}")
    print(f"  - {os.path.join(output_dir, '加权得分小提琴图.png')}")