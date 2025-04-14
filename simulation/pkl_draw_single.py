import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import warnings
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
warnings.filterwarnings('ignore')

# 设置中文字体
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'Source Han Serif SC', 'AR PL UMing CN', 'WenQuanYi Zen Hei'] + plt.rcParams['font.sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 增加字体大小和粗细
plt.rcParams['font.size'] = 60  # 基础字体大小
plt.rcParams['axes.titlesize'] = 34  # 轴标题
plt.rcParams['axes.labelsize'] = 34  # 轴标签
plt.rcParams['xtick.labelsize'] = 34  # x轴刻度标签
plt.rcParams['ytick.labelsize'] = 34  # y轴刻度标签
plt.rcParams['legend.fontsize'] = 34  # 图例
plt.rcParams['figure.titlesize'] = 36  # 图表标题

# 设置字体加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 轴标题加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签加粗
plt.rcParams['font.weight'] = 'bold'  # 整体字体加粗

# 定义自定义 unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 处理 numpy._core
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        # 处理其他可能的路径变化
        elif module == 'numpy.core.multiarray':
            if name == '_reconstruct':
                return np.core.multiarray._reconstruct
        return super().find_class(module, name)

# 文件路径
pkl_file_path = '/home/wu/workspace/LLM_Distribution_Center/simulation/simulation_v2_multi_thread_model_metrics/results/evaluation_results.pkl'  # 替换为您的文件路径

# 定义自定义配色方案
custom_scheme = {
    'name': 'custom_colors',
    'colors': {
        '主动推理': '#16068A',  # 深蓝色 (R022 G006 B138)
        '轮询算法': '#9E189D',  # 紫红色 (R158 G024 B157)
        'DQN': '#EC7853'        # 珊瑚色 (R236 G120 B083)
    },
    'bg_color': '#FFFFFF',      # 纯白色
    'text_color': '#333333',    # 深灰色文本
    'grid_color': '#E0E0E0'     # 浅灰色网格线
}

# 尝试加载数据
try:
    # 为每个工况绘制性能指标图
    def plot_metrics_for_condition(bandwidth, batch_size):
        """
        为指定工况（带宽和批次大小组合）绘制性能指标图
        
        参数:
        bandwidth - 带宽值，如10, 50, 100
        batch_size - 批次大小，如10, 30, 60
        """
        # 获取配色方案
        scheme_colors = custom_scheme['colors']
        BG_COLOR = custom_scheme['bg_color']
        TEXT_COLOR = custom_scheme['text_color']
        GRID_COLOR = custom_scheme['grid_color']
        
        # 筛选指定工况的数据
        condition_data = metrics_df[(metrics_df['bandwidth'] == bandwidth) & (metrics_df['batch_size'] == batch_size)]
        
        # 如果没有数据，直接返回
        if len(condition_data) == 0:
            print(f"警告: 带宽 {bandwidth}Mbps, 批次大小 {batch_size} 没有数据，跳过绘图")
            return
        
        # 指标列表
        metrics = ['accuracy', 'latency', 'resource']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(metrics, metric_titles):
            # 创建图表
            fig, ax = plt.figure(figsize=(14, 12)), plt.gca()
            
            # 设置背景色
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            
            # 准备数据
            sorted_data = condition_data.sort_values('algorithm_zh')
            algo_names = sorted_data['algorithm_zh'].tolist()
            values = sorted_data[metric].tolist()
            
            # 设置条形图的位置
            x_pos = np.arange(len(algo_names))
            
            # 使用配色方案
            colors_selected = [scheme_colors[name] for name in algo_names]
            
            # 绘制条形图
            bars = ax.bar(x_pos, values, align='center', alpha=0.9, 
                  color=colors_selected, 
                  width=0.7, edgecolor=TEXT_COLOR, linewidth=1.5)
            
            # 添加数据标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', 
                        fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 设置x轴标签
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algo_names, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 设置y轴范围
            if metric == 'accuracy':
                ax.set_ylim([0.5, 0.9])
            elif metric == 'latency':
                ax.set_ylim([0, max(values) * 1.2])
            elif metric == 'resource':
                ax.set_ylim([0.3, 0.5])
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.4, axis='y', linewidth=1.5, color=GRID_COLOR)
            
            # 设置标题和标签
            ax.set_title(f'带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_xlabel('算法', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_ylabel(title, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 增加图表边框
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)
                spine.set_color(TEXT_COLOR)
            
            # 调整刻度标签颜色
            ax.tick_params(axis='both', colors=TEXT_COLOR)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 为每个工况绘制历史趋势图
    def plot_history_for_condition(bandwidth, batch_size):
        """
        为指定工况（带宽和批次大小组合）绘制历史趋势图
        
        参数:
        bandwidth - 带宽值，如10, 50, 100
        batch_size - 批次大小，如10, 30, 60
        """
        # 获取配色方案
        scheme_colors = custom_scheme['colors']
        BG_COLOR = custom_scheme['bg_color']
        TEXT_COLOR = custom_scheme['text_color']
        GRID_COLOR = custom_scheme['grid_color']
        
        config = f'bw{bandwidth}_bs{batch_size}'
        
        # 检查配置是否存在
        if config not in data:
            print(f"警告: 配置 {config} 不存在，跳过绘图")
            return
        
        # 获取该配置的数据
        config_data = data[config]
        
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(history_metrics, metric_titles):
            # 创建图表
            fig, ax = plt.figure(figsize=(14, 12)), plt.gca()
            
            # 设置背景色
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            
            # 检查是否所有算法都有该指标的历史数据
            missing_data = False
            for algo in config_data:
                if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                    print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                    missing_data = True
                    break
            
            if missing_data:
                plt.close()
                continue
            
            # 确定所有算法中最短的历史记录长度
            min_length = min(len(config_data[algo]['history'][metric]) for algo in config_data)
            
            # 为每个算法绘制历史趋势
            for algo, metrics_data in config_data.items():
                # 获取历史数据
                values = metrics_data['history'][metric][:min_length]
                
                # 应用平滑处理，减少陡峭变化
                sigma = 10  # 平滑强度
                values_smoothed = gaussian_filter1d(values, sigma)
                
                # 减少数据点，只保留一部分点
                step = max(1, len(values_smoothed) // 12)  # 确保显示适量的点
                x_values = np.arange(0, len(values_smoothed), step)
                y_values = values_smoothed[::step]
                
                # 绘制趋势线
                algo_zh = algo_names.get(algo, algo)
                ax.plot(x_values, y_values, 
                       label=algo_zh,
                       marker=markers[algo_zh], 
                       color=scheme_colors[algo_zh],  # 使用配色方案中的颜色
                       linewidth=5.0,  # 线宽
                       markersize=20,  # 标记大小
                       linestyle=line_styles[algo_zh],
                       markeredgecolor=TEXT_COLOR,  # 标记边缘颜色
                       markeredgewidth=2.0)  # 标记边缘宽度
            
            # 设置标题和标签
            ax.set_title(f'带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}变化', 
                        fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_xlabel('迭代次数', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_ylabel(title, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.5, color=GRID_COLOR)
            
            # 调整Y轴范围
            if metric == 'latencies' and max(values) > 1000:
                ax.set_yscale('log')
            elif metric == 'accuracies':
                ax.set_ylim([0.4, 1.0])
            elif metric == 'resources':
                ax.set_ylim([0.2, 0.6])
            
            # 添加图例
            legend = ax.legend(fontsize=34, loc='best', frameon=True, facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
            for text in legend.get_texts():
                text.set_fontweight('bold')
                text.set_color(TEXT_COLOR)
            
            # 增加图表边框
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)
                spine.set_color(TEXT_COLOR)
            
            # 调整刻度标签颜色
            ax.tick_params(axis='both', colors=TEXT_COLOR)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}_history.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 为每个工况绘制箱型图
    def plot_boxplot_for_condition(bandwidth, batch_size):
        """
        为指定工况（带宽和批次大小组合）绘制箱型图
        
        参数:
        bandwidth - 带宽值，如10, 50, 100
        batch_size - 批次大小，如10, 30, 60
        """
        # 获取配色方案
        scheme_colors = custom_scheme['colors']
        BG_COLOR = custom_scheme['bg_color']
        TEXT_COLOR = custom_scheme['text_color']
        GRID_COLOR = custom_scheme['grid_color']
        
        config = f'bw{bandwidth}_bs{batch_size}'
        
        # 检查配置是否存在
        if config not in data:
            print(f"警告: 配置 {config} 不存在，跳过绘图")
            return
        
        # 获取该配置的数据
        config_data = data[config]
        
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(history_metrics, metric_titles):
            # 创建图表
            fig, ax = plt.figure(figsize=(14, 12)), plt.gca()
            
            # 设置背景色
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            
            # 准备箱型图数据
            boxplot_data = []
            
            # 检查是否所有算法都有该指标的历史数据
            missing_data = False
            for algo in config_data:
                if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                    print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                    missing_data = True
                    break
            
            if missing_data:
                plt.close()
                continue
            
            # 为每个算法收集数据
            for algo, metrics_data in config_data.items():
                values = metrics_data['history'][metric]
                # 减少数据点，只保留一部分点以减少过度拥挤
                step = max(1, len(values) // 30)  # 最多保留30个数据点
                sampled_values = values[::step]
                
                for value in sampled_values:
                    boxplot_data.append({
                        'algorithm': algo,
                        'algorithm_zh': algo_names.get(algo, algo),
                        'value': value
                    })
            
            # 转换为DataFrame
            boxplot_df = pd.DataFrame(boxplot_data)
            
            # 使用seaborn的箱型图，按算法分组
            box_props = {'boxprops': {'linewidth': 2.5, 'edgecolor': TEXT_COLOR},
                        'whiskerprops': {'linewidth': 2.5, 'color': TEXT_COLOR},
                        'medianprops': {'color': TEXT_COLOR, 'linewidth': 2.5},
                        'capprops': {'linewidth': 2.5, 'color': TEXT_COLOR}}
            
            # 创建自定义调色板，确保使用配色方案中的颜色
            custom_palette = {algo_zh: scheme_colors[algo_zh] for algo_zh in boxplot_df['algorithm_zh'].unique()}
            
            # 使用自定义调色板
            sns.boxplot(x='algorithm_zh', y='value', data=boxplot_df, ax=ax, 
                      palette=custom_palette, linewidth=3.0, width=0.7, **box_props)
            
            # 为不同算法使用不同形状的点
            for j, algo_zh in enumerate(boxplot_df['algorithm_zh'].unique()):
                # 获取该算法的数据
                algo_data = boxplot_df[boxplot_df['algorithm_zh'] == algo_zh]
                
                # 使用对应算法的标记形状
                marker = markers.get(algo_zh, 'o')
                
                # 手动添加散点，确保使用正确的形状
                ax.scatter(
                    x=[j] * len(algo_data),  # x坐标为箱型图的索引
                    y=algo_data['value'],    # y坐标为值
                    color=TEXT_COLOR,        # 点的颜色
                    marker=marker,           # 使用算法对应的标记形状
                    alpha=0.6,               # 透明度
                    s=100,                   # 点的大小
                    edgecolor='white',       # 边缘颜色
                    linewidth=1.5,           # 边缘宽度
                    zorder=3                 # 确保点显示在箱型图上方
                )
            
            # 设置标题和标签
            ax.set_title(f'带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}分布', 
                        fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_xlabel('算法', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_ylabel(title, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 调整Y轴范围
            if metric == 'accuracies':
                ax.set_ylim([0.4, 1.0])
            elif metric == 'latencies':
                ax.set_yscale('log')
                y_max = boxplot_df['value'].max() * 2
                y_min = max(10, boxplot_df['value'].min() * 0.5)
                ax.set_ylim([y_min, y_max])
            elif metric == 'resources':
                ax.set_ylim([0.2, 0.6])
            
            # 加粗x轴标签
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_color(TEXT_COLOR)
            
            # 加粗y轴标签
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color(TEXT_COLOR)
            
            # 增加图表边框
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)
                spine.set_color(TEXT_COLOR)
            
            # 调整刻度标签颜色
            ax.tick_params(axis='both', colors=TEXT_COLOR)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    with open(pkl_file_path, 'rb') as f:
        data = CustomUnpickler(f).load()
    
    print("成功加载数据!")
    print(f"数据类型: {type(data)}")
    
    # 构建性能指标数据框
    metrics_data = []
    for config, algos in data.items():
        bw, bs = config.split('_')
        bw_val = int(bw.replace('bw', ''))
        bs_val = int(bs.replace('bs', ''))
        
        for algo, metrics in algos.items():
            metrics_data.append({
                'bandwidth': bw_val,
                'batch_size': bs_val,
                'algorithm': algo,
                'accuracy': metrics['avg_accuracy'],
                'latency': metrics['avg_latency'],
                'resource': metrics['avg_resource']
            })

    metrics_df = pd.DataFrame(metrics_data)

    # 算法中文名称映射
    algo_names = {
        'active_reasoning': '主动推理',
        'round_robin': '轮询算法',
        'dqn': 'DQN'
    }

    # 指标中文名称映射
    metric_names = {
        'accuracy': '平均准确率',
        'latency': '平均延迟 (ms)',
        'resource': '平均资源利用率'
    }
    
    # 历史数据指标名称映射
    history_metric_names = {
        'accuracies': '准确率',
        'latencies': '延迟 (ms)',
        'resources': '资源利用率'
    }

    # 设置图表样式
    markers = {'主动推理': 'o', '轮询算法': 's', 'DQN': '^'}
    line_styles = {'主动推理': '-', '轮询算法': '--', 'DQN': '-.'}

    # 添加中文算法名称
    metrics_df['algorithm_zh'] = metrics_df['algorithm'].map(algo_names)

    # 创建输出目录
    os.makedirs('custom_color_scheme', exist_ok=True)
    os.chdir('custom_color_scheme')

    print("开始使用自定义配色方案绘制图表...")
    
    # 为所有工况绘制图表
    for bandwidth in [10, 50, 100]:
        for batch_size in [10, 30, 60]:
            print(f"  绘制工况: 带宽 {bandwidth}Mbps, 批次大小 {batch_size}")
            plot_metrics_for_condition(bandwidth, batch_size)
            plot_history_for_condition(bandwidth, batch_size)
            plot_boxplot_for_condition(bandwidth, batch_size)
    
    print("\n所有图表绘制完成！")

except Exception as e:
    print(f"加载或绘图失败: {e}")
    import traceback
    traceback.print_exc()