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
plt.rcParams['axes.titlesize'] = 100 # 轴标题
plt.rcParams['axes.labelsize'] = 100  # 轴标签
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

# 批次大小到子图标记的映射
batch_to_subfig = {
    10: '(a)',
    30: '(b)',
    60: '(c)'
}

# 创建日志文件
log_file = open('performance_comparison.log', 'w', encoding='utf-8')

# 尝试加载数据
try:
    # 计算主动推理算法相对于其他算法的性能优势
    def calculate_performance_advantage(metrics_df):
        """
        计算主动推理算法相对于其他算法的性能优势
        
        参数:
        metrics_df - 包含性能指标的DataFrame
        
        返回:
        优势分析结果的字符串
        """
        result_str = "# 主动推理算法性能优势分析\n\n"
        
        # 对于每个带宽值，计算主动推理算法的平均性能优势
        for bandwidth in sorted(metrics_df['bandwidth'].unique()):
            result_str += f"## 带宽 {bandwidth}Mbps 条件下的性能对比\n\n"
            
            bw_data = metrics_df[metrics_df['bandwidth'] == bandwidth]
            
            # 对于每个性能指标，计算主动推理算法的平均优势
            for metric, metric_name in zip(['accuracy', 'latency', 'resource'], 
                                         ['准确率', '耗时 (ms)', '资源利用率']):
                result_str += f"### {metric_name}\n\n"
                
                # 计算每个批次大小下的优势
                for batch_size in sorted(bw_data['batch_size'].unique()):
                    result_str += f"- 批次大小 {batch_size}:\n"
                    
                    # 获取该工况下的数据
                    condition_data = bw_data[bw_data['batch_size'] == batch_size]
                    
                    # 获取主动推理算法的性能
                    active_data = condition_data[condition_data['algorithm'] == 'active_reasoning']
                    if len(active_data) == 0:
                        result_str += f"  - 没有主动推理算法数据\n"
                        continue
                    
                    active_value = active_data[metric].values[0]
                    
                    # 与其他算法比较
                    for algo in ['round_robin', 'dqn']:
                        algo_data = condition_data[condition_data['algorithm'] == algo]
                        if len(algo_data) == 0:
                            continue
                            
                        algo_value = algo_data[metric].values[0]
                        algo_zh = algo_names.get(algo, algo)
                        
                        # 计算性能差异
                        if metric == 'accuracy' or metric == 'resource':
                            # 对于准确率和资源利用率，更高更好
                            diff = active_value - algo_value
                            diff_percent = diff / algo_value * 100
                            if diff > 0:
                                comparison = "优于"
                            else:
                                comparison = "劣于"
                                diff = -diff
                                diff_percent = -diff_percent
                        else:  # latency
                            # 对于延迟，更低更好
                            diff = algo_value - active_value
                            diff_percent = diff / algo_value * 100
                            if diff > 0:
                                comparison = "优于"
                            else:
                                comparison = "劣于"
                                diff = -diff
                                diff_percent = -diff_percent
                                
                        result_str += f"  - 相比{algo_zh}：{comparison} {diff:.4f} ({diff_percent:.2f}%)\n"
                
                result_str += "\n"
            
            # 计算带宽条件下的平均优势
            result_str += "### 平均优势\n\n"
            
            # 计算每个指标的平均优势
            for metric, metric_name in zip(['accuracy', 'latency', 'resource'], 
                                         ['准确率', '耗时 (ms)', '资源利用率']):
                # 获取主动推理算法的平均性能
                active_avg = bw_data[bw_data['algorithm'] == 'active_reasoning'][metric].mean()
                
                # 与其他算法比较
                for algo in ['round_robin', 'dqn']:
                    algo_avg = bw_data[bw_data['algorithm'] == algo][metric].mean()
                    algo_zh = algo_names.get(algo, algo)
                    
                    # 计算性能差异
                    if metric == 'accuracy' or metric == 'resource':
                        # 对于准确率和资源利用率，更高更好
                        diff = active_avg - algo_avg
                        diff_percent = diff / algo_avg * 100
                        if diff > 0:
                            comparison = "优于"
                        else:
                            comparison = "劣于"
                            diff = -diff
                            diff_percent = -diff_percent
                    else:  # latency
                        # 对于延迟，更低更好
                        diff = algo_avg - active_avg
                        diff_percent = diff / algo_avg * 100
                        if diff > 0:
                            comparison = "优于"
                        else:
                            comparison = "劣于"
                            diff = -diff
                            diff_percent = -diff_percent
                            
                    result_str += f"- {metric_name}：相比{algo_zh}平均{comparison} {diff:.4f} ({diff_percent:.2f}%)\n"
                
                result_str += "\n"
            
            result_str += "---\n\n"
            
        return result_str
    
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
        
        # 获取子图标记
        subfig_mark = batch_to_subfig.get(batch_size, '')
        
        # 筛选指定工况的数据
        condition_data = metrics_df[(metrics_df['bandwidth'] == bandwidth) & (metrics_df['batch_size'] == batch_size)]
        
        # 如果没有数据，直接返回
        if len(condition_data) == 0:
            print(f"警告: 带宽 {bandwidth}Mbps, 批次大小 {batch_size} 没有数据，跳过绘图")
            return
        
        # 指标列表
        metrics = ['accuracy', 'latency', 'resource']
        metric_titles = ['准确率', '耗时 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(metrics, metric_titles):
            # 创建图表 - 增加底部空间用于放置标题
            fig, ax = plt.subplots(figsize=(14, 13))  # 减小高度以便更好控制标题位置
            
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
            
            # 设置标签
            ax.set_xlabel('算法', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_ylabel(title, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 增加图表边框
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)
                spine.set_color(TEXT_COLOR)
            
            # 调整刻度标签颜色
            ax.tick_params(axis='both', colors=TEXT_COLOR)
            
            # 增加底部空间用于放置标题，但比之前更小
            plt.subplots_adjust(bottom=0.15)
            
            # 在x轴下方添加标题文本，位置更靠近x轴，并添加子图标记
            title_text = f'{subfig_mark} 带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}'
            plt.figtext(0.5, 0.08, title_text, ha='center', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 保存为矢量图
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 保留更小的底部空间给标题
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}.svg', format='svg', bbox_inches='tight')
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
        
        # 获取子图标记
        subfig_mark = batch_to_subfig.get(batch_size, '')
        
        config = f'bw{bandwidth}_bs{batch_size}'
        
        # 检查配置是否存在
        if config not in data:
            print(f"警告: 配置 {config} 不存在，跳过绘图")
            return
        
        # 获取该配置的数据
        config_data = data[config]
        
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '耗时 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(history_metrics, metric_titles):
            # 创建图表 - 增加底部空间用于放置标题
            fig, ax = plt.subplots(figsize=(14, 13))  # 减小高度以便更好控制标题位置
            
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
            
            # 设置标签
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
            
            # 增加底部空间用于放置标题，但比之前更小
            plt.subplots_adjust(bottom=0.15)
            
            # 在x轴下方添加标题文本，位置更靠近x轴，并添加子图标记
            title_text = f'{subfig_mark} 带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}变化'
            plt.figtext(0.5, 0.08, title_text, ha='center', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 保存为矢量图
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 保留更小的底部空间给标题
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}_history.svg', format='svg', bbox_inches='tight')
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
        
        # 获取子图标记
        subfig_mark = batch_to_subfig.get(batch_size, '')
        
        config = f'bw{bandwidth}_bs{batch_size}'
        
        # 检查配置是否存在
        if config not in data:
            print(f"警告: 配置 {config} 不存在，跳过绘图")
            return
        
        # 获取该配置的数据
        config_data = data[config]
        
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '耗时 (ms)', '资源利用率']
        
        # 为每个指标创建单独的图
        for metric, title in zip(history_metrics, metric_titles):
            # 创建图表 - 使用更大的图形尺寸以确保完整显示
            fig, ax = plt.subplots(figsize=(14, 14))
            
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
            
            # 计算数据范围，确保合理的Y轴范围
            min_val = boxplot_df['value'].min()
            max_val = boxplot_df['value'].max()
            
            # 使用seaborn的箱型图，按算法分组
            box_props = {
                'boxprops': {'linewidth': 2.5, 'edgecolor': TEXT_COLOR},
                'whiskerprops': {'linewidth': 2.5, 'color': TEXT_COLOR},
                'medianprops': {'color': TEXT_COLOR, 'linewidth': 2.5},
                'capprops': {'linewidth': 2.5, 'color': TEXT_COLOR},
                'flierprops': {'marker': 'o', 'markerfacecolor': BG_COLOR, 'markeredgecolor': TEXT_COLOR, 
                              'markersize': 8, 'markeredgewidth': 1.5}
            }
            
            # 创建自定义调色板，确保使用配色方案中的颜色
            custom_palette = {algo_zh: scheme_colors[algo_zh] for algo_zh in boxplot_df['algorithm_zh'].unique()}
            
            # 使用自定义调色板
            sns.boxplot(x='algorithm_zh', y='value', data=boxplot_df, ax=ax, 
                      palette=custom_palette, linewidth=3.0, width=0.7, **box_props,
                      showfliers=False)  # 不显示异常值点，以避免过度拥挤
            
            # 为不同算法使用不同形状的点，只显示少量点以避免过度拥挤
            for j, algo_zh in enumerate(boxplot_df['algorithm_zh'].unique()):
                # 获取该算法的数据
                algo_data = boxplot_df[boxplot_df['algorithm_zh'] == algo_zh]
                
                # 进一步减少显示的点数，只显示一小部分
                sample_size = min(20, len(algo_data))
                sampled_data = algo_data.sample(sample_size) if len(algo_data) > sample_size else algo_data
                
                # 使用对应算法的标记形状
                marker = markers.get(algo_zh, 'o')
                
                # 手动添加散点，确保使用正确的形状
                ax.scatter(
                    x=[j] * len(sampled_data),  # x坐标为箱型图的索引
                    y=sampled_data['value'],    # y坐标为值
                    color=TEXT_COLOR,        # 点的颜色
                    marker=marker,           # 使用算法对应的标记形状
                    alpha=0.6,               # 透明度
                    s=100,                   # 点的大小
                    edgecolor='white',       # 边缘颜色
                    linewidth=1.5,           # 边缘宽度
                    zorder=3                 # 确保点显示在箱型图上方
                )
            
            # 设置标签
            ax.set_xlabel('算法', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            ax.set_ylabel(title, fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 调整Y轴范围，确保箱型图完全显示
            if metric == 'accuracies':
                # 为准确率设置固定范围
                ax.set_ylim([0.3, 1.05])
            elif metric == 'latencies':
                if max_val > 1000:
                    ax.set_yscale('log')
                    # 对数刻度下，确保足够的上下边界
                    y_min = max(1, min_val * 0.5)
                    y_max = max_val * 2
                    ax.set_ylim([y_min, y_max])
                else:
                    # 线性刻度下，添加足够的边界
                    margin = (max_val - min_val) * 0.2  # 20%的边界
                    ax.set_ylim([max(0, min_val - margin), max_val + margin])
            elif metric == 'resources':
                # 为资源利用率设置合理范围
                margin = (max_val - min_val) * 0.2  # 20%的边界
                ax.set_ylim([max(0, min_val - margin), min(1.0, max_val + margin)])
            
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
            
            # 增加底部空间用于放置标题
            plt.subplots_adjust(bottom=0.15)
            
            # 在x轴下方添加标题文本，位置更靠近x轴，并添加子图标记
            title_text = f'{subfig_mark} 带宽 {bandwidth}Mbps, 批次大小 {batch_size} 下的{title}分布'
            plt.figtext(0.5, 0.08, title_text, ha='center', fontsize=34, fontweight='bold', color=TEXT_COLOR)
            
            # 保存为矢量图
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])
            plt.savefig(f'bw{bandwidth}_bs{batch_size}_{metric}_boxplot.svg', format='svg', bbox_inches='tight')
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
        'latency': '平均耗时 (ms)',
        'resource': '平均资源利用率'
    }
    
    # 历史数据指标名称映射
    history_metric_names = {
        'accuracies': '准确率',
        'latencies': '耗时 (ms)',
        'resources': '资源利用率'
    }

    # 设置图表样式
    markers = {'主动推理': 'o', '轮询算法': 's', 'DQN': '^'}
    line_styles = {'主动推理': '-', '轮询算法': '--', 'DQN': '-.'}

    # 添加中文算法名称
    metrics_df['algorithm_zh'] = metrics_df['algorithm'].map(algo_names)
    
    # 计算主动推理算法的性能优势并写入日志
    performance_analysis = calculate_performance_advantage(metrics_df)
    log_file.write(performance_analysis)
    log_file.close()
    print("性能对比分析已写入 performance_comparison.log 文件")

    # 创建输出目录
    os.makedirs('vector_graphics_with_analysis', exist_ok=True)
    os.chdir('vector_graphics_with_analysis')

    print("开始生成矢量图...")
    
    # 为所有工况绘制图表
    for bandwidth in [10, 50, 100]:
        for batch_size in [10, 30, 60]:
            print(f"  绘制工况: 带宽 {bandwidth}Mbps, 批次大小 {batch_size}")
            plot_metrics_for_condition(bandwidth, batch_size)
            plot_history_for_condition(bandwidth, batch_size)
            plot_boxplot_for_condition(bandwidth, batch_size)
    
    print("\n所有矢量图生成完成！保存在 vector_graphics_with_analysis 目录下")

except Exception as e:
    print(f"加载或绘图失败: {e}")
    import traceback
    traceback.print_exc()
    log_file.write(f"错误: {e}\n")
    log_file.write(traceback.format_exc())
    log_file.close()