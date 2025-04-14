import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimSun', 'Source Han Serif SC', 'AR PL UMing CN', 'WenQuanYi Zen Hei'] + plt.rcParams['font.sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 调整字体大小和粗细
plt.rcParams['font.size'] = 50  # 基础字体大小
plt.rcParams['axes.titlesize'] = 28  # 轴标题
plt.rcParams['axes.labelsize'] = 28  # 轴标签
plt.rcParams['xtick.labelsize'] = 28  # x轴刻度标签
plt.rcParams['ytick.labelsize'] = 28  # y轴刻度标签
plt.rcParams['legend.fontsize'] = 28  # 图例
plt.rcParams['figure.titlesize'] = 28  # 图表标题

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

# 尝试加载数据
try:
    # 新增函数：为同一带宽条件创建组合图
    def plot_combined_metrics_for_bandwidth(bandwidth):
        """
        为指定带宽创建组合图，展示不同算法在各种批次大小下的表现
        
        参数:
        bandwidth - 要展示的带宽值，如10, 50, 100
        """
        # 指标列表
        metrics = ['accuracy', 'latency', 'resource']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 筛选指定带宽的数据
        bw_data = metrics_df[metrics_df['bandwidth'] == bandwidth]
        
        # 如果没有数据，直接返回
        if len(bw_data) == 0:
            print(f"警告: 带宽 {bandwidth}Mbps 没有数据，跳过绘图")
            return
        
        # 获取该带宽下的所有批次大小
        batch_sizes = sorted(bw_data['batch_size'].unique())
        
        # 创建一个大图，包含所有指标的对比
        fig = plt.figure(figsize=(18,  28))
        
        # 定义子图布局 (3行1列)
        gs = fig.add_gridspec(3, 1, hspace=0.4)
        
        # 为每个指标创建一个子图
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = fig.add_subplot(gs[i, 0])
            
            # 为每个算法绘制一条线，展示不同批次大小下的表现
            for algo in bw_data['algorithm_zh'].unique():
                algo_data = bw_data[bw_data['algorithm_zh'] == algo]
                
                # 按批次大小排序
                algo_data = algo_data.sort_values('batch_size')
                
                # 绘制线图
                ax.plot(algo_data['batch_size'], algo_data[metric], 
                       marker=markers[algo], label=algo, 
                       color=colors[algo], linewidth=3.0, 
                       markersize=12, linestyle=line_styles[algo])
                
                # 添加数据标签
                for bs, val in zip(algo_data['batch_size'], algo_data[metric]):
                    ax.text(bs, val, f'{val:.2f}', 
                           ha='center', va='bottom', 
                           fontsize=28, color=colors[algo],
                           fontweight='bold')
            
            # 设置子图标题和标签
            ax.set_title(f'带宽 {bandwidth}Mbps 下的{metric_titles[i]}对比', fontsize=16, fontweight='bold')
            ax.set_ylabel(metric_titles[i], fontsize=28, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # 增大刻度标签字体
            ax.tick_params(axis='both', which='major', labelsize=28)
            
            # 添加图例
            ax.legend(fontsize= 28, loc='best')
            
            # 调整Y轴范围
            if metric == 'accuracy':
                ax.set_ylim([0.6, 0.85])
            elif metric == 'latency':
                ax.set_yscale('log')
                ax.set_ylim([50, 10000])
            elif metric == 'resource':
                ax.set_ylim([0.36, 0.43])
            
            # 修改x轴标签为批次大小
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([f'({chr(97 + idx)}) 批次大小 {bs}' for idx, bs in enumerate(batch_sizes)], fontweight='bold')
            
            # 只在最后一个子图设置x轴标签
            if i == len(metrics) - 1:
                ax.set_xlabel('批次大小', fontsize=28, fontweight='bold')
        
        # 设置整体标题
        fig.suptitle(f'带宽 {bandwidth}Mbps 下的算法性能综合对比', fontsize=18, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为整体标题留出空间
        
        # 保存图表
        plt.savefig(f'带宽{bandwidth}Mbps_综合性能对比.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 新增函数：为同一带宽条件创建历史趋势组合图
    def plot_combined_history_for_bandwidth(bandwidth):
        """
        为指定带宽创建历史趋势组合图，展示不同算法在各种批次大小下的历史表现
        
        参数:
        bandwidth - 要展示的带宽值，如10, 50, 100
        """
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 获取该带宽下的所有配置
        configs = []
        for bs in [10, 30, 60]:
            config = f'bw{bandwidth}_bs{bs}'
            if config in data:
                configs.append((bs, config))
        
        # 按批次大小排序
        configs.sort()
        
        # 如果没有配置，直接返回
        if not configs:
            print(f"警告: 带宽 {bandwidth}Mbps 没有可用配置，跳过绘图")
            return
        
        # 为每个指标创建一个大图，包含该带宽下所有批次大小的历史趋势
        for metric_idx, (metric, title) in enumerate(zip(history_metrics, metric_titles)):
            # 创建子图网格
            fig, axes = plt.subplots(len(configs), 1, figsize=( 28, 6*len(configs)), sharex=True)
            
            # 如果只有一个配置，确保axes是数组
            if len(configs) == 1:
                axes = [axes]
            
            # 首先确定所有配置中最短的历史记录长度
            min_global_length = float('inf')
            for bs, config in configs:
                config_data = data[config]
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据")
                        missing_data = True
                        break
                
                if not missing_data:
                    # 找出当前配置中最短的历史记录长度
                    config_min_length = min(len(config_data[algo]['history'][metric]) for algo in config_data)
                    min_global_length = min(min_global_length, config_min_length)
            
            if min_global_length == float('inf'):
                print(f"警告: 带宽 {bandwidth}Mbps 下所有配置的历史记录都为空或缺失，跳过绘图")
                return
            
            # 为每个配置绘制历史趋势
            for i, (bs, config) in enumerate(configs):
                ax = axes[i]
                
                # 获取该配置下的所有算法数据
                config_data = data[config]
                
                # 准备历史数据
                history_data = []
                
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                        missing_data = True
                        break
                
                if missing_data:
                    ax.text(0.5, 0.5, f'数据缺失', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 为每个算法收集数据，使用全局最短长度
                for algo, metrics_data in config_data.items():
                    # 截取到全局最短长度
                    values = metrics_data['history'][metric][:min_global_length]
                    
                    # 只选择部分点，减少密集度
                    step = max(1, min_global_length // 10)  # 确保最多显示10个点
                    
                    # 使用平滑处理
                    if len(values) > 10:
                        # 使用滑动平均平滑数据
                        window_size = min(5, len(values) // 5)
                        if window_size > 1:
                            values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    
                    # 采样数据点
                    timestamps = list(range(len(values)))
                    sampled_indices = list(range(0, len(values), step))
                    
                    # 确保包含最后一个点
                    if len(values) - 1 not in sampled_indices:
                        sampled_indices.append(len(values) - 1)
                    
                    for idx in sampled_indices:
                        if idx < len(values):  # 确保索引有效
                            history_data.append({
                                'algorithm': algo,
                                'algorithm_zh': algo_names.get(algo, algo),
                                'timestamp': timestamps[idx],
                                'value': values[idx]
                            })
                
                history_df = pd.DataFrame(history_data)
                
                # 绘制时间序列趋势图
                for algo_zh in history_df['algorithm_zh'].unique():
                    algo_data = history_df[history_df['algorithm_zh'] == algo_zh]
                    ax.plot(algo_data['timestamp'], algo_data['value'], 
                           marker=markers.get(algo_zh, 'o'), label=algo_zh, 
                           color=colors.get(algo_zh, 'black'), linewidth=3.0,
                           markersize=10, linestyle=line_styles.get(algo_zh, '-'))
                
                # 设置y轴标签
                ax.set_ylabel(title, fontsize= 28, fontweight='bold')
                
                # 增大刻度标签字体
                ax.tick_params(axis='both', which='major', labelsize= 28)
                
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 调整Y轴范围
                if metric == 'latencies' and history_df['value'].max() > 1000:
                    ax.set_yscale('log')
                    ax.set_ylim([min(50, history_df['value'].min() * 0.9), history_df['value'].max() * 1.1])
                elif metric == 'accuracies':
                    ax.set_ylim([max(0.4, history_df['value'].min() * 0.9), min(1.0, history_df['value'].max() * 1.1)])
                elif metric == 'resources':
                    ax.set_ylim([history_df['value'].min() * 0.9, history_df['value'].max() * 1.1])
                
                # 添加图例 - 加粗
                legend = ax.legend(fontsize= 28, loc='best')
                for text in legend.get_texts():
                    text.set_fontweight('bold')
                
                # 修正：在x轴上添加批次大小标记，对所有子图都使用xlabel
                subplot_label = chr(97 + i)  # 97是字符'a'的ASCII码
                ax.set_xlabel(f'({subplot_label}) 批次大小 {bs}', fontsize= 28, fontweight='bold')
                
                # 移除原来可能在上方显示的标题
                ax.set_title("")
            
            # 设置整体标题
            fig.suptitle(f'带宽 {bandwidth}Mbps 下的{title}历史趋势', fontsize=18, fontweight='bold')
            
            # 调整子图之间的间距
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, hspace=0.35)
            
            plt.savefig(f'带宽{bandwidth}Mbps_{title}_历史趋势组合图.png', dpi=300, bbox_inches='tight')
            plt.close()

    # 新增函数：为同一带宽条件创建箱型图组合
    def plot_combined_boxplots_for_bandwidth(bandwidth):
        """
        为指定带宽创建箱型图组合，展示不同算法在各种批次大小下的数据分布
        
        参数:
        bandwidth - 要展示的带宽值，如10, 50, 100
        """
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 获取该带宽下的所有配置
        configs = []
        for bs in [10, 30, 60]:
            config = f'bw{bandwidth}_bs{bs}'
            if config in data:
                configs.append((bs, config))
        
        # 按批次大小排序
        configs.sort()
        
        # 如果没有配置，直接返回
        if not configs:
            print(f"警告: 带宽 {bandwidth}Mbps 没有可用配置，跳过绘图")
            return
        
        # 为每个指标创建一个大图，包含该带宽下所有批次大小的箱型图
        for metric_idx, (metric, title) in enumerate(zip(history_metrics, metric_titles)):
            # 创建子图网格
            fig, axes = plt.subplots(1, len(configs), figsize=(7*len(configs), 7), sharey=True)
            
            # 如果只有一个配置，确保axes是数组
            if len(configs) == 1:
                axes = [axes]
            
            # 为每个配置绘制箱型图
            for i, (bs, config) in enumerate(configs):
                ax = axes[i]
                
                # 获取该配置下的所有算法数据
                config_data = data[config]
                
                # 准备箱型图数据
                boxplot_data = []
                
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                        missing_data = True
                        break
                
                if missing_data:
                    ax.text(0.5, 0.5, f'数据缺失', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 为每个算法收集数据
                for algo, metrics_data in config_data.items():
                    # 只取部分数据点，减少密集度
                    values = metrics_data['history'][metric]
                    if len(values) > 10:  # 如果数据点过多，只取10个样本
                        step = len(values) // 10
                        sampled_values = values[::step]
                    else:
                        sampled_values = values
                    
                    for value in sampled_values:
                        boxplot_data.append({
                            'algorithm': algo,
                            'algorithm_zh': algo_names.get(algo, algo),
                            'value': value
                        })
                
                # 如果没有数据，显示提示信息
                if not boxplot_data:
                    ax.text(0.5, 0.5, f'无数据', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 转换为DataFrame
                boxplot_df = pd.DataFrame(boxplot_data)
                
                # 使用seaborn的箱型图，按算法分组
                sns.boxplot(x='algorithm_zh', y='value', data=boxplot_df, ax=ax, palette=colors, linewidth=2.5)
                
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
                        color='black',           # 点的颜色
                        marker=marker,           # 使用算法对应的标记形状
                        alpha=0.5,               # 透明度
                        s=35,                    # 增大点的大小
                        edgecolor='none',        # 边缘颜色
                        zorder=3                 # 确保点显示在箱型图上方
                    )
                
                # 设置子图标题
                ax.set_title(f'{title}', fontsize=16, fontweight='bold')
                
                # 增大刻度标签字体
                ax.tick_params(axis='both', which='major', labelsize= 28)
                
                # 只在第一个子图上显示Y轴标签
                if i == 0:
                    ax.set_ylabel(title, fontsize= 28, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
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
                
                # 只在第一个子图上添加图例
                if i == 0:
                    # 创建自定义图例，显示算法对应的标记形状
                    handles = []
                    for algo_zh in boxplot_df['algorithm_zh'].unique():
                        handles.append(plt.Line2D([0], [0], marker=markers.get(algo_zh, 'o'), 
                                                color='black', linestyle='None', 
                                                markersize=10, label=algo_zh))
                    
                    legend = ax.legend(handles=handles, title='算法', fontsize= 14, title_fontsize= 14, loc='upper right')
                    # 加粗图例文字
                    for text in legend.get_texts():
                        text.set_fontweight('bold')
                    legend.get_title().set_fontweight('bold')
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                
                # 在x轴上添加批次大小标记
                subplot_label = chr(97 + i)  # 97是字符'a'的ASCII码
                ax.set_xlabel(f'({subplot_label}) 批次大小 {bs}', fontsize= 28, fontweight='bold')
                
                # 加粗x轴标签
                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
            
            # 设置整体标题
            fig.suptitle(f'带宽 {bandwidth}Mbps 下的{title}分布', fontsize=18, fontweight='bold')
            
            # 调整子图之间的间距
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            
            plt.savefig(f'带宽{bandwidth}Mbps_{title}_箱型图组合.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 新增函数：为同一批次大小创建组合图
    def plot_combined_metrics_for_batch_size(batch_size):
        """
        为指定批次大小创建组合图，展示不同算法在各种带宽下的表现
        
        参数:
        batch_size - 要展示的批次大小，如10, 30, 60
        """
        # 指标列表
        metrics = ['accuracy', 'latency', 'resource']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 筛选指定批次大小的数据
        bs_data = metrics_df[metrics_df['batch_size'] == batch_size]
        
        # 如果没有数据，直接返回
        if len(bs_data) == 0:
            print(f"警告: 批次大小 {batch_size} 没有数据，跳过绘图")
            return
        
        # 获取该批次大小下的所有带宽
        bandwidths = sorted(bs_data['bandwidth'].unique())
        
        # 创建一个大图，包含所有指标的对比
        fig = plt.figure(figsize=(18,  28))
        
        # 定义子图布局 (3行1列)
        gs = fig.add_gridspec(3, 1, hspace=0.4)
        
        # 为每个指标创建一个子图
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = fig.add_subplot(gs[i, 0])
            
            # 为每个算法绘制一条线，展示不同带宽下的表现
            for algo in bs_data['algorithm_zh'].unique():
                algo_data = bs_data[bs_data['algorithm_zh'] == algo]
                
                # 按带宽排序
                algo_data = algo_data.sort_values('bandwidth')
                
                # 绘制线图
                ax.plot(algo_data['bandwidth'], algo_data[metric], 
                       marker=markers[algo], label=algo, 
                       color=colors[algo], linewidth=3.0, 
                       markersize=12, linestyle=line_styles[algo])
                
                # 添加数据标签
                for bw, val in zip(algo_data['bandwidth'], algo_data[metric]):
                    ax.text(bw, val, f'{val:.2f}', 
                           ha='center', va='bottom', 
                           fontsize= 28, color=colors[algo],
                           fontweight='bold')
            
            # 设置子图标题和标签
            ax.set_title(f'批次大小 {batch_size} 下的{metric_titles[i]}对比', fontsize=16, fontweight='bold')
            ax.set_ylabel(metric_titles[i], fontsize= 28, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # 增大刻度标签字体
            ax.tick_params(axis='both', which='major', labelsize= 28)
            
            # 添加图例 - 加粗
            legend = ax.legend(fontsize= 28, loc='best')
            for text in legend.get_texts():
                text.set_fontweight('bold')
            
            # 调整Y轴范围
            if metric == 'accuracy':
                ax.set_ylim([0.6, 0.85])
            elif metric == 'latency':
                ax.set_yscale('log')
                ax.set_ylim([50, 10000])
            elif metric == 'resource':
                ax.set_ylim([0.36, 0.43])
            
            # 修改x轴标签为带宽大小
            ax.set_xticks(bandwidths)
            ax.set_xticklabels([f'({chr(97 + idx)}) 带宽 {bw}Mbps' for idx, bw in enumerate(bandwidths)], fontweight='bold')
            
            # 只在最后一个子图设置x轴标签
            if i == len(metrics) - 1:
                ax.set_xlabel('带宽 (Mbps)', fontsize= 28, fontweight='bold')
        
        # 设置整体标题
        fig.suptitle(f'批次大小 {batch_size} 下的算法性能综合对比', fontsize=18, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 为整体标题留出空间
        
        # 保存图表
        plt.savefig(f'批次大小{batch_size}_综合性能对比.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 新增函数：为同一批次大小创建历史趋势组合图
    def plot_combined_history_for_batch_size(batch_size):
        """
        为指定批次大小创建历史趋势组合图，展示不同算法在各种带宽下的历史表现
        
        参数:
        batch_size - 要展示的批次大小，如10, 30, 60
        """
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 获取该批次大小下的所有配置
        configs = []
        for bw in [10, 50, 100]:
            config = f'bw{bw}_bs{batch_size}'
            if config in data:
                configs.append((bw, config))
        
        # 按带宽排序
        configs.sort()
        
        # 如果没有配置，直接返回
        if not configs:
            print(f"警告: 批次大小 {batch_size} 没有可用配置，跳过绘图")
            return
        
        # 为每个指标创建一个大图，包含该批次大小下所有带宽的历史趋势
        for metric_idx, (metric, title) in enumerate(zip(history_metrics, metric_titles)):
            # 创建子图网格
            fig, axes = plt.subplots(len(configs), 1, figsize=( 28, 6*len(configs)), sharex=True)
            
            # 如果只有一个配置，确保axes是数组
            if len(configs) == 1:
                axes = [axes]
            
            # 首先确定所有配置中最短的历史记录长度
            min_global_length = float('inf')
            for bw, config in configs:
                config_data = data[config]
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据")
                        missing_data = True
                        break
                
                if not missing_data:
                    # 找出当前配置中最短的历史记录长度
                    config_min_length = min(len(config_data[algo]['history'][metric]) for algo in config_data)
                    min_global_length = min(min_global_length, config_min_length)
            
            if min_global_length == float('inf'):
                print(f"警告: 批次大小 {batch_size} 下所有配置的历史记录都为空或缺失，跳过绘图")
                return
            
            # 为每个配置绘制历史趋势
            for i, (bw, config) in enumerate(configs):
                ax = axes[i]
                
                # 获取该配置下的所有算法数据
                config_data = data[config]
                
                # 准备历史数据
                history_data = []
                
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                        missing_data = True
                        break
                
                if missing_data:
                    ax.text(0.5, 0.5, f'数据缺失', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 为每个算法收集数据，使用全局最短长度
                for algo, metrics_data in config_data.items():
                    # 截取到全局最短长度
                    values = metrics_data['history'][metric][:min_global_length]
                    
                    # 只选择部分点，减少密集度
                    step = max(1, min_global_length // 10)  # 确保最多显示10个点
                    
                    # 使用平滑处理
                    if len(values) > 10:
                        # 使用滑动平均平滑数据
                        window_size = min(5, len(values) // 5)
                        if window_size > 1:
                            values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    
                    # 采样数据点
                    timestamps = list(range(len(values)))
                    sampled_indices = list(range(0, len(values), step))
                    
                    # 确保包含最后一个点
                    if len(values) - 1 not in sampled_indices:
                        sampled_indices.append(len(values) - 1)
                    
                    for idx in sampled_indices:
                        if idx < len(values):  # 确保索引有效
                            history_data.append({
                                'algorithm': algo,
                                'algorithm_zh': algo_names.get(algo, algo),
                                'timestamp': timestamps[idx],
                                'value': values[idx]
                            })
                
                history_df = pd.DataFrame(history_data)
                
                # 绘制时间序列趋势图
                for algo_zh in history_df['algorithm_zh'].unique():
                    algo_data = history_df[history_df['algorithm_zh'] == algo_zh]
                    ax.plot(algo_data['timestamp'], algo_data['value'], 
                           marker=markers.get(algo_zh, 'o'), label=algo_zh, 
                           color=colors.get(algo_zh, 'black'), linewidth=3.0,
                           markersize=10, linestyle=line_styles.get(algo_zh, '-'))
                
                # 设置y轴标签
                ax.set_ylabel(title, fontsize= 28, fontweight='bold')
                
                # 增大刻度标签字体
                ax.tick_params(axis='both', which='major', labelsize= 28)
                
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 调整Y轴范围
                if metric == 'latencies' and history_df['value'].max() > 1000:
                    ax.set_yscale('log')
                    ax.set_ylim([min(50, history_df['value'].min() * 0.9), history_df['value'].max() * 1.1])
                elif metric == 'accuracies':
                    ax.set_ylim([max(0.4, history_df['value'].min() * 0.9), min(1.0, history_df['value'].max() * 1.1)])
                elif metric == 'resources':
                    ax.set_ylim([history_df['value'].min() * 0.9, history_df['value'].max() * 1.1])
                
                # 添加图例 - 加粗
                legend = ax.legend(fontsize= 28, loc='best')
                for text in legend.get_texts():
                    text.set_fontweight('bold')
                
                # 修正：在x轴上添加带宽大小标记，对所有子图都使用xlabel
                subplot_label = chr(97 + i)  # 97是字符'a'的ASCII码
                ax.set_xlabel(f'({subplot_label}) 带宽 {bw}Mbps', fontsize= 28, fontweight='bold')
                
                # 移除原来可能在上方显示的标题
                ax.set_title("")
            
            # 设置整体标题
            fig.suptitle(f'批次大小 {batch_size} 下的{title}历史趋势', fontsize=18, fontweight='bold')
            
            # 调整子图之间的间距
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, hspace=0.35)
            
            plt.savefig(f'批次大小{batch_size}_{title}_历史趋势组合图.png', dpi=300, bbox_inches='tight')
            plt.close()

    # 新增函数：为同一批次大小创建箱型图组合
    def plot_combined_boxplots_for_batch_size(batch_size):
        """
        为指定批次大小创建箱型图组合，展示不同算法在各种带宽下的数据分布
        
        参数:
        batch_size - 要展示的批次大小，如10, 30, 60
        """
        # 指标列表
        history_metrics = ['accuracies', 'latencies', 'resources']
        metric_titles = ['准确率', '延迟 (ms)', '资源利用率']
        
        # 获取该批次大小下的所有配置
        configs = []
        for bw in [10, 50, 100]:
            config = f'bw{bw}_bs{batch_size}'
            if config in data:
                configs.append((bw, config))
        
        # 按带宽排序
        configs.sort()
        
        # 如果没有配置，直接返回
        if not configs:
            print(f"警告: 批次大小 {batch_size} 没有可用配置，跳过绘图")
            return
        
        # 为每个指标创建一个大图，包含该批次大小下所有带宽的箱型图
        for metric_idx, (metric, title) in enumerate(zip(history_metrics, metric_titles)):
            # 创建子图网格
            fig, axes = plt.subplots(1, len(configs), figsize=(7*len(configs), 7), sharey=True)
            
            # 如果只有一个配置，确保axes是数组
            if len(configs) == 1:
                axes = [axes]
            
            # 为每个配置绘制箱型图
            for i, (bw, config) in enumerate(configs):
                ax = axes[i]
                
                # 获取该配置下的所有算法数据
                config_data = data[config]
                
                # 准备箱型图数据
                boxplot_data = []
                
                # 检查每个算法是否都有历史数据
                missing_data = False
                for algo in config_data:
                    if 'history' not in config_data[algo] or metric not in config_data[algo]['history']:
                        print(f"警告: 算法 {algo} 在配置 {config} 中没有 {metric} 历史数据，跳过")
                        missing_data = True
                        break
                
                if missing_data:
                    ax.text(0.5, 0.5, f'数据缺失', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 为每个算法收集数据
                for algo, metrics_data in config_data.items():
                    # 只取部分数据点，减少密集度
                    values = metrics_data['history'][metric]
                    if len(values) > 10:  # 如果数据点过多，只取10个样本
                        step = len(values) // 10
                        sampled_values = values[::step]
                    else:
                        sampled_values = values
                    
                    for value in sampled_values:
                        boxplot_data.append({
                            'algorithm': algo,
                            'algorithm_zh': algo_names.get(algo, algo),
                            'value': value
                        })
                
                # 如果没有数据，显示提示信息
                if not boxplot_data:
                    ax.text(0.5, 0.5, f'无数据', ha='center', va='center', fontsize= 28, fontweight='bold', transform=ax.transAxes)
                    continue
                
                # 转换为DataFrame
                boxplot_df = pd.DataFrame(boxplot_data)
                
                # 使用seaborn的箱型图，按算法分组
                sns.boxplot(x='algorithm_zh', y='value', data=boxplot_df, ax=ax, palette=colors, linewidth=2.5)
                
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
                        color='black',           # 点的颜色
                        marker=marker,           # 使用算法对应的标记形状
                        alpha=0.5,               # 透明度
                        s=35,                    # 增大点的大小
                        edgecolor='none',        # 边缘颜色
                        zorder=3                 # 确保点显示在箱型图上方
                    )
                
                # 设置子图标题
                ax.set_title(f'{title}', fontsize=16, fontweight='bold')
                
                # 增大刻度标签字体
                ax.tick_params(axis='both', which='major', labelsize= 28)
                
                # 只在第一个子图上显示Y轴标签
                if i == 0:
                    ax.set_ylabel(title, fontsize= 28, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
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
                
                # 只在第一个子图上添加图例
                if i == 0:
                    # 创建自定义图例，显示算法对应的标记形状
                    handles = []
                    for algo_zh in boxplot_df['algorithm_zh'].unique():
                        handles.append(plt.Line2D([0], [0], marker=markers.get(algo_zh, 'o'), 
                                                color='black', linestyle='None', 
                                                markersize=10, label=algo_zh))
                    
                    legend = ax.legend(handles=handles, title='算法', fontsize= 14, title_fontsize= 14, loc='upper right')
                    # 加粗图例文字
                    for text in legend.get_texts():
                        text.set_fontweight('bold')
                    legend.get_title().set_fontweight('bold')
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                
                # 在x轴上添加带宽大小标记
                subplot_label = chr(97 + i)  # 97是字符'a'的ASCII码
                ax.set_xlabel(f'({subplot_label}) 带宽 {bw}Mbps', fontsize= 28, fontweight='bold')
                
                # 加粗x轴标签
                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
            
            # 设置整体标题
            fig.suptitle(f'批次大小 {batch_size} 下的{title}分布', fontsize=18, fontweight='bold')
            
            # 调整子图之间的间距
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            
            plt.savefig(f'批次大小{batch_size}_{title}_箱型图组合.png', dpi=300, bbox_inches='tight')
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
    colors = {'主动推理': '#1f77b4', '轮询算法': '#ff7f0e', 'DQN': '#2ca02c'}
    line_styles = {'主动推理': '-', '轮询算法': '--', 'DQN': '-.'}

    # 添加中文算法名称
    metrics_df['algorithm_zh'] = metrics_df['algorithm'].map(algo_names)

    # 创建输出目录
    os.makedirs('plots_bold_font', exist_ok=True)
    os.chdir('plots_bold_font')

    print("开始绘制加粗字体图表...")

    # 为每个带宽创建组合图
    for bw in [10, 50, 100]:
        print(f"为带宽 {bw}Mbps 创建组合图...")
        plot_combined_metrics_for_bandwidth(bw)
        plot_combined_history_for_bandwidth(bw)
        plot_combined_boxplots_for_bandwidth(bw)
    
    # 为每个批次大小创建组合图
    for bs in [10, 30, 60]:
        print(f"为批次大小 {bs} 创建组合图...")
        plot_combined_metrics_for_batch_size(bs)
        plot_combined_history_for_batch_size(bs)
        plot_combined_boxplots_for_batch_size(bs)

    print("所有组合图绘制完成！保存在 plots_bold_font 目录下")

except Exception as e:
    print(f"加载或绘图失败: {e}")
    import traceback
    traceback.print_exc()