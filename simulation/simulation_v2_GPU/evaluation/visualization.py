import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

class ResultVisualizer:
    def __init__(self, output_dir="results"):
        """
        初始化可视化工具
        
        参数:
        - output_dir: 输出图像的目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置更美观的可视化样式
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        # 设置更优雅的调色板
        self.palette = sns.color_palette("muted")
        
        # 为不同算法定义标记和颜色
        self.algorithm_markers = {
            "active_reasoning": "o",  # 圆形
            "round_robin": "s",       # 方形
            "dqn": "^"                # 三角形
        }
        
        self.algorithm_colors = {
            "active_reasoning": self.palette[0],  # 蓝色
            "round_robin": self.palette[1],       # 橙色
            "dqn": self.palette[2]                # 绿色
        }
        
        self.algorithm_names = {
            "active_reasoning": "Active Reasoning",
            "round_robin": "Round Robin",
            "dqn": "DQN"
        }
        
        # 生成时间戳用于文件命名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def visualize_bar_charts(self, results, bandwidths, batch_sizes):
        """
        为每个带宽和批次大小组合生成单独的条形图
        
        参数:
        - results: 评估结果
        - bandwidths: 带宽列表
        - batch_sizes: 批次大小列表
        """
        metrics = ["avg_accuracy", "avg_latency", "avg_resource", "convergence_time"]
        metric_titles = {
            "avg_accuracy": "Average Accuracy",
            "avg_latency": "Average Latency (ms)",
            "avg_resource": "Average Resource Usage",
            "convergence_time": "Convergence Time (s)"
        }
        
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 计算总图表数
        total_charts = len(bandwidths) * len(batch_sizes) * len(metrics)
        
        # 使用tqdm创建进度条
        with tqdm(total=total_charts, desc="Generating bar charts") as pbar:
            # 为每个带宽和批次大小组合生成单独的图表
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    # 创建DataFrame用于seaborn可视化
                    data = []
                    for metric in metrics:
                        for algo in algorithms:
                            value = results[key][algo][metric]
                            # 如果是收敛时间且值为None，设为最大迭代次数
                            if metric == "convergence_time" and value is None:
                                value = 1000
                            
                            data.append({
                                "Metric": metric_titles[metric],
                                "Algorithm": self.algorithm_names[algo],
                                "Value": value
                            })
                    
                    df = pd.DataFrame(data)
                    
                    # 为每个指标创建单独的图表
                    for metric in metrics:
                        plt.figure(figsize=(10, 6))
                        metric_df = df[df["Metric"] == metric_titles[metric]]
                        
                        # 使用seaborn创建美观的条形图
                        ax = sns.barplot(
                            x="Algorithm", 
                            y="Value", 
                            data=metric_df,
                            palette=[self.algorithm_colors[algo] for algo in algorithms],
                            edgecolor='black',
                            linewidth=1
                        )
                        
                        # 为条形添加标记
                        for i, bar in enumerate(ax.patches):
                            algo = algorithms[i]
                            x = bar.get_x() + bar.get_width()/2
                            y = bar.get_height() * 1.01
                            ax.plot(x, y, marker=self.algorithm_markers[algo], 
                                   color='black', markersize=8, zorder=10)
                        
                        # 添加数值标签
                        for i, bar in enumerate(ax.patches):
                            value = bar.get_height()
                            ax.annotate(
                                f'{value:.2f}',
                                (bar.get_x() + bar.get_width() / 2, value),
                                ha='center', va='bottom',
                                size=10, xytext=(0, 3),
                                textcoords='offset points'
                            )
                        
                        plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s, BS={batch_size})")
                        plt.ylabel(metric_titles[metric])
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # 保存图表
                        filename = f"{metric}_{key}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # 更新进度条
                        pbar.update(1)
    
    def visualize_convergence(self, results, bandwidths, batch_sizes):
        """
        可视化算法收敛过程
        
        参数:
        - results: 评估结果
        - bandwidths: 带宽列表
        - batch_sizes: 批次大小列表
        """
        metrics = ["accuracies", "latencies", "resources"]
        metric_titles = {
            "accuracies": "Accuracy over Time",
            "latencies": "Latency over Time (ms)",
            "resources": "Resource Usage over Time"
        }
        
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 计算总图表数
        total_charts = len(bandwidths) * len(batch_sizes) * len(metrics)
        
        # 使用tqdm创建进度条
        with tqdm(total=total_charts, desc="Generating convergence plots") as pbar:
            # 为每个带宽和批次大小组合生成单独的图表
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    for metric in metrics:
                        plt.figure(figsize=(12, 7))
                        
                        for algo in algorithms:
                            history = results[key][algo]["history"][metric]
                            
                            # 每10个点取一个以减少噪声
                            x = list(range(0, len(history), 10))
                            y = [np.mean(history[i:i+10]) for i in range(0, len(history), 10)]
                            
                            # 使用seaborn的lineplot创建更美观的线图
                            plt.plot(
                                x, y,
                                marker=self.algorithm_markers[algo],
                                color=self.algorithm_colors[algo],
                                label=self.algorithm_names[algo],
                                markevery=max(1, len(x)//10),  # 每10个点放一个标记
                                markersize=8,
                                linewidth=2,
                                alpha=0.8
                            )
                        
                        plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s, BS={batch_size})")
                        plt.xlabel("Iteration")
                        plt.ylabel(metric_titles[metric])
                        plt.grid(True, linestyle='--', alpha=0.7)
                        
                        # 添加图例，位置自动调整到最佳位置
                        plt.legend(loc='best', frameon=True, framealpha=0.9)
                        
                        # 添加水平参考线表示平均性能
                        for algo in algorithms:
                            history = results[key][algo]["history"][metric]
                            avg_value = np.mean(history)
                            plt.axhline(y=avg_value, color=self.algorithm_colors[algo], 
                                      linestyle='--', alpha=0.4)
                        
                        plt.tight_layout()
                        
                        # 保存图表
                        filename = f"{metric}_convergence_{key}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # 更新进度条
                        pbar.update(1)
    
    def visualize_bandwidth_impact(self, results, bandwidths, batch_size):
        """
        可视化带宽对性能的影响
        
        参数:
        - results: 评估结果
        - bandwidths: 带宽列表
        - batch_size: 固定的批次大小
        """
        metrics = ["avg_accuracy", "avg_latency", "avg_resource"]
        metric_titles = {
            "avg_accuracy": "Average Accuracy vs Bandwidth",
            "avg_latency": "Average Latency vs Bandwidth",
            "avg_resource": "Average Resource Usage vs Bandwidth"
        }
        
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for metric in metrics:
            for algo in algorithms:
                for bw in bandwidths:
                    key = f"bw{bw}_bs{batch_size}"
                    value = results[key][algo][metric]
                    
                    data.append({
                        "Metric": metric_titles[metric],
                        "Algorithm": self.algorithm_names[algo],
                        "Bandwidth": bw,
                        "Value": value
                    })
        
        df = pd.DataFrame(data)
        
        # 为每个指标创建单独的图表
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            metric_df = df[df["Metric"] == metric_titles[metric]]
            
            # 使用seaborn创建美观的线图
            ax = sns.lineplot(
                x="Bandwidth", 
                y="Value", 
                hue="Algorithm", 
                style="Algorithm",
                markers=True,
                dashes=False,
                data=metric_df,
                palette=[self.algorithm_colors[algo] for algo in algorithms],
                markersize=10,
                linewidth=2.5
            )
            
            # 自定义标记
            for i, algo in enumerate(algorithms):
                ax.lines[i].set_marker(self.algorithm_markers[algo])
            
            plt.title(f"{metric_titles[metric]} (BS={batch_size})")
            plt.xlabel("Bandwidth (MB/s)")
            plt.ylabel(metric.replace("avg_", "").capitalize())
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加图例，位置自动调整到最佳位置
            plt.legend(loc='best', frameon=True, framealpha=0.9)
            
            # 设置x轴刻度
            plt.xticks(bandwidths)
            
            # 添加数据点标签
            for algo in algorithms:
                algo_data = metric_df[metric_df["Algorithm"] == self.algorithm_names[algo]]
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f'{row["Value"]:.2f}',
                        (row["Bandwidth"], row["Value"]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8
                    )
            
            plt.tight_layout()
            
            # 保存图表
            filename = f"{metric}_vs_bandwidth_bs{batch_size}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_batch_size_impact(self, results, bandwidth, batch_sizes):
        """
        可视化批次大小对性能的影响
        
        参数:
        - results: 评估结果
        - bandwidth: 固定的带宽
        - batch_sizes: 批次大小列表
        """
        metrics = ["avg_accuracy", "avg_latency", "avg_resource"]
        metric_titles = {
            "avg_accuracy": "Average Accuracy vs Batch Size",
            "avg_latency": "Average Latency vs Batch Size",
            "avg_resource": "Average Resource Usage vs Batch Size"
        }
        
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for metric in metrics:
            for algo in algorithms:
                for bs in batch_sizes:
                    key = f"bw{bandwidth}_bs{bs}"
                    value = results[key][algo][metric]
                    
                    data.append({
                        "Metric": metric_titles[metric],
                        "Algorithm": self.algorithm_names[algo],
                        "Batch Size": bs,
                        "Value": value
                    })
        
        df = pd.DataFrame(data)
        
        # 为每个指标创建单独的图表
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            metric_df = df[df["Metric"] == metric_titles[metric]]
            
            # 使用seaborn创建美观的线图
            ax = sns.lineplot(
                x="Batch Size", 
                y="Value", 
                hue="Algorithm", 
                style="Algorithm",
                markers=True,
                dashes=False,
                data=metric_df,
                palette=[self.algorithm_colors[algo] for algo in algorithms],
                markersize=10,
                linewidth=2.5
            )
            
            # 自定义标记
            for i, algo in enumerate(algorithms):
                ax.lines[i].set_marker(self.algorithm_markers[algo])
            
            plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s)")
            plt.xlabel("Batch Size")
            plt.ylabel(metric.replace("avg_", "").capitalize())
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加图例，位置自动调整到最佳位置
            plt.legend(loc='best', frameon=True, framealpha=0.9)
            
            # 设置x轴刻度
            plt.xticks(batch_sizes)
            
            # 添加数据点标签
            for algo in algorithms:
                algo_data = metric_df[metric_df["Algorithm"] == self.algorithm_names[algo]]
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f'{row["Value"]:.2f}',
                        (row["Batch Size"], row["Value"]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8
                    )
            
            plt.tight_layout()
            
            # 保存图表
            filename = f"{metric}_vs_batchsize_bw{bandwidth}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    # ================ 模型卸载相关可视化 ================
    
    def visualize_model_transfer_time(self, results, bandwidths, batch_sizes):
        """可视化模型传输时间"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for bandwidth in bandwidths:
            for batch_size in batch_sizes:
                key = f"bw{bandwidth}_bs{batch_size}"
                
                for algo in algorithms:
                    # 计算平均传输时间（秒）
                    avg_transfer = results[key][algo].get("avg_transfer_time", 0)
                    
                    data.append({
                        "Bandwidth": bandwidth,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Transfer Time": avg_transfer
                    })
        
        df = pd.DataFrame(data)
        
        # 创建分组条形图
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating model transfer time charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    plt.figure(figsize=(10, 6))
                    
                    # 筛选当前带宽和批次大小的数据
                    current_df = df[(df["Bandwidth"] == bandwidth) & (df["Batch Size"] == batch_size)]
                    
                    # 使用seaborn创建美观的条形图
                    ax = sns.barplot(
                        x="Allocator", 
                        y="Transfer Time",
                        data=current_df,
                        palette=[self.algorithm_colors[algo] for algo in algorithms],
                        edgecolor='black',
                        linewidth=1
                    )
                    
                    # 为条形添加标记
                    for i, bar in enumerate(ax.patches):
                        algo = algorithms[i]
                        x = bar.get_x() + bar.get_width()/2
                        y = bar.get_height() * 1.01
                        ax.plot(x, y, marker=self.algorithm_markers[algo], 
                               color='black', markersize=8, zorder=10)
                    
                    # 添加数值标签
                    for i, bar in enumerate(ax.patches):
                        value = bar.get_height()
                        ax.annotate(
                            f'{value:.2f}s',
                            (bar.get_x() + bar.get_width() / 2, value),
                            ha='center', va='bottom',
                            size=10, xytext=(0, 3),
                            textcoords='offset points'
                        )
                    
                    plt.title(f"Average Model Transfer Time (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.ylabel("Transfer Time (seconds)")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"model_transfer_time_bw{bandwidth}_bs{batch_size}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    pbar.update(1)
        
        # 创建带宽对比图
        plt.figure(figsize=(14, 8))
        
        # 对数据进行透视
        pivot_df = df.pivot_table(
            index="Allocator", 
            columns="Bandwidth", 
            values="Transfer Time",
            aggfunc="mean"
        )
        
        # 绘制热力图
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt=".2f", 
            cmap="YlGnBu", 
            linewidths=0.5,
            cbar_kws={'label': 'Transfer Time (seconds)'}
        )
        
        plt.title("Model Transfer Time by Allocator and Bandwidth")
        plt.tight_layout()
        
        # 保存图表
        filename = f"model_transfer_time_heatmap_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_model_deployment_frequency(self, results, bandwidths, batch_sizes):
        """可视化模型部署频率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建大的组合图
        fig, axes = plt.subplots(len(bandwidths), len(batch_sizes), figsize=(18, 12))
        
        if len(bandwidths) == 1 and len(batch_sizes) == 1:
            axes = np.array([[axes]])
        elif len(bandwidths) == 1:
            axes = np.array([axes])
        elif len(batch_sizes) == 1:
            axes = np.array([axes]).T
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating model deployment frequency charts") as pbar:
            for i, bandwidth in enumerate(bandwidths):
                for j, batch_size in enumerate(batch_sizes):
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    for algo in algorithms:
                        # 获取模型部署历史
                        deploy_history = results[key][algo].get("deployments", [])
                        
                        if deploy_history:
                            # 对历史数据进行滑动窗口平均，每10个点取一个平均值
                            window_size = 10
                            x = list(range(0, len(deploy_history), window_size))
                            y = [np.mean(deploy_history[i:i+window_size]) for i in range(0, len(deploy_history), window_size)]
                            
                            axes[i, j].plot(
                                x, y,
                                marker=self.algorithm_markers[algo],
                                color=self.algorithm_colors[algo],
                                label=self.algorithm_names[algo],
                                markevery=max(1, len(x)//10),
                                markersize=6,
                                linewidth=1.5
                            )
                    
                    axes[i, j].set_title(f"BW={bandwidth} MB/s, BS={batch_size}")
                    axes[i, j].set_xlabel("Iteration")
                    axes[i, j].set_ylabel("Deployments/Iteration")
                    axes[i, j].grid(True, linestyle='--', alpha=0.7)
                    axes[i, j].legend(loc='upper right', fontsize=8)
                    
                    pbar.update(1)
        
        plt.suptitle("Model Deployment Frequency", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"model_deployment_frequency_matrix_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_cache_hit_rate(self, results, bandwidths, batch_sizes):
        """可视化模型缓存命中率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for bandwidth in bandwidths:
            for batch_size in batch_sizes:
                key = f"bw{bandwidth}_bs{batch_size}"
                
                for algo in algorithms:
                    # 获取缓存命中率
                    hit_rate = results[key][algo].get("cache_hit_rate", 0) * 100  # 转换为百分比
                    
                    data.append({
                        "Bandwidth": bandwidth,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Cache Hit Rate": hit_rate
                    })
        
        df = pd.DataFrame(data)
        
        # 创建综合对比图
        plt.figure(figsize=(14, 10))
        
        # 创建子图
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # 1. 所有算法的缓存命中率 (按带宽分组)
        ax1 = plt.subplot(gs[0, 0])
        sns.barplot(
            x="Allocator", 
            y="Cache Hit Rate", 
            hue="Bandwidth",
            data=df,
            palette="Blues_d",
            ax=ax1,
            edgecolor='black',
            linewidth=1
        )
        ax1.set_title("Cache Hit Rate by Bandwidth")
        ax1.set_ylabel("Cache Hit Rate (%)")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 所有算法的缓存命中率 (按批次大小分组)
        ax2 = plt.subplot(gs[0, 1])
        sns.barplot(
            x="Allocator", 
            y="Cache Hit Rate", 
            hue="Batch Size",
            data=df,
            palette="Greens_d",
            ax=ax2,
            edgecolor='black',
            linewidth=1
        )
        ax2.set_title("Cache Hit Rate by Batch Size")
        ax2.set_ylabel("Cache Hit Rate (%)")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. 带宽对缓存命中率的影响 (按算法分组)
        ax3 = plt.subplot(gs[1, 0])
        sns.lineplot(
            x="Bandwidth", 
            y="Cache Hit Rate", 
            hue="Allocator",
            style="Allocator",
            markers=True,
            dashes=False,
            data=df,
            palette=[self.algorithm_colors[algo] for algo in algorithms],
            ax=ax3,
            markersize=10,
            linewidth=2
        )
        # 自定义标记
        for i, algo in enumerate(algorithms):
            ax3.lines[i].set_marker(self.algorithm_markers[algo])
        ax3.set_title("Cache Hit Rate vs Bandwidth")
        ax3.set_xlabel("Bandwidth (MB/s)")
        ax3.set_ylabel("Cache Hit Rate (%)")
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 批次大小对缓存命中率的影响 (按算法分组)
        ax4 = plt.subplot(gs[1, 1])
        sns.lineplot(
            x="Batch Size", 
            y="Cache Hit Rate", 
            hue="Allocator",
            style="Allocator",
            markers=True,
            dashes=False,
            data=df,
            palette=[self.algorithm_colors[algo] for algo in algorithms],
            ax=ax4,
            markersize=10,
            linewidth=2
        )
        # 自定义标记
        for i, algo in enumerate(algorithms):
            ax4.lines[i].set_marker(self.algorithm_markers[algo])
        ax4.set_title("Cache Hit Rate vs Batch Size")
        ax4.set_xlabel("Batch Size")
        ax4.set_ylabel("Cache Hit Rate (%)")
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle("Model Cache Hit Rate Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"model_cache_hit_rate_analysis_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_bandwidth_utilization(self, results, bandwidths, batch_sizes):
        """可视化带宽利用效率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for batch_size in batch_sizes:
            for algo in algorithms:
                utilization_rates = []
                for bw in bandwidths:
                    key = f"bw{bw}_bs{batch_size}"
                    # 计算带宽利用率
                    utilization = results[key][algo].get("bandwidth_utilization", 0) * 100  # 转换为百分比
                    
                    data.append({
                        "Bandwidth": bw,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Utilization": utilization
                    })
        
        df = pd.DataFrame(data)
        
        # 创建热力图
        plt.figure(figsize=(15, 10))
        
        # 对数据进行透视
        for i, bs in enumerate(batch_sizes):
            plt.subplot(len(batch_sizes), 1, i+1)
            
            # 筛选当前批次大小的数据
            current_df = df[df["Batch Size"] == bs]
            
            # 透视数据
            pivot_df = current_df.pivot_table(
                index="Allocator", 
                columns="Bandwidth", 
                values="Utilization",
                aggfunc="mean"
            )
            
            # 绘制热力图
            sns.heatmap(
                pivot_df, 
                annot=True, 
                fmt=".1f", 
                cmap="YlGnBu", 
                linewidths=0.5,
                cbar_kws={'label': 'Bandwidth Utilization (%)'}
            )
            
            plt.title(f"Bandwidth Utilization (Batch Size = {bs})")
        
        plt.suptitle("Bandwidth Utilization Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"bandwidth_utilization_heatmap_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建线图
        plt.figure(figsize=(12, 8))
        
        for algo in algorithms:
            algo_df = df[df["Allocator"] == self.algorithm_names[algo]]
            
            # 按批次大小分组计算平均值
            mean_df = algo_df.groupby("Bandwidth")["Utilization"].mean().reset_index()
            
            plt.plot(
                mean_df["Bandwidth"], 
                mean_df["Utilization"],
                marker=self.algorithm_markers[algo],
                color=self.algorithm_colors[algo],
                label=self.algorithm_names[algo],
                linewidth=2.5,
                markersize=10
            )
        
        plt.title("Average Bandwidth Utilization by Algorithm")
        plt.xlabel("Bandwidth (MB/s)")
        plt.ylabel("Bandwidth Utilization (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.xticks(bandwidths)
        
        # 添加数据点标签
        for algo in algorithms:
            algo_df = df[df["Allocator"] == self.algorithm_names[algo]]
            mean_df = algo_df.groupby("Bandwidth")["Utilization"].mean().reset_index()
            
            for _, row in mean_df.iterrows():
                plt.annotate(
                    f'{row["Utilization"]:.1f}%',
                    (row["Bandwidth"], row["Utilization"]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=9
                )
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"bandwidth_utilization_line_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_total_transfer_data(self, results, bandwidths, batch_sizes):
        """可视化总传输数据量"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for bandwidth in bandwidths:
            for batch_size in batch_sizes:
                key = f"bw{bandwidth}_bs{batch_size}"
                
                for algo in algorithms:
                    # 获取总传输数据量 (MB)
                    data_volume = results[key][algo].get("total_transfer_data", 0)
                    
                    data.append({
                        "Bandwidth": bandwidth,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Data Volume": data_volume
                    })
        
        df = pd.DataFrame(data)
        
        # 创建综合对比图
        plt.figure(figsize=(16, 12))
        
        # 创建子图
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # 1. 按带宽和算法的总传输数据量
        ax1 = plt.subplot(gs[0, 0])
        sns.barplot(
            x="Allocator", 
            y="Data Volume", 
            hue="Bandwidth",
            data=df,
            palette="Blues_d",
            ax=ax1,
            edgecolor='black',
            linewidth=1
        )
        ax1.set_title("Total Transfer Data by Bandwidth")
        ax1.set_ylabel("Data Volume (MB)")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 按批次大小和算法的总传输数据量
        ax2 = plt.subplot(gs[0, 1])
        sns.barplot(
            x="Allocator", 
            y="Data Volume", 
            hue="Batch Size",
            data=df,
            palette="Greens_d",
            ax=ax2,
            edgecolor='black',
            linewidth=1
        )
        ax2.set_title("Total Transfer Data by Batch Size")
        ax2.set_ylabel("Data Volume (MB)")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. 带宽对总传输数据量的影响
        ax3 = plt.subplot(gs[1, 0])
        sns.lineplot(
            x="Bandwidth", 
            y="Data Volume", 
            hue="Allocator",
            style="Allocator",
            markers=True,
            dashes=False,
            data=df,
            palette=[self.algorithm_colors[algo] for algo in algorithms],
            ax=ax3,
            markersize=10,
            linewidth=2
        )
        # 自定义标记
        for i, algo in enumerate(algorithms):
            ax3.lines[i].set_marker(self.algorithm_markers[algo])
        ax3.set_title("Total Transfer Data vs Bandwidth")
        ax3.set_xlabel("Bandwidth (MB/s)")
        ax3.set_ylabel("Data Volume (MB)")
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 批次大小对总传输数据量的影响
        ax4 = plt.subplot(gs[1, 1])
        sns.lineplot(
            x="Batch Size", 
            y="Data Volume", 
            hue="Allocator",
            style="Allocator",
            markers=True,
            dashes=False,
            data=df,
            palette=[self.algorithm_colors[algo] for algo in algorithms],
            ax=ax4,
            markersize=10,
            linewidth=2
        )
        # 自定义标记
        for i, algo in enumerate(algorithms):
            ax4.lines[i].set_marker(self.algorithm_markers[algo])
        ax4.set_title("Total Transfer Data vs Batch Size")
        ax4.set_xlabel("Batch Size")
        ax4.set_ylabel("Data Volume (MB)")
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle("Total Model Transfer Data Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"total_transfer_data_analysis_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_latency_composition(self, results, bandwidths, batch_sizes):
        """可视化延迟构成"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建DataFrame用于seaborn可视化
        data = []
        for bandwidth in bandwidths:
            for batch_size in batch_sizes:
                key = f"bw{bandwidth}_bs{batch_size}"
                
                for algo in algorithms:
                    # 获取推理延迟和传输延迟
                    inference = results[key][algo].get("avg_inference_latency", 0)
                    transfer = results[key][algo].get("avg_transfer_latency", 0)
                    total = inference + transfer
                    
                    # 添加推理延迟数据
                    data.append({
                        "Bandwidth": bandwidth,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Latency Type": "Inference Latency",
                        "Latency": inference,
                        "Percentage": (inference / total * 100) if total > 0 else 0
                    })
                    
                    # 添加传输延迟数据
                    data.append({
                        "Bandwidth": bandwidth,
                        "Batch Size": batch_size,
                        "Allocator": self.algorithm_names[algo],
                        "Latency Type": "Transfer Latency",
                        "Latency": transfer,
                        "Percentage": (transfer / total * 100) if total > 0 else 0
                    })
        
        df = pd.DataFrame(data)
        
        # 创建堆叠条形图
        plt.figure(figsize=(15, 10))
        
        for i, bw in enumerate(bandwidths):
            for j, bs in enumerate(batch_sizes):
                plt_idx = i * len(batch_sizes) + j + 1
                plt.subplot(len(bandwidths), len(batch_sizes), plt_idx)
                
                # 筛选当前带宽和批次大小的数据
                current_df = df[(df["Bandwidth"] == bw) & (df["Batch Size"] == bs)]
                
                # 创建透视表
                pivot_df = current_df.pivot_table(
                    index="Allocator", 
                    columns="Latency Type", 
                    values="Latency",
                    aggfunc="sum"
                )
                
                # 绘制堆叠条形图
                pivot_df.plot(
                    kind='bar', 
                    stacked=True, 
                    ax=plt.gca(),
                    color=['skyblue', 'coral'],
                    edgecolor='black',
                    linewidth=1
                )
                
                # 添加总延迟标签
                for k, allocator in enumerate(pivot_df.index):
                    total = pivot_df.loc[allocator].sum()
                    plt.annotate(
                        f'{total:.1f}ms',
                        (k, total),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8
                    )
                
                plt.title(f"BW={bw} MB/s, BS={bs}")
                plt.xlabel("")
                plt.ylabel("Latency (ms)")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 只在第一个子图显示图例
                if i == 0 and j == 0:
                    plt.legend(title="Latency Component")
                else:
                    plt.legend().remove()
        
        plt.suptitle("Latency Composition Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"latency_composition_matrix_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建延迟占比饼图
        plt.figure(figsize=(15, 10))
        
        for i, algo in enumerate(algorithms):
            plt.subplot(1, 3, i+1)
            
            # 筛选当前算法的数据
            algo_df = df[df["Allocator"] == self.algorithm_names[algo]]
            
            # 计算平均占比
            avg_percentages = algo_df.groupby("Latency Type")["Percentage"].mean()
            
            # 绘制饼图
            plt.pie(
                avg_percentages, 
                labels=avg_percentages.index, 
                autopct='%1.1f%%',
                colors=['skyblue', 'coral'],
                startangle=90,
                shadow=True,
                explode=(0.05, 0)
            )
            
            plt.title(f"{self.algorithm_names[algo]}")
        
        plt.suptitle("Average Latency Composition by Algorithm", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"latency_composition_pie_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_offloading_decision_heatmap(self, results, bandwidths, batch_sizes):
        """可视化模型卸载决策热力图"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建大的组合图
        fig, axes = plt.subplots(len(algorithms), len(bandwidths), figsize=(5*len(bandwidths), 4*len(algorithms)))
        
        if len(algorithms) == 1 and len(bandwidths) == 1:
            axes = np.array([[axes]])
        elif len(algorithms) == 1:
            axes = np.array([axes])
        elif len(bandwidths) == 1:
            axes = np.array([axes]).T
        
        with tqdm(total=len(bandwidths)*len(algorithms), desc="Generating offloading decision heatmaps") as pbar:
            for i, algo in enumerate(algorithms):
                for j, bandwidth in enumerate(bandwidths):
                    # 选择批次大小 (默认使用最大批次大小)
                    batch_size = max(batch_sizes)
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    # 获取卸载决策数据
                    decision_map = results[key][algo].get("offloading_decision_map", None)
                    
                    if decision_map is not None and isinstance(decision_map, np.ndarray) and len(decision_map.shape) == 2:
                        # 使用更高级的热力图
                        sns.heatmap(
                            decision_map, 
                            cmap="YlGnBu", 
                            annot=False,
                            ax=axes[i, j],
                            cbar_kws={'label': 'Offloading Decision (1=offload, 0=use cached)'}
                        )
                        
                        axes[i, j].set_title(f"{self.algorithm_names[algo]}: BW={bandwidth} MB/s")
                        axes[i, j].set_xlabel("Task Feature 2")
                        axes[i, j].set_ylabel("Task Feature 1")
                    else:
                        axes[i, j].text(0.5, 0.5, "No data available", 
                                       ha='center', va='center', fontsize=12)
                        axes[i, j].set_title(f"{self.algorithm_names[algo]}: BW={bandwidth} MB/s")
                    
                    pbar.update(1)
        
        plt.suptitle("Model Offloading Decision Heatmaps", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"offloading_decision_heatmap_matrix_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_model_offloading_metrics(self, results, bandwidths, batch_sizes):
        """可视化所有模型卸载相关指标"""
        print("Visualizing model offloading metrics...")
        
        # 1. 模型传输时间
        self.visualize_model_transfer_time(results, bandwidths, batch_sizes)
        
        # 2. 模型部署频率
        self.visualize_model_deployment_frequency(results, bandwidths, batch_sizes)
        
        # 3. 模型缓存命中率
        self.visualize_cache_hit_rate(results, bandwidths, batch_sizes)
        
        # 4. 带宽利用效率
        self.visualize_bandwidth_utilization(results, bandwidths, batch_sizes)
        
        # 5. 总传输数据量
        self.visualize_total_transfer_data(results, bandwidths, batch_sizes)
        
        # 6. 延迟构成
        self.visualize_latency_composition(results, bandwidths, batch_sizes)
        
        # 7. 模型卸载决策热力图
        self.visualize_offloading_decision_heatmap(results, bandwidths, batch_sizes)
        
        # 8. 创建模型卸载综合仪表盘
        self.create_offloading_dashboard(results, bandwidths, batch_sizes)
        
        print("Model offloading visualization completed!")
    
    def create_offloading_dashboard(self, results, bandwidths, batch_sizes):
        """创建模型卸载综合仪表盘"""
        # 选择一个代表性的配置进行展示
        bandwidth = bandwidths[len(bandwidths)//2]
        batch_size = batch_sizes[len(batch_sizes)//2]
        key = f"bw{bandwidth}_bs{batch_size}"
        
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        # 创建汇总数据
        summary_data = []
        for algo in algorithms:
            # 收集关键指标
            cache_hit_rate = results[key][algo].get("cache_hit_rate", 0) * 100
            avg_transfer_time = results[key][algo].get("avg_transfer_time", 0)
            total_transfer_data = results[key][algo].get("total_transfer_data", 0)
            bandwidth_utilization = results[key][algo].get("bandwidth_utilization", 0) * 100
            avg_accuracy = results[key][algo].get("avg_accuracy", 0)
            avg_latency = results[key][algo].get("avg_latency", 0)
            
            # 计算传输开销 (传输延迟占总延迟的百分比)
            avg_inference_latency = results[key][algo].get("avg_inference_latency", 0)
            avg_transfer_latency = results[key][algo].get("avg_transfer_latency", 0)
            total_latency = avg_inference_latency + avg_transfer_latency
            transmission_overhead = (avg_transfer_latency / total_latency * 100) if total_latency > 0 else 0
            
            summary_data.append({
                "Allocator": self.algorithm_names[algo],
                "Cache Hit Rate": cache_hit_rate,
                "Avg Transfer Time": avg_transfer_time,
                "Total Transfer Data": total_transfer_data,
                "Bandwidth Utilization": bandwidth_utilization,
                "Avg Accuracy": avg_accuracy,
                "Avg Latency": avg_latency,
                "Transmission Overhead": transmission_overhead
            })
        
        df = pd.DataFrame(summary_data)
        
        # 创建仪表盘
        plt.figure(figsize=(20, 15))
        
        # 定义算法标记
        ALLOCATOR_MARKERS = {self.algorithm_names[algo]: self.algorithm_markers[algo] for algo in algorithms}
        
        # 创建子图网格
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # 1. 缓存命中率和传输开销对比
        axes = []
        axes.append(plt.subplot(gs[0, 0]))
        axes.append(plt.subplot(gs[0, 1]))
        axes.append(plt.subplot(gs[1, 0]))
        axes.append(plt.subplot(gs[1, 1]))
        
        # 1. 缓存命中率对比
        try:
            bars = sns.barplot(x="Allocator", y="Cache Hit Rate", data=df, ax=axes[0, 0])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    allocator = df["Allocator"].iloc[i]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[0, 0].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
            
            # 添加数值标签
            for i, p in enumerate(bars.patches):
                axes[0, 0].annotate(f'{p.get_height():.1f}%', 
                                  (p.get_x() + p.get_width()/2, p.get_height()), 
                                  ha='center', va='bottom', fontsize=10)
            
            axes[0, 0].set_title("Cache Hit Rate")
            axes[0, 0].set_ylabel("Hit Rate (%)")
            axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制缓存命中率图表时出错: {e}")
            axes[0, 0].set_title("Cache Hit Rate (Error)")
        
        # 2. 传输数据量对比
        try:
            bars = sns.barplot(x="Allocator", y="Total Transfer Data", data=df, ax=axes[0, 1])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    allocator = df["Allocator"].iloc[i]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[0, 1].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
            
            # 添加数值标签
            for i, p in enumerate(bars.patches):
                axes[0, 1].annotate(f'{p.get_height():.1f} MB', 
                                  (p.get_x() + p.get_width()/2, p.get_height()), 
                                  ha='center', va='bottom', fontsize=10)
            
            axes[0, 1].set_title("Total Model Transfer Data")
            axes[0, 1].set_ylabel("Data Volume (MB)")
            axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制传输数据量图表时出错: {e}")
            axes[0, 1].set_title("Total Model Transfer Data (Error)")
        
        # 3. 带宽利用率对比
        try:
            bars = sns.barplot(x="Allocator", y="Bandwidth Utilization", data=df, ax=axes[1, 0])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    allocator = df["Allocator"].iloc[i]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[1, 0].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
            
            # 添加数值标签
            for i, p in enumerate(bars.patches):
                axes[1, 0].annotate(f'{p.get_height():.1f}%', 
                                  (p.get_x() + p.get_width()/2, p.get_height()), 
                                  ha='center', va='bottom', fontsize=10)
            
            axes[1, 0].set_title("Bandwidth Utilization")
            axes[1, 0].set_ylabel("Utilization (%)")
            axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制带宽利用率图表时出错: {e}")
            axes[1, 0].set_title("Bandwidth Utilization (Error)")
        
        # 4. 传输开销对比
        try:
            bars = sns.barplot(x="Allocator", y="Transmission Overhead", data=df, ax=axes[1, 1])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    allocator = df["Allocator"].iloc[i]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[1, 1].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
            
            # 添加数值标签
            for i, p in enumerate(bars.patches):
                axes[1, 1].annotate(f'{p.get_height():.1f}%', 
                                  (p.get_x() + p.get_width()/2, p.get_height()), 
                                  ha='center', va='bottom', fontsize=10)
            
            axes[1, 1].set_title("Model Transmission Overhead")
            axes[1, 1].set_ylabel("Overhead (%)")
            axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制传输开销图表时出错: {e}")
            axes[1, 1].set_title("Model Transmission Overhead (Error)")
        
        plt.suptitle(f"Model Offloading Metrics Dashboard (BW={bandwidth} MB/s, BS={batch_size})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"model_offloading_dashboard_{key}_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建所有带宽的综合对比图
        plt.figure(figsize=(20, 15))
        
        # 准备数据
        all_data = []
        for bw in bandwidths:
            for bs in batch_sizes:
                key = f"bw{bw}_bs{bs}"
                
                for algo in algorithms:
                    # 收集关键指标
                    cache_hit_rate = results[key][algo].get("cache_hit_rate", 0) * 100
                    avg_transfer_time = results[key][algo].get("avg_transfer_time", 0)
                    total_transfer_data = results[key][algo].get("total_transfer_data", 0)
                    bandwidth_utilization = results[key][algo].get("bandwidth_utilization", 0) * 100
                    
                    # 计算传输开销
                    avg_inference_latency = results[key][algo].get("avg_inference_latency", 0)
                    avg_transfer_latency = results[key][algo].get("avg_transfer_latency", 0)
                    total_latency = avg_inference_latency + avg_transfer_latency
                    transmission_overhead = (avg_transfer_latency / total_latency * 100) if total_latency > 0 else 0
                    
                    all_data.append({
                        "Allocator": self.algorithm_names[algo],
                        "Bandwidth": bw,
                        "Batch Size": bs,
                        "Cache Hit Rate": cache_hit_rate,
                        "Avg Transfer Time": avg_transfer_time,
                        "Total Transfer Data": total_transfer_data,
                        "Bandwidth Utilization": bandwidth_utilization,
                        "Transmission Overhead": transmission_overhead
                    })
        
        all_df = pd.DataFrame(all_data)
        
        # 创建子图
        axes = []
        axes.append(plt.subplot(gs[0, 0]))
        axes.append(plt.subplot(gs[0, 1]))
        axes.append(plt.subplot(gs[1, 0]))
        axes.append(plt.subplot(gs[1, 1]))
        
        # 1. 缓存命中率对比
        try:
            if "Batch Size" in all_df.columns and all_df["Batch Size"].notna().any():
                bars = sns.barplot(x="Allocator", y="Cache Hit Rate", hue="Bandwidth", data=all_df, ax=axes[0, 0])
            else:
                bars = sns.barplot(x="Allocator", y="Cache Hit Rate", hue="Bandwidth", data=all_df, ax=axes[0, 0])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    # 获取对应的分配器名称
                    allocator_idx = i % len(all_df["Allocator"].unique())
                    allocator = all_df["Allocator"].unique()[allocator_idx]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[0, 0].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
                
            axes[0, 0].set_title("Cache Hit Rate")
            axes[0, 0].set_ylabel("Hit Rate (%)")
            axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制缓存命中率图表时出错: {e}")
            axes[0, 0].set_title("Cache Hit Rate (Error)")
        
        # 2. 传输数据量对比
        try:
            if "Batch Size" in all_df.columns and all_df["Batch Size"].notna().any():
                bars = sns.barplot(x="Allocator", y="Total Transfer Data", hue="Bandwidth", data=all_df, ax=axes[0, 1])
            else:
                bars = sns.barplot(x="Allocator", y="Total Transfer Data", hue="Bandwidth", data=all_df, ax=axes[0, 1])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    # 获取对应的分配器名称
                    allocator_idx = i % len(all_df["Allocator"].unique())
                    allocator = all_df["Allocator"].unique()[allocator_idx]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[0, 1].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
                
            axes[0, 1].set_title("Total Model Transfer Data")
            axes[0, 1].set_ylabel("Data Volume (MB)")
            axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制传输数据量图表时出错: {e}")
            axes[0, 1].set_title("Total Model Transfer Data (Error)")
        
        # 3. 带宽利用率对比
        try:
            if "Batch Size" in all_df.columns and all_df["Batch Size"].notna().any():
                bars = sns.barplot(x="Allocator", y="Bandwidth Utilization", hue="Bandwidth", data=all_df, ax=axes[1, 0])
            else:
                bars = sns.barplot(x="Allocator", y="Bandwidth Utilization", hue="Bandwidth", data=all_df, ax=axes[1, 0])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    # 获取对应的分配器名称
                    allocator_idx = i % len(all_df["Allocator"].unique())
                    allocator = all_df["Allocator"].unique()[allocator_idx]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[1, 0].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
                
            axes[1, 0].set_title("Bandwidth Utilization")
            axes[1, 0].set_ylabel("Utilization (%)")
            axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制带宽利用率图表时出错: {e}")
            axes[1, 0].set_title("Bandwidth Utilization (Error)")
        
        # 4. 传输开销对比
        try:
            if "Batch Size" in all_df.columns and all_df["Batch Size"].notna().any():
                bars = sns.barplot(x="Allocator", y="Transmission Overhead", hue="Bandwidth", data=all_df, ax=axes[1, 1])
            else:
                bars = sns.barplot(x="Allocator", y="Transmission Overhead", hue="Bandwidth", data=all_df, ax=axes[1, 1])
            
            # 为条形图添加标记
            if hasattr(bars, 'patches'):
                for i, p in enumerate(bars.patches):
                    # 获取对应的分配器名称
                    allocator_idx = i % len(all_df["Allocator"].unique())
                    allocator = all_df["Allocator"].unique()[allocator_idx]
                    marker = ALLOCATOR_MARKERS.get(allocator, 'o')
                    axes[1, 1].plot(p.get_x() + p.get_width()/2, p.get_height(), marker=marker, 
                                   color='black', markersize=8)
                
            axes[1, 1].set_title("Model Transmission Overhead")
            axes[1, 1].set_ylabel("Overhead (%)")
            axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"绘制传输开销图表时出错: {e}")
            axes[1, 1].set_title("Model Transmission Overhead (Error)")
        
        plt.suptitle("Model Metrics Comparison (All Bandwidths)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        filename = f"model_metrics_comparison_all_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()