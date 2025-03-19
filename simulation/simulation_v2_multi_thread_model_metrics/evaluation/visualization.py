import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns

class ResultVisualizer:
    def __init__(self, output_dir="results"):
        """
        初始化可视化工具
        
        参数:
        - output_dir: 输出图像的目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 为不同算法定义标记和颜色
        self.algorithm_markers = {
            "active_reasoning": "o",  # 圆形
            "round_robin": "s",       # 方形
            "dqn": "^"                # 三角形
        }
        
        self.algorithm_colors = {
            "active_reasoning": "blue",
            "round_robin": "red",
            "dqn": "green"
        }
        
        self.algorithm_names = {
            "active_reasoning": "Active Reasoning",
            "round_robin": "Round Robin",
            "dqn": "DQN"
        }
    
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
                    
                    for metric in metrics:
                        plt.figure(figsize=(10, 6))
                        
                        # 提取数据
                        data = []
                        for algo in algorithms:
                            value = results[key][algo][metric]
                            # 如果是收敛时间且值为None，设为最大迭代次数
                            if metric == "convergence_time" and value is None:
                                value = 1000
                            data.append(value)
                        
                        # 绘制条形图
                        bars = plt.bar(
                            [self.algorithm_names[algo] for algo in algorithms],
                            data,
                            color=[self.algorithm_colors[algo] for algo in algorithms]
                        )
                        
                        # 添加数值标签
                        for bar, value in zip(bars, data):
                            height = bar.get_height()
                            plt.text(
                                bar.get_x() + bar.get_width() / 2.,
                                height * 1.01,
                                f'{value:.2f}',
                                ha='center',
                                va='bottom',
                                fontsize=10
                            )
                        
                        plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s, BS={batch_size})")
                        plt.ylabel(metric_titles[metric])
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # 保存图表
                        filename = f"{metric}_{key}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath, dpi=300)
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
                            y = [np.mean(history[i:i+10]) for i in x]
                            
                            # 使用不同的标记和颜色
                            plt.plot(
                                x, y,
                                marker=self.algorithm_markers[algo],
                                color=self.algorithm_colors[algo],
                                label=self.algorithm_names[algo],
                                markevery=max(1, len(x)//10),  # 每10个点放一个标记
                                markersize=8
                            )
                        
                        plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s, BS={batch_size})")
                        plt.xlabel("Iteration")
                        plt.ylabel(metric_titles[metric])
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend()
                        plt.tight_layout()
                        
                        # 保存图表
                        filename = f"{metric}_convergence_{key}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath, dpi=300)
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
        
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            
            for algo in algorithms:
                values = []
                for bw in bandwidths:
                    key = f"bw{bw}_bs{batch_size}"
                    values.append(results[key][algo][metric])
                
                plt.plot(
                    bandwidths, values,
                    marker=self.algorithm_markers[algo],
                    color=self.algorithm_colors[algo],
                    label=self.algorithm_names[algo],
                    linewidth=2,
                    markersize=10
                )
            
            plt.title(f"{metric_titles[metric]} (BS={batch_size})")
            plt.xlabel("Bandwidth (MB/s)")
            plt.ylabel(metric_titles[metric])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = f"{metric}_vs_bandwidth_bs{batch_size}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300)
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
        
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            
            for algo in algorithms:
                values = []
                for bs in batch_sizes:
                    key = f"bw{bandwidth}_bs{bs}"
                    values.append(results[key][algo][metric])
                
                plt.plot(
                    batch_sizes, values,
                    marker=self.algorithm_markers[algo],
                    color=self.algorithm_colors[algo],
                    label=self.algorithm_names[algo],
                    linewidth=2,
                    markersize=10
                )
            
            plt.title(f"{metric_titles[metric]} (BW={bandwidth} MB/s)")
            plt.xlabel("Batch Size")
            plt.ylabel(metric_titles[metric])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = f"{metric}_vs_batchsize_bw{bandwidth}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
    
    # ================ 模型卸载相关可视化 ================
    
    def visualize_model_transfer_time(self, results, bandwidths, batch_sizes):
        """可视化模型传输时间"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating model transfer time charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    plt.figure(figsize=(10, 6))
                    
                    # 提取数据
                    transfer_times = []
                    for algo in algorithms:
                        # 计算平均传输时间（秒）
                        avg_transfer = results[key][algo].get("avg_transfer_time", 0)
                        transfer_times.append(avg_transfer)
                    
                    # 绘制条形图
                    bars = plt.bar(
                        [self.algorithm_names[algo] for algo in algorithms],
                        transfer_times,
                        color=[self.algorithm_colors[algo] for algo in algorithms]
                    )
                    
                    # 添加数值标签
                    for bar, value in zip(bars, transfer_times):
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height * 1.01,
                            f'{value:.2f}s',
                            ha='center',
                            va='bottom',
                            fontsize=10
                        )
                    
                    plt.title(f"Average Model Transfer Time (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.ylabel("Transfer Time (seconds)")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"model_transfer_time_{key}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    
                    pbar.update(1)
    
    def visualize_model_deployment_frequency(self, results, bandwidths, batch_sizes):
        """可视化模型部署频率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating model deployment frequency charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    plt.figure(figsize=(12, 7))
                    
                    for algo in algorithms:
                        # 获取模型部署历史
                        deploy_history = results[key][algo].get("deployments", [])
                        
                        if deploy_history:
                            # 对历史数据进行滑动窗口平均，每10个点取一个平均值
                            window_size = 10
                            x = list(range(0, len(deploy_history), window_size))
                            y = [np.mean(deploy_history[i:i+window_size]) for i in x]
                            
                            plt.plot(
                                x, y,
                                marker=self.algorithm_markers[algo],
                                color=self.algorithm_colors[algo],
                                label=self.algorithm_names[algo],
                                markevery=max(1, len(x)//10),
                                markersize=8
                            )
                    
                    plt.title(f"Model Deployment Frequency (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.xlabel("Iteration")
                    plt.ylabel("Deployments per Iteration")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"model_deployment_frequency_{key}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    
                    pbar.update(1)
    
    def visualize_cache_hit_rate(self, results, bandwidths, batch_sizes):
        """可视化模型缓存命中率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating cache hit rate charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    plt.figure(figsize=(10, 6))
                    
                    # 提取数据
                    hit_rates = []
                    for algo in algorithms:
                        # 获取缓存命中率
                        hit_rate = results[key][algo].get("cache_hit_rate", 0) * 100  # 转换为百分比
                        hit_rates.append(hit_rate)
                    
                    # 绘制条形图
                    bars = plt.bar(
                        [self.algorithm_names[algo] for algo in algorithms],
                        hit_rates,
                        color=[self.algorithm_colors[algo] for algo in algorithms]
                    )
                    
                    # 添加数值标签
                    for bar, value in zip(bars, hit_rates):
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height * 1.01,
                            f'{value:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=10
                        )
                    
                    plt.title(f"Model Cache Hit Rate (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.ylabel("Cache Hit Rate (%)")
                    plt.ylim(0, 100)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"model_cache_hit_rate_{key}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    
                    pbar.update(1)
    
    def visualize_bandwidth_utilization(self, results, bandwidths, batch_sizes):
        """可视化带宽利用效率"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(batch_sizes), desc="Generating bandwidth utilization charts") as pbar:
            for batch_size in batch_sizes:
                plt.figure(figsize=(12, 7))
                
                for algo in algorithms:
                    utilization_rates = []
                    for bw in bandwidths:
                        key = f"bw{bw}_bs{batch_size}"
                        # 计算带宽利用率
                        utilization = results[key][algo].get("bandwidth_utilization", 0) * 100  # 转换为百分比
                        utilization_rates.append(utilization)
                    
                    plt.plot(
                        bandwidths, utilization_rates,
                        marker=self.algorithm_markers[algo],
                        color=self.algorithm_colors[algo],
                        label=self.algorithm_names[algo],
                        linewidth=2,
                        markersize=10
                    )
                
                plt.title(f"Bandwidth Utilization Efficiency (BS={batch_size})")
                plt.xlabel("Bandwidth (MB/s)")
                plt.ylabel("Bandwidth Utilization (%)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                
                # 保存图表
                filename = f"bandwidth_utilization_bs{batch_size}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300)
                plt.close()
                
                pbar.update(1)
    
    def visualize_total_transfer_data(self, results, bandwidths, batch_sizes):
        """可视化总传输数据量"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating total transfer data charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    plt.figure(figsize=(10, 6))
                    
                    # 提取数据
                    transfer_data = []
                    for algo in algorithms:
                        # 获取总传输数据量 (MB)
                        data = results[key][algo].get("total_transfer_data", 0)
                        transfer_data.append(data)
                    
                    # 绘制条形图
                    bars = plt.bar(
                        [self.algorithm_names[algo] for algo in algorithms],
                        transfer_data,
                        color=[self.algorithm_colors[algo] for algo in algorithms]
                    )
                    
                    # 添加数值标签
                    for bar, value in zip(bars, transfer_data):
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height * 1.01,
                            f'{value:.1f} MB',
                            ha='center',
                            va='bottom',
                            fontsize=10
                        )
                    
                    plt.title(f"Total Model Transfer Data (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.ylabel("Data Volume (MB)")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"total_transfer_data_{key}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    
                    pbar.update(1)
    
    def visualize_latency_composition(self, results, bandwidths, batch_sizes):
        """可视化延迟构成"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes), desc="Generating latency composition charts") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    plt.figure(figsize=(12, 7))
                    
                    # 提取数据
                    inference_latencies = []
                    transfer_latencies = []
                    
                    for algo in algorithms:
                        # 获取推理延迟和传输延迟
                        inference = results[key][algo].get("avg_inference_latency", 0)
                        transfer = results[key][algo].get("avg_transfer_latency", 0)
                        
                        inference_latencies.append(inference)
                        transfer_latencies.append(transfer)
                    
                    x = np.arange(len(algorithms))
                    width = 0.6
                    
                    # 创建堆叠条形图
                    plt.bar(x, inference_latencies, width, label='Inference Latency', 
                           color='skyblue', alpha=0.7)
                    plt.bar(x, transfer_latencies, width, bottom=inference_latencies, 
                           label='Transfer Latency', color='coral', alpha=0.7)
                    
                    # 添加总延迟标签
                    for i in range(len(x)):
                        total = inference_latencies[i] + transfer_latencies[i]
                        plt.text(x[i], total * 1.01, f'{total:.1f}ms', 
                                ha='center', va='bottom', fontsize=10)
                    
                    plt.title(f"Latency Composition (BW={bandwidth} MB/s, BS={batch_size})")
                    plt.ylabel("Latency (ms)")
                    plt.xticks(x, [self.algorithm_names[algo] for algo in algorithms])
                    plt.legend()
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存图表
                    filename = f"latency_composition_{key}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                    
                    pbar.update(1)
    
    def visualize_offloading_decision_heatmap(self, results, bandwidths, batch_sizes):
        """可视化模型卸载决策热力图"""
        algorithms = ["active_reasoning", "round_robin", "dqn"]
        
        with tqdm(total=len(bandwidths)*len(batch_sizes)*len(algorithms), desc="Generating offloading decision heatmaps") as pbar:
            for bandwidth in bandwidths:
                for batch_size in batch_sizes:
                    key = f"bw{bandwidth}_bs{batch_size}"
                    
                    for algo in algorithms:
                        # 获取卸载决策数据
                        decision_map = results[key][algo].get("offloading_decision_map", None)
                        
                        if decision_map is not None and isinstance(decision_map, np.ndarray) and len(decision_map.shape) == 2:
                            plt.figure(figsize=(10, 8))
                            
                            # 绘制热力图
                            sns.heatmap(decision_map, cmap="YlGnBu", annot=False)
                            
                            plt.title(f"{self.algorithm_names[algo]}: Offloading Decision Heatmap\n(BW={bandwidth} MB/s, BS={batch_size})")
                            plt.xlabel("Task Feature 2")
                            plt.ylabel("Task Feature 1")
                            plt.tight_layout()
                            
                            # 保存图表
                            filename = f"offloading_decision_heatmap_{algo}_{key}.png"
                            filepath = os.path.join(self.output_dir, filename)
                            plt.savefig(filepath, dpi=300)
                            plt.close()
                        
                        pbar.update(1)
    
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
        
        print("Model offloading visualization completed!")