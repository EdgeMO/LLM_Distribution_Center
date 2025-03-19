import numpy as np
import time
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
from joblib import Parallel, delayed
from models.model import Model
from models.edge_node import EdgeNode
from models.network_env import NetworkEnvironment
from algorithms.active_reasoning import ActiveReasoningTaskAllocator
from algorithms.round_robin import RoundRobinAllocator
from algorithms.dqn import DQNTaskAllocator

class TaskAllocationEvaluator:
    def __init__(self, num_edge_nodes=5, num_models=10, feature_dim=10, num_iterations=1000):
        """
        初始化评估器
        
        参数:
        - num_edge_nodes: 边缘节点数量
        - num_models: 模型数量
        - feature_dim: 任务特征维度
        - num_iterations: 评估迭代次数
        """
        self.num_edge_nodes = num_edge_nodes
        self.num_models = num_models
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        
        # 检查 GPU 可用性
        self.using_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        if self.using_gpu:
            print("评估器将使用 GPU 加速")
        
        # 创建模型库
        self.models = self._create_models()
        
        # 创建边缘节点
        self.edge_nodes = self._create_edge_nodes()
        
        # 创建网络环境
        self.network_env = NetworkEnvironment()
        
        # 创建三种算法实例
        self.active_reasoning = ActiveReasoningTaskAllocator(self.edge_nodes, self.models, self.network_env, feature_dim)
        self.round_robin = RoundRobinAllocator(self.edge_nodes, self.models, self.network_env)
        self.dqn = DQNTaskAllocator(self.edge_nodes, self.models, self.network_env, feature_dim)
        
        # 存储评估结果
        self.results = {
            "active_reasoning": {"accuracies": [], "latencies": [], "resources": []},
            "round_robin": {"accuracies": [], "latencies": [], "resources": []},
            "dqn": {"accuracies": [], "latencies": [], "resources": []}
        }
    
    def _create_models(self):
        """创建模拟模型库"""
        print("Creating model library...")
        models = []
        np.random.seed(42)  # 固定随机种子以便复现
        
        for i in range(self.num_models):
            # 创建不同特性的模型
            if i < self.num_models // 3:
                # 高准确率但大尺寸和高资源使用的模型
                accuracy = 0.8 + 0.2 * np.random.random()
                inference_time = 50 + 50 * np.random.random()
                resource_usage = 0.6 + 0.4 * np.random.random()
                size_mb = 100 + 100 * np.random.random()
            elif i < 2 * self.num_models // 3:
                # 中等特性的模型
                accuracy = 0.6 + 0.3 * np.random.random()
                inference_time = 20 + 30 * np.random.random()
                resource_usage = 0.3 + 0.4 * np.random.random()
                size_mb = 50 + 50 * np.random.random()
            else:
                # 低准确率但小尺寸和低资源使用的模型
                accuracy = 0.4 + 0.2 * np.random.random()
                inference_time = 5 + 15 * np.random.random()
                resource_usage = 0.1 + 0.2 * np.random.random()
                size_mb = 10 + 40 * np.random.random()
            
            model = Model(f"model_{i}", accuracy, inference_time, resource_usage, size_mb)
            models.append(model)
        
        print(f"Created {len(models)} models")
        return models
    
    def _create_edge_nodes(self):
        """创建边缘节点"""
        print("Creating edge nodes...")
        nodes = []
        np.random.seed(42)  # 固定随机种子以便复现
        
        for i in range(self.num_edge_nodes):
            # 创建不同特性的边缘节点
            compute_power = 0.5 + 0.5 * np.random.random()  # 计算能力 (0.5-1.0)
            memory_capacity = 100 + 200 * np.random.random()  # 内存容量 (100-300 MB)
            
            # 为节点预装一个小型模型
            initial_models = [self.models[-(i % 3 + 1)]]
            
            node = EdgeNode(f"node_{i}", compute_power, memory_capacity, initial_models)
            nodes.append(node)
        
        print(f"Created {len(nodes)} edge nodes")
        return nodes
    
    def generate_task_batch(self, batch_size):
        """生成随机任务批次"""
        return np.random.rand(batch_size, self.feature_dim)
    
    def execute_allocations(self, allocations):
        """执行分配的任务并返回结果"""
        executed_allocations = []
        
        for alloc in allocations:
            task_features = alloc["task_features"]
            node_id = alloc["node_id"]
            model_id = alloc["model_id"]
            transfer_time = alloc["transfer_time"]
            
            # 找到对应的节点和模型
            node = next((n for n in self.edge_nodes if n.node_id == node_id), None)
            model = next((m for m in self.models if m.model_id == model_id), None)
            
            if node is None or model is None:
                continue
            
            try:
                # 执行任务
                result = node.execute_task(task_features, model)
                
                # 记录执行结果
                executed_alloc = alloc.copy()
                executed_alloc["result"] = result
                executed_allocations.append(executed_alloc)
            except ValueError:
                # 如果模型未部署，跳过
                continue
        
        # 更新所有节点的负载
        for node in self.edge_nodes:
            node.update_load()
        
        return executed_allocations
    
    def _execute_allocations(self, allocations, edge_nodes):
        """执行分配的任务并返回结果 (使用指定的边缘节点)"""
        executed_allocations = []
        
        for alloc in allocations:
            task_features = alloc["task_features"]
            node_id = alloc["node_id"]
            model_id = alloc["model_id"]
            transfer_time = alloc["transfer_time"]
            
            # 找到对应的节点和模型
            node = next((n for n in edge_nodes if n.node_id == node_id), None)
            model = next((m for m in self.models if m.model_id == model_id), None)
            
            if node is None or model is None:
                continue
            
            try:
                # 执行任务
                result = node.execute_task(task_features, model)
                
                # 记录执行结果
                executed_alloc = alloc.copy()
                executed_alloc["result"] = result
                executed_allocations.append(executed_alloc)
            except ValueError:
                # 如果模型未部署，跳过
                continue
        
        # 更新所有节点的负载
        for node in edge_nodes:
            node.update_load()
        
        return executed_allocations
    
    def _reset_edge_nodes(self, nodes=None):
        """重置边缘节点状态"""
        nodes_to_reset = nodes if nodes is not None else self.edge_nodes
        for node in nodes_to_reset:
            node.current_load = 0.0
    
    def _update_offloading_map(self, decision_map, allocations, task_batch):
        """更新卸载决策热力图"""
        for i, alloc in enumerate(allocations):
            if i >= len(task_batch):
                continue
                
            # 提取任务特征的前两个维度
            if len(task_batch[i]) >= 2:
                x = min(int(task_batch[i][0] * 10), 9)  # 将特征映射到0-9的网格索引
                y = min(int(task_batch[i][1] * 10), 9)
                
                # 记录模型传输 (1表示传输了新模型，0表示使用已部署模型)
                decision_map[x][y] = 1 if alloc["transfer_time"] > 0 else 0
    
    def _calculate_bandwidth_utilization(self, total_transfer_data, bandwidth, num_iterations):
        """计算带宽利用率"""
        # 理论最大传输量 = 带宽 * 总时间
        # 假设每次迭代的时间为1秒
        theoretical_max = bandwidth * num_iterations
        
        # 实际传输数据量
        actual_transfer = total_transfer_data
        
        # 带宽利用率
        if theoretical_max > 0:
            return actual_transfer / theoretical_max
        else:
            return 0
    
    def _evaluate_single_config(self, bandwidth, batch_size):
        """评估单个带宽和批次大小组合"""
        print(f"\nEvaluating with bandwidth={bandwidth} MB/s, batch_size={batch_size}")
        
        # 创建新的算法实例和边缘节点，确保线程安全
        edge_nodes = self._create_edge_nodes()
        network_env = NetworkEnvironment()
        
        active_reasoning = ActiveReasoningTaskAllocator(edge_nodes, self.models, network_env, self.feature_dim)
        round_robin = RoundRobinAllocator(edge_nodes, self.models, network_env)
        dqn = DQNTaskAllocator(edge_nodes, self.models, network_env, self.feature_dim)
        
        # 重置结果
        results = {
            "active_reasoning": {"accuracies": [], "latencies": [], "resources": []},
            "round_robin": {"accuracies": [], "latencies": [], "resources": []},
            "dqn": {"accuracies": [], "latencies": [], "resources": []}
        }
        
        # 创建模型卸载决策热力图数据结构
        offloading_decision_maps = {
            "active_reasoning": np.zeros((10, 10)),  # 10x10网格代表任务特征空间
            "round_robin": np.zeros((10, 10)),
            "dqn": np.zeros((10, 10))
        }
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(total=self.num_iterations, 
                           desc=f"BW={bandwidth}, BS={batch_size}",
                           position=0, leave=True)
        
        for i in range(self.num_iterations):
            # 生成任务批次
            task_batch = self.generate_task_batch(batch_size)
            
            # 评估主动推理算法
            ar_allocations = active_reasoning.allocate_tasks(task_batch, bandwidth)
            ar_executed = self._execute_allocations(ar_allocations, edge_nodes)
            active_reasoning.update_feedback(ar_executed)
            
            # 计算指标
            if ar_executed:
                ar_accuracy = np.mean([alloc["result"]["accuracy"] for alloc in ar_executed])
                ar_latency = np.mean([alloc["result"]["latency"] + alloc["transfer_time"] * 1000 for alloc in ar_executed])
                ar_resource = np.mean([alloc["result"]["resource_usage"] for alloc in ar_executed])
                
                results["active_reasoning"]["accuracies"].append(ar_accuracy)
                results["active_reasoning"]["latencies"].append(ar_latency)
                results["active_reasoning"]["resources"].append(ar_resource)
                
                # 更新卸载决策热力图
                self._update_offloading_map(offloading_decision_maps["active_reasoning"], ar_allocations, task_batch)
            
            # 重置边缘节点状态
            self._reset_edge_nodes(edge_nodes)
            
            # 评估轮询算法
            rr_allocations = round_robin.allocate_tasks(task_batch, bandwidth)
            rr_executed = self._execute_allocations(rr_allocations, edge_nodes)
            round_robin.update_feedback(rr_executed)
            
            # 计算指标
            if rr_executed:
                rr_accuracy = np.mean([alloc["result"]["accuracy"] for alloc in rr_executed])
                rr_latency = np.mean([alloc["result"]["latency"] + alloc["transfer_time"] * 1000 for alloc in rr_executed])
                rr_resource = np.mean([alloc["result"]["resource_usage"] for alloc in rr_executed])
                
                results["round_robin"]["accuracies"].append(rr_accuracy)
                results["round_robin"]["latencies"].append(rr_latency)
                results["round_robin"]["resources"].append(rr_resource)
                
                # 更新卸载决策热力图
                self._update_offloading_map(offloading_decision_maps["round_robin"], rr_allocations, task_batch)
            
            # 重置边缘节点状态
            self._reset_edge_nodes(edge_nodes)
            
            # 评估DQN算法
            dqn_allocations = dqn.allocate_tasks(task_batch, bandwidth)
            dqn_executed = self._execute_allocations(dqn_allocations, edge_nodes)
            dqn.update_feedback(dqn_executed)
            
            # 计算指标
            if dqn_executed:
                dqn_accuracy = np.mean([alloc["result"]["accuracy"] for alloc in dqn_executed])
                dqn_latency = np.mean([alloc["result"]["latency"] + alloc["transfer_time"] * 1000 for alloc in dqn_executed])
                dqn_resource = np.mean([alloc["result"]["resource_usage"] for alloc in dqn_executed])
                
                results["dqn"]["accuracies"].append(dqn_accuracy)
                results["dqn"]["latencies"].append(dqn_latency)
                results["dqn"]["resources"].append(dqn_resource)
                
                # 更新卸载决策热力图
                self._update_offloading_map(offloading_decision_maps["dqn"], dqn_allocations, task_batch)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 每50次迭代显示当前性能
            if (i+1) % 50 == 0:
                progress_bar.set_postfix({
                    'AR acc': np.mean(results["active_reasoning"]["accuracies"][-50:]) if results["active_reasoning"]["accuracies"] else 0,
                    'RR acc': np.mean(results["round_robin"]["accuracies"][-50:]) if results["round_robin"]["accuracies"] else 0,
                    'DQN acc': np.mean(results["dqn"]["accuracies"][-50:]) if results["dqn"]["accuracies"] else 0
                })
        
        # 关闭进度条
        progress_bar.close()
        
        # 获取各算法的模型卸载指标
        ar_offloading_metrics = active_reasoning.get_offloading_metrics()
        rr_offloading_metrics = round_robin.get_offloading_metrics()
        dqn_offloading_metrics = dqn.get_offloading_metrics()
        
        # 添加卸载决策热力图
        ar_offloading_metrics["offloading_decision_map"] = offloading_decision_maps["active_reasoning"]
        rr_offloading_metrics["offloading_decision_map"] = offloading_decision_maps["round_robin"]
        dqn_offloading_metrics["offloading_decision_map"] = offloading_decision_maps["dqn"]
        
        # 计算带宽利用率
        ar_offloading_metrics["bandwidth_utilization"] = self._calculate_bandwidth_utilization(
            ar_offloading_metrics["total_transfer_data"], bandwidth, self.num_iterations)
        rr_offloading_metrics["bandwidth_utilization"] = self._calculate_bandwidth_utilization(
            rr_offloading_metrics["total_transfer_data"], bandwidth, self.num_iterations)
        dqn_offloading_metrics["bandwidth_utilization"] = self._calculate_bandwidth_utilization(
            dqn_offloading_metrics["total_transfer_data"], bandwidth, self.num_iterations)
        
        # 构建结果
        config_result = {
            "active_reasoning": {
                "avg_accuracy": np.mean(results["active_reasoning"]["accuracies"]),
                "avg_latency": np.mean(results["active_reasoning"]["latencies"]),
                "avg_resource": np.mean(results["active_reasoning"]["resources"]),
                "convergence_time": active_reasoning.get_convergence_time(),
                "history": {
                    "accuracies": results["active_reasoning"]["accuracies"],
                    "latencies": results["active_reasoning"]["latencies"],
                    "resources": results["active_reasoning"]["resources"]
                },
                **ar_offloading_metrics  # 添加模型卸载指标
            },
            "round_robin": {
                "avg_accuracy": np.mean(results["round_robin"]["accuracies"]),
                "avg_latency": np.mean(results["round_robin"]["latencies"]),
                "avg_resource": np.mean(results["round_robin"]["resources"]),
                "convergence_time": round_robin.get_convergence_time(),
                "history": {
                    "accuracies": results["round_robin"]["accuracies"],
                    "latencies": results["round_robin"]["latencies"],
                    "resources": results["round_robin"]["resources"]
                },
                **rr_offloading_metrics  # 添加模型卸载指标
            },
            "dqn": {
                "avg_accuracy": np.mean(results["dqn"]["accuracies"]),
                "avg_latency": np.mean(results["dqn"]["latencies"]),
                "avg_resource": np.mean(results["dqn"]["resources"]),
                "convergence_time": dqn.get_convergence_time(),
                "history": {
                    "accuracies": results["dqn"]["accuracies"],
                    "latencies": results["dqn"]["latencies"],
                    "resources": results["dqn"]["resources"]
                },
                **dqn_offloading_metrics  # 添加模型卸载指标
            }
        }
        
        print(f"Results for bandwidth={bandwidth} MB/s, batch_size={batch_size}:")
        for algo in ["active_reasoning", "round_robin", "dqn"]:
            print(f"  {algo}:")
            print(f"    Accuracy: {config_result[algo]['avg_accuracy']:.4f}")
            print(f"    Latency: {config_result[algo]['avg_latency']:.4f} ms")
            print(f"    Resource: {config_result[algo]['avg_resource']:.4f}")
            print(f"    Convergence time: {config_result[algo]['convergence_time']} s")
            print(f"    Transfer time: {config_result[algo]['avg_transfer_time']:.4f} s")
            print(f"    Cache hit rate: {config_result[algo]['cache_hit_rate']*100:.2f}%")
            print(f"    Total transfer data: {config_result[algo]['total_transfer_data']:.2f} MB")
        
        return (f"bw{bandwidth}_bs{batch_size}", config_result)
    
    def evaluate(self, bandwidths=[10, 50, 100], batch_sizes=[10, 50, 100]):
        """评估所有算法在不同带宽和批次大小下的表现"""
        # 计算要评估的配置数量
        configs = [(bw, bs) for bw in bandwidths for bs in batch_sizes]
        print(f"开始评估 {len(configs)} 种配置...")
        
        # 如果使用 GPU，顺序执行配置以最大化 GPU 使用率
        if self.using_gpu:
            print("使用 GPU 顺序评估")
            all_results = {}
            for bw, bs in configs:
                result = self._evaluate_single_config(bw, bs)
                all_results[result[0]] = result[1]
            return all_results
        else:
            # 使用 CPU 并行评估
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            print(f"使用 {num_cores} 个 CPU 核心并行处理")
            
            # 分批执行评估，每批最多4个配置
            max_parallel_jobs = min(num_cores, 4)  # 最多同时运行4个作业
            all_results = {}
            
            for i in range(0, len(configs), max_parallel_jobs):
                batch_configs = configs[i:i+max_parallel_jobs]
                print(f"正在评估配置批次 {i//max_parallel_jobs + 1}/{(len(configs) + max_parallel_jobs - 1)//max_parallel_jobs}")
                
                results = Parallel(n_jobs=len(batch_configs))(
                    delayed(self._evaluate_single_config)(bw, bs) for bw, bs in batch_configs
                )
                
                # 将结果合并到all_results
                for key, value in dict(results).items():
                    all_results[key] = value
            
            return all_results