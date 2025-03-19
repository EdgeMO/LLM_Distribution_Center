from abc import ABC, abstractmethod
import time
import numpy as np

class BaseAllocator(ABC):
    def __init__(self, edge_nodes, models, network_env):
        """
        初始化基础任务分配器
        
        参数:
        - edge_nodes: 边缘节点列表
        - models: 可用模型列表
        - network_env: 网络环境
        """
        self.edge_nodes = edge_nodes
        self.models = models
        self.network_env = network_env
        
        # 性能跟踪
        self.performance_history = []
        self.start_time = time.time()
        self.converged = False
        self.convergence_time = None
        
        # 模型部署缓存 - 确保为每个节点创建一个空集合
        self.model_deployment_cache = {}
        for node in edge_nodes:
            self.model_deployment_cache[node.node_id] = set()
            
            # 添加初始预装的模型到缓存
            for model_id in node.deployed_models:
                self.model_deployment_cache[node.node_id].add(model_id)
        
        # 模型卸载相关指标
        self.reset_offloading_metrics()
    
    def reset_offloading_metrics(self):
        """重置模型卸载相关指标"""
        self.transfer_times = []
        self.deployments_count = 0
        self.cache_hits = 0
        self.total_tasks = 0
        self.total_transfer_data = 0
        self.model_selection_counts = {}
        self.model_offloading_benefits = []
        self.inference_latencies = []
        self.transfer_latencies = []
        self.deployment_history = []
    
    @abstractmethod
    def allocate_tasks(self, task_batch, current_bandwidth):
        """分配任务到边缘节点并选择模型"""
        pass
    
    @abstractmethod
    def update_feedback(self, executed_allocations):
        """更新来自边缘节点的反馈"""
        pass
    
    def _deploy_model(self, model_id, node_id):
        """将模型部署到节点上"""
        # 获取模型和节点对象
        model = next((m for m in self.models if m.model_id == model_id), None)
        node = next((n for n in self.edge_nodes if n.node_id == node_id), None)
        
        if model is None or node is None:
            return False
        
        # 检查模型是否已部署
        if model_id in self.model_deployment_cache.get(node_id, set()):
            # 记录缓存命中
            self.cache_hits += 1
            return True
        
        # 尝试部署模型
        if node.can_deploy_model(model):
            success = node.deploy_model(model)
            if success:
                # 更新缓存
                self.model_deployment_cache[node_id].add(model_id)
                
                # 记录部署次数和传输数据
                self.deployments_count += 1
                self.total_transfer_data += model.size_mb
                
                # 记录部署历史
                self.deployment_history.append(1)  # 1表示部署了新模型
                
                return True
        
        return False
    
    def _calculate_model_transfer_time(self, model_id, node_id):
        """计算模型传输时间"""
        # 如果模型已经部署在节点上，传输时间为0
        if model_id in self.model_deployment_cache.get(node_id, set()):
            return 0
        
        # 获取模型对象
        model = next((m for m in self.models if m.model_id == model_id), None)
        if model is None:
            return float('inf')  # 模型不存在
        
        # 计算传输时间
        transfer_time = self.network_env.calculate_transfer_time(model, node_id)
        
        return transfer_time
    
    def _get_default_accuracy(self, task_features=None):
        """获取默认准确率（用于计算模型卸载收益）"""
        # 简单实现：使用所有模型的平均准确率
        return np.mean([model.accuracy for model in self.models]) if self.models else 0.5
    
    def get_convergence_time(self):
        """返回算法收敛所需时间"""
        return self.convergence_time if self.converged else None
    
    def get_performance_metrics(self):
        """返回性能指标"""
        if not self.performance_history:
            return None
            
        # 计算模型卸载相关指标
        offloading_metrics = self.get_offloading_metrics()
        
        return {
            "final_performance": self.performance_history[-1],
            "avg_performance": np.mean(self.performance_history),
            "convergence_time": self.get_convergence_time(),
            **offloading_metrics  # 添加模型卸载相关指标
        }
    
    def get_offloading_metrics(self):
        """获取模型卸载相关指标"""
        metrics = {}
        
        # 平均传输时间
        if self.transfer_times:
            metrics["avg_transfer_time"] = np.mean(self.transfer_times)
        else:
            metrics["avg_transfer_time"] = 0
        
        # 缓存命中率
        if self.total_tasks > 0:
            metrics["cache_hit_rate"] = self.cache_hits / self.total_tasks
        else:
            metrics["cache_hit_rate"] = 0
        
        # 总传输数据量
        metrics["total_transfer_data"] = self.total_transfer_data
        
        # 模型选择分布
        if self.total_tasks > 0:
            distribution = {model_id: count / self.total_tasks 
                          for model_id, count in self.model_selection_counts.items()}
            metrics["model_selection_distribution"] = distribution
        else:
            metrics["model_selection_distribution"] = {}
        
        # 模型卸载收益
        metrics["model_offloading_benefits"] = self.model_offloading_benefits
        
        # 延迟构成
        if self.inference_latencies:
            metrics["avg_inference_latency"] = np.mean(self.inference_latencies)
        else:
            metrics["avg_inference_latency"] = 0
            
        if self.transfer_latencies:
            metrics["avg_transfer_latency"] = np.mean(self.transfer_latencies)
        else:
            metrics["avg_transfer_latency"] = 0
        
        # 部署历史
        metrics["deployments"] = self.deployment_history
        
        return metrics