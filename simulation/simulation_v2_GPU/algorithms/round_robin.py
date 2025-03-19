import numpy as np
import time
from algorithms.base_allocator import BaseAllocator

class RoundRobinAllocator(BaseAllocator):
    def __init__(self, edge_nodes, models, network_env):
        """
        初始化轮询任务分配器
        
        参数:
        - edge_nodes: 边缘节点列表
        - models: 可用模型列表
        - network_env: 网络环境
        """
        super().__init__(edge_nodes, models, network_env)
        self.current_node = 0
        self.current_model = 0
    
    def _deploy_model(self, model_id, node_id):
        """将模型部署到节点上 (如果重写了基类方法)"""
        # 获取模型和节点对象
        model = next((m for m in self.models if m.model_id == model_id), None)
        node = next((n for n in self.edge_nodes if n.node_id == node_id), None)
        
        if model is None or node is None:
            return False
        
        # 检查模型是否已部署
        if model_id in self.model_deployment_cache.get(node_id, set()):
            return True
        
        # 尝试部署模型
        if node.can_deploy_model(model):
            success = node.deploy_model(model)
            if success:
                # 更新缓存
                self.model_deployment_cache[node_id].add(model_id)
                return True
        
        # 如果无法直接部署，尝试移除其他模型
        if len(node.deployed_models) > 0:
            # 简单策略：移除第一个找到的模型
            for deployed_model_id in list(node.deployed_models.keys()):
                # 确保模型存在于缓存中再移除
                if deployed_model_id in self.model_deployment_cache.get(node_id, set()):
                    node.remove_model(deployed_model_id)
                    self.model_deployment_cache[node_id].remove(deployed_model_id)
                    
                    # 尝试部署新模型
                    if node.can_deploy_model(model):
                        success = node.deploy_model(model)
                        if success:
                            self.model_deployment_cache[node_id].add(model_id)
                            return True
        
        return False
    
    def allocate_tasks(self, task_batch, current_bandwidth):
        """分配任务到边缘节点并选择模型"""
        # 更新网络环境带宽
        for node in self.edge_nodes:
            self.network_env.set_bandwidth(node.node_id, current_bandwidth)
        
        allocations = []
        
        for task_features in task_batch:
            # 轮询选择节点
            node = self.edge_nodes[self.current_node]
            self.current_node = (self.current_node + 1) % len(self.edge_nodes)
            
            # 轮询选择模型
            model = self.models[self.current_model]
            self.current_model = (self.current_model + 1) % len(self.models)
            
            # 计算传输时间
            transfer_time = self._calculate_model_transfer_time(model.model_id, node.node_id)
            
            # 尝试部署模型
            deploy_success = self._deploy_model(model.model_id, node.node_id)
            
            if deploy_success:
                allocations.append({
                    "task_features": task_features,
                    "node_id": node.node_id,
                    "model_id": model.model_id,
                    "transfer_time": transfer_time
                })
            else:
                # 如果部署失败，选择已部署的模型
                deployed_models = list(self.model_deployment_cache.get(node.node_id, set()))
                if deployed_models:
                    model_id = deployed_models[0]  # 简单选择第一个
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node.node_id,
                        "model_id": model_id,
                        "transfer_time": 0  # 已部署，无需传输
                    })
                else:
                    # 如果节点没有模型，选择最小的模型
                    smallest_model = min(self.models, key=lambda m: m.size_mb)
                    transfer_time = self._calculate_model_transfer_time(smallest_model.model_id, node.node_id)
                    self._deploy_model(smallest_model.model_id, node.node_id)
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node.node_id,
                        "model_id": smallest_model.model_id,
                        "transfer_time": transfer_time
                    })
        
        return allocations
    
    def update_feedback(self, executed_allocations):
        """更新来自边缘节点的反馈"""
        total_accuracy = 0
        total_latency = 0
        total_resource = 0
        count = 0
        
        for alloc in executed_allocations:
            transfer_time = alloc["transfer_time"]
            accuracy = alloc["result"]["accuracy"]
            latency = alloc["result"]["latency"]
            resource = alloc["result"]["resource_usage"]
            
            # 考虑传输时间
            total_latency_with_transfer = latency + transfer_time * 1000  # 转换为ms
            
            # 累计性能指标
            total_accuracy += accuracy
            total_latency += total_latency_with_transfer
            total_resource += resource
            count += 1
        
        if count > 0:
            avg_accuracy = total_accuracy / count
            avg_latency = total_latency / count
            avg_resource = total_resource / count
            
            # 计算综合性能得分
            performance = avg_accuracy - 0.01 * avg_latency + 0.5 * avg_resource
            self.performance_history.append(performance)
            
            # 检查收敛性
            if len(self.performance_history) > 20 and not self.converged:
                recent_perf = self.performance_history[-10:]
                if np.std(recent_perf) < 0.01:  # 如果性能稳定，认为已收敛
                    self.converged = True
                    self.convergence_time = time.time() - self.start_time