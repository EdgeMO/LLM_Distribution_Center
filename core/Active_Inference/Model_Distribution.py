import numpy as np
from typing import List, Dict
import logging
import os
import sys
current_working_directory = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from common_tool.init import model_init
from config.type import ModelInfo

class ModelOffloadingSystem:
    def __init__(self, num_edge_nodes: int, model_list: List[ModelInfo], 
                 accuracy_threshold: float = 0.8, 
                 latency_threshold: float = 0.3,
                 throughput_threshold: float = 10.0):
        """
        模型卸载系统
        
        Args:
            num_edge_nodes: 边缘节点数量
            model_list: 可用模型列表
            accuracy_threshold: 准确率阈值
            latency_threshold: 延迟阈值
            throughput_threshold: 吞吐量阈值
        """
        self.num_edge_nodes = num_edge_nodes
        self.model_list = model_list
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold = latency_threshold
        self.throughput_threshold = throughput_threshold
        
        # 节点性能历史
        self.node_performance_history = [[] for _ in range(num_edge_nodes)]
        
        # 当前节点加载的模型
        self.current_node_models = [1] * num_edge_nodes  # 默认每个节点加载模型ID为1
        
        # 模型卸载相关
        self.epoch_count = 0
        self.performance_metrics = {
            'accuracy': [],
            'latency': [],
            'throughput': []
        }

    def update_performance_history(self, node_performance: List[Dict[str, float]]):
        """
        更新节点性能历史
        
        Args:
            node_performance: 各节点性能指标
        """
        # 更新节点性能历史
        for i, perf in enumerate(node_performance):
            self.node_performance_history[i].append(perf)
            if len(self.node_performance_history[i]) > 100:  # 保持最近100个时间片的历史
                self.node_performance_history[i].pop(0)
        
        # 更新全局性能指标
        self.performance_metrics['accuracy'].append(np.mean([node['accuracy'] for node in node_performance]))
        self.performance_metrics['latency'].append(np.mean([node['latency'] for node in node_performance]))
        self.performance_metrics['throughput'].append(np.mean([node['avg_throughput'] for node in node_performance]))
        
        self.epoch_count += 1

    def should_offload(self) -> bool:
        """
        判断是否需要进行模型卸载
        
        Returns:
            是否需要卸载
        """
        # 至少运行10个周期后才考虑卸载
        if self.epoch_count < 10:
            return False
        
        # 计算最近性能指标的平均值
        window_size = min(50, len(self.performance_metrics['accuracy']))
        recent_accuracy = np.mean(self.performance_metrics['accuracy'][-window_size:])
        recent_latency = np.mean(self.performance_metrics['latency'][-window_size:])
        recent_throughput = np.mean(self.performance_metrics['throughput'][-window_size:])
        
        # 判断是否需要卸载
        needs_offload = (
            recent_accuracy < self.accuracy_threshold or
            recent_latency > self.latency_threshold or
            recent_throughput < self.throughput_threshold
        )
        
        if needs_offload:
            logging.info(f"触发模型卸载: 准确率={recent_accuracy:.4f}, 延迟={recent_latency:.4f}, 吞吐量={recent_throughput:.4f}")
        
        return needs_offload

    def get_node_performance_profile(self, node_idx: int) -> Dict[str, float]:
        """
        获取节点性能概况
        
        Args:
            node_idx: 节点索引
            
        Returns:
            节点性能概况
        """
        if not self.node_performance_history[node_idx]:
            return {
                'avg_accuracy': 0.8,  # 默认值
                'avg_latency': 0.2,   # 默认值
                'avg_throughput': 10.0,  # 默认值
                'accuracy_variance': 0.0,
                'latency_variance': 0.0,
                'throughput_variance': 0.0,
                'load_stability': 1.0  # 1.0表示非常稳定
            }
        
        # 获取最近的性能记录
        recent_history = self.node_performance_history[node_idx][-10:]
        
        # 计算平均值
        accuracy_values = [h['accuracy'] for h in recent_history]
        latency_values = [h['latency'] for h in recent_history]
        throughput_values = [h['avg_throughput'] for h in recent_history]
        
        avg_accuracy = np.mean(accuracy_values)
        avg_latency = np.mean(latency_values)
        avg_throughput = np.mean(throughput_values)
        
        # 计算方差(稳定性指标)
        accuracy_variance = np.var(accuracy_values) if len(accuracy_values) > 1 else 0
        latency_variance = np.var(latency_values) if len(latency_values) > 1 else 0
        throughput_variance = np.var(throughput_values) if len(throughput_values) > 1 else 0
        
        # 计算负载稳定性(值越小表示波动越大)
        if len(throughput_values) > 1:
            load_stability = 1.0 / (1.0 + throughput_variance / (avg_throughput + 1e-5))
        else:
            load_stability = 1.0
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_latency': avg_latency,
            'avg_throughput': avg_throughput,
            'accuracy_variance': accuracy_variance,
            'latency_variance': latency_variance,
            'throughput_variance': throughput_variance,
            'load_stability': load_stability
        }

    def select_best_model_for_node(self, node_idx: int) -> int:
        """
        为节点选择最佳模型
        
        Args:
            node_idx: 节点索引
            
        Returns:
            最佳模型ID
        """
        node_profile = self.get_node_performance_profile(node_idx)
        
        best_model = None
        best_score = float('-inf')
        
        for model in self.model_list:
            # 计算模型得分
            score = self._calculate_model_score(model, node_profile)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        if best_model:
            return best_model.id
        else:
            # 如果没有找到合适的模型，保持当前模型
            return self.current_node_models[node_idx]

    def _calculate_model_score(self, model: ModelInfo, node_profile: Dict[str, float]) -> float:
        """
        计算模型在特定节点上的适应性得分
        
        Args:
            model: 模型信息
            node_profile: 节点性能概况
            
        Returns:
            模型得分
        """
        # 准确率因子: 困惑度越低，准确率越高
        accuracy_factor = 1.0 / (model.perplexity + 1e-5)
        
        # 延迟因子: token处理时间越短越好
        latency_factor = 1.0 / (model.token_processing_time + 1e-5)
        
        # 加载时间因子: 加载时间越短越好
        load_time_factor = 1.0 / (model.load_time + 1e-5)
        
        # 资源因子: 参数量和磁盘空间越小越好
        resource_factor = 1.0 / (0.5 * model.parameter_count / 100 + 0.5 * model.disk_space / 1000 + 1e-5)
        
        # 节点适应性因子
        node_adaptation = 0.0
        
        # 如果节点准确率低，更倾向于选择低困惑度的模型
        if node_profile['avg_accuracy'] < self.accuracy_threshold:
            node_adaptation += 2.0 * accuracy_factor
        
        # 如果节点延迟高，更倾向于选择处理速度快的模型
        if node_profile['avg_latency'] > self.latency_threshold:
            node_adaptation += 1.5 * latency_factor
        
        # 如果节点吞吐量低，更倾向于选择资源消耗小的模型
        if node_profile['avg_throughput'] < self.throughput_threshold:
            node_adaptation += 1.2 * resource_factor
        
        # 如果节点负载不稳定，更倾向于选择加载时间短的模型
        if node_profile['load_stability'] < 0.7:
            node_adaptation += 1.3 * load_time_factor
        
        # 基础得分
        base_score = (
            2.0 * accuracy_factor +  # 准确率权重最高
            1.5 * latency_factor +   # 延迟其次
            1.0 * resource_factor +  # 资源消耗再次
            0.5 * load_time_factor   # 加载时间影响最小
        )
        
        # 最终得分 = 基础得分 + 节点适应性调整
        return base_score + node_adaptation

    def offload_models(self, node_performance: List[Dict[str, float]], current_loaded_models: List[int] = None) -> List[int]:
        """
        进行模型卸载
        
        Args:
            node_performance: 各节点性能指标
            current_loaded_models: 当前各节点已加载的模型ID，如果提供则更新当前模型状态
            
        Returns:
            各节点应该加载的模型ID列表
        """
        # 如果提供了当前加载的模型信息，更新当前状态
        if current_loaded_models is not None and len(current_loaded_models) == self.num_edge_nodes:
            self.current_node_models = current_loaded_models.copy()
            logging.info(f"更新当前节点模型状态: {self.current_node_models}")
        
        # 更新性能历史
        self.update_performance_history(node_performance)
        
        # 判断是否需要卸载
        if not self.should_offload():
            return self.current_node_models
        
        logging.info("开始模型卸载过程...")
        
        # 为每个节点选择最佳模型
        new_model_assignments = []
        for i in range(self.num_edge_nodes):
            best_model_id = self.select_best_model_for_node(i)
            new_model_assignments.append(best_model_id)
            
        # 打印卸载建议
        logging.info("模型卸载建议:")
        for node, model_id in enumerate(new_model_assignments):
            current_model_id = self.current_node_models[node]
            if model_id != current_model_id:
                logging.info(f"边缘节点 {node} 建议从模型 {current_model_id} 切换到模型 {model_id}")
            else:
                logging.info(f"边缘节点 {node} 保持当前模型 {model_id}")
        
        # 更新当前节点模型
        self.current_node_models = new_model_assignments
        
        # 重置计数器
        self.epoch_count = 0
        
        return new_model_assignments
# 使用示例
def main():
    # 创建多个模型信息
    model_list = model_init()
    # 创建模型卸载系统
    num_edge_nodes = 3
    offloading_system = ModelOffloadingSystem(
        num_edge_nodes=num_edge_nodes,
        model_list=model_list,
        accuracy_threshold=0.8,
        latency_threshold=0.3,
        throughput_threshold=10.0
    )
    
    # 当前各节点加载的模型ID
    current_loaded_models = [1, 2, 3]  # 初始状态：节点0加载模型1，节点1加载模型2，节点2加载模型3
    
    # 模拟边缘节点性能反馈
    for step in range(100):
        print(f"\n步骤 {step + 1}:")
        
        # 模拟节点性能
        if step < 30:
            # 前30步，性能良好
            node_performance = [
                {
                    'accuracy': np.random.uniform(0.8, 0.9),
                    'latency': np.random.uniform(0.1, 0.3),
                    'avg_throughput': np.random.uniform(10, 15)
                } for _ in range(num_edge_nodes)
            ]
        elif step < 60:
            # 30-60步，性能下降
            node_performance = [
                {
                    'accuracy': np.random.uniform(0.6, 0.8),
                    'latency': np.random.uniform(0.3, 0.6),
                    'avg_throughput': np.random.uniform(5, 10)
                } for _ in range(num_edge_nodes)
            ]
        else:
            # 60步之后，性能严重下降
            node_performance = [
                {
                    'accuracy': np.random.uniform(0.5, 0.7),
                    'latency': np.random.uniform(0.5, 0.8),
                    'avg_throughput': np.random.uniform(3, 8)
                } for _ in range(num_edge_nodes)
            ]
        
        # 进行模型卸载，同时传入当前加载的模型信息
        model_assignments = offloading_system.offload_models(
            node_performance=node_performance,
            current_loaded_models=current_loaded_models
        )
        
        # 更新当前加载的模型（在实际系统中，这里会是真实的模型加载操作的结果）
        current_loaded_models = model_assignments.copy()
        
        # 打印结果
        print(f"当前节点性能: {node_performance}")
        print(f"模型分配结果: {model_assignments}")
        
        # 模拟模型加载过程（在实际系统中，这里可能会有延迟和失败的情况）
        print(f"正在更新节点模型...")
        for node_idx, model_id in enumerate(model_assignments):
            print(f"节点 {node_idx} 加载模型 {model_id}")
if __name__ == "__main__":
    main()