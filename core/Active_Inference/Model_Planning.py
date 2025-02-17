import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List, Dict
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelInfo:
    def __init__(self, id, avg_inference_latency, avg_accuracy, memory_usage, energy_consumption, parameter_size, transmission_cost):
        self.id = id
        self.avg_inference_latency = avg_inference_latency
        self.avg_accuracy = avg_accuracy
        self.memory_usage = memory_usage
        self.energy_consumption = energy_consumption
        self.parameter_size = parameter_size
        self.transmission_cost = transmission_cost

class ActiveInferenceTaskAllocation:
    def __init__(self, num_edge_nodes: int, num_features: int, model_list: List[ModelInfo], accuracy_threshold: float):
        self.num_edge_nodes = num_edge_nodes
        self.num_features = num_features
        self.model_list = model_list
        self.accuracy_threshold = accuracy_threshold

        # 转移模型（神经网络）
        hidden_layer_size = max(64, num_features + num_edge_nodes)
        self.transition_model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size, hidden_layer_size // 2), max_iter=1000)
        self.transition_model_trained = False

        # 模型卸载相关
        self.epoch_count = 0
        self.accuracy_history = []

        # 节点性能历史
        self.node_performance_history = [[] for _ in range(num_edge_nodes)]

    def allocate_tasks(self, task_features: np.ndarray) -> np.ndarray:
        num_tasks = len(task_features)
        if not self.transition_model_trained:
            # 如果模型未训练，使用轮询分配以确保负载均衡
            return np.array([i % self.num_edge_nodes for i in range(num_tasks)])
        
        # 使用转移模型预测每个任务分配到每个节点的性能
        predictions = np.zeros((self.num_edge_nodes, num_tasks, 3))  # 3 for accuracy, latency, throughput
        for node in range(self.num_edge_nodes):
            node_assignments = np.full(num_tasks, node)
            X = np.column_stack([task_features, node_assignments])
            predictions[node] = self.transition_model.predict(X)
        
        # 计算每个节点的当前负载
        current_load = np.zeros(self.num_edge_nodes)
        for i, history in enumerate(self.node_performance_history):
            if history:
                # 使用最后10个记录，如果不足10个，就使用所有可用的记录
                recent_history = history[-10:]
                current_load[i] = np.mean([h['avg_throughput'] for h in recent_history])
        
        # 对每个任务，选择最佳节点
        assignments = np.zeros(num_tasks, dtype=int)
        for task in range(num_tasks):
            # 计算综合得分：高准确率，低延迟，考虑负载均衡
            scores = (
                predictions[:, task, 0] -  # 准确率 (越高越好)
                predictions[:, task, 1] -  # 延迟 (越低越好)
                0.1 * current_load  # 负载均衡因子 (当前负载越低越好)
            )
            best_node = np.argmax(scores)
            assignments[task] = best_node
            current_load[best_node] += 1  # 更新预计负载
        
        return assignments

    def update_model(self, task_features: np.ndarray, assignments: np.ndarray, performance: np.ndarray):
        X = np.column_stack([task_features, assignments])
        y = np.zeros((len(assignments), performance.shape[1]))
        
        for i, node in enumerate(assignments):
            y[i] = performance[node]
        
        if not self.transition_model_trained:
            self.transition_model.fit(X, y)
            self.transition_model_trained = True
        else:
            self.transition_model.partial_fit(X, y)

    def process(self, tasks: List[Dict]) -> List[int]:
        task_ids = [task['id'] for task in tasks]
        task_features = np.array([list(task['features'].values()) for task in tasks])

        assignments = self.allocate_tasks(task_features)

        self.last_task_ids = task_ids
        self.last_task_features = task_features
        self.last_assignments = assignments

        return assignments.tolist()

    def feedback(self, node_performance: List[Dict[str, float]]):
        performance = np.array([list(node.values()) for node in node_performance])

        # 更新节点性能历史
        for i, perf in enumerate(node_performance):
            self.node_performance_history[i].append(perf)
            if len(self.node_performance_history[i]) > 100:  # 保持最近100个时间片的历史
                self.node_performance_history[i].pop(0)

        self.update_model(self.last_task_features, self.last_assignments, performance)

        self.accuracy_history.append(np.mean([node['accuracy'] for node in node_performance]))
        self.epoch_count += 1

        if self.epoch_count >= 500 and np.mean(self.accuracy_history[-500:]) < self.accuracy_threshold:
            self.model_offloading()

        # 打印当前负载情况
        current_load = [np.mean([h['avg_throughput'] for h in history[-10:]]) if history else 0 
                        for history in self.node_performance_history]
        print("当前节点负载:", current_load)

    def model_offloading(self):
        logging.info("开始模型卸载过程...")
        
        dummy_features = np.zeros((self.num_edge_nodes, self.num_features))
        node_assignments = np.arange(self.num_edge_nodes)
        X = np.column_stack([dummy_features, node_assignments])
        node_states = self.transition_model.predict(X)
        
        offloading_suggestions = []
        for i, state in enumerate(node_states):
            best_model = None
            best_score = float('-inf')
            for model in self.model_list:
                score = (model.avg_accuracy / state[0] +
                         state[1] / model.avg_inference_latency +
                         model.memory_usage -
                         model.transmission_cost)
                if score > best_score:
                    best_score = score
                    best_model = model
            
            offloading_suggestions.append((i, best_model.id))
        
        logging.info("模型卸载建议:")
        for node, model_id in offloading_suggestions:
            logging.info(f"边缘节点 {node} 建议加载模型 ID: {model_id}")
        
        self.epoch_count = 0
        self.accuracy_history = []

# 运行示例
if __name__ == "__main__":
    num_edge_nodes = 3
    num_features = 7
    num_tasks = 10

    model_list = [
        ModelInfo(1, 0.1, 0.9, 100, 10, 1000000, 5),
        ModelInfo(2, 0.2, 0.95, 200, 20, 2000000, 10),
        ModelInfo(3, 0.3, 0.98, 300, 30, 3000000, 15)
    ]

    accuracy_threshold = 0.8

    allocator = ActiveInferenceTaskAllocation(num_edge_nodes, num_features, model_list, accuracy_threshold)

    for step in range(600):
        print(f"\n步骤 {step + 1}:")
        
        tasks = [
            {
                'id': i,
                'features': {
                    'vocabulary_complexity': np.random.uniform(0, 1),
                    'syntactic_complexity': np.random.uniform(0, 1),
                    'task_type': np.random.uniform(0, 1),
                    'context_dependency': np.random.uniform(0, 1),
                    'ambiguity_level': np.random.uniform(0, 1),
                    'information_density': np.random.uniform(0, 1),
                    'special_symbol_ratio': np.random.uniform(0, 1)
                }
            } for i in range(num_tasks)
        ]

        assignments = allocator.process(tasks)

        print("任务分配结果:")
        for i, task in enumerate(tasks):
            print(f"任务 {task['id']} 分配给节点: {assignments[i]}")

        node_performance = [
            {
                'accuracy': np.random.uniform(0.7, 0.9),
                'latency': np.random.uniform(0.1, 0.5),
                'avg_throughput': np.random.uniform(5, 15)
            } for _ in range(num_edge_nodes)
        ]

        allocator.feedback(node_performance)

        print("节点性能:", node_performance)
        print("节点任务数量:", [assignments.count(i) for i in range(num_edge_nodes)])