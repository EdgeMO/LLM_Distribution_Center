import numpy as np
from scipy.special import softmax
from typing import List, Dict
import logging

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

        self.num_performance_metrics = 3  # 准确率, 延迟, 负载

        # 内部生成模型参数
        self.A = np.random.rand(num_edge_nodes, self.num_performance_metrics, num_features)
        self.B = np.eye(self.num_performance_metrics)
        self.C = np.array([0.9, -0.1, -0.1])  # 期望状态 [高准确率, 低延迟, 低负载]
        
        # 信念状态
        self.mu = np.zeros((num_edge_nodes, self.num_performance_metrics))
        self.Sigma = np.eye(self.num_performance_metrics) * 0.1
        
        # 任务计数
        self.task_count = np.zeros(num_edge_nodes)

        self.epoch_count = 0
        self.accuracy_history = []
        self.performance_history = []

        # 自适应学习率参数
        self.learning_rate = 0.01
        self.min_learning_rate = 0.001
        self.learning_rate_decay = 0.999
        self.learning_rate_increase = 1.001

        # 动态权重参数
        self.accuracy_weight = 1.0
        self.latency_weight = 0.5
        self.load_weight = 0.3
        self.weight_adjust_rate = 0.01

    def safe_softmax(self, x):
        x_max = np.max(x)
        x_safe = x - x_max
        exp_x = np.exp(x_safe)
        return exp_x / (np.sum(exp_x) + 1e-10)

    def free_energy(self, s, a):
        predicted_obs = s
        prediction_error = s - self.mu[a]
        complexity = 0.5 * np.log(np.linalg.det(self.Sigma) + 1e-10)
        
        total_tasks = np.sum(self.task_count) + 1e-10
        load_balancing_error = self.task_count[a] / total_tasks - 1 / self.num_edge_nodes
        
        fe = (self.accuracy_weight * prediction_error[0]**2 +
              self.latency_weight * prediction_error[1]**2 +
              self.load_weight * load_balancing_error**2 +
              complexity)
        
        return np.clip(fe, -1e5, 1e5)

    def update_beliefs(self, obs, a):
        prediction_error = obs - self.mu[a]
        self.Sigma = np.linalg.inv(np.linalg.inv(self.Sigma) + np.eye(self.num_performance_metrics))
        self.mu[a] += np.dot(self.Sigma, prediction_error) * self.learning_rate

        # 自适应学习率
        if np.linalg.norm(prediction_error) > np.linalg.norm(self.mu[a]) * 0.1:
            self.learning_rate *= self.learning_rate_increase
        else:
            self.learning_rate *= self.learning_rate_decay
        self.learning_rate = np.clip(self.learning_rate, self.min_learning_rate, 0.1)

        # 添加约束以防止参数变得不稳定
        self.mu = np.clip(self.mu, -10, 10)
        self.Sigma = np.clip(self.Sigma, 1e-5, 10)

    def select_action(self, task_features):
        F = np.zeros(self.num_edge_nodes)
        for a in range(self.num_edge_nodes):
            s = np.dot(self.A[a], task_features)
            F[a] = self.free_energy(s, a)
        
        probabilities = self.safe_softmax(-F)
        
        if np.any(np.isnan(probabilities)) or np.sum(probabilities) == 0:
            logging.warning(f"Invalid probabilities detected: {probabilities}")
            logging.warning(f"Free energies: {F}")
            probabilities = np.ones(self.num_edge_nodes) / self.num_edge_nodes
        
        probabilities /= np.sum(probabilities)
        
        if np.any(np.isnan(probabilities)):
            logging.warning("NaN still present after normalization, using uniform distribution")
            probabilities = np.ones(self.num_edge_nodes) / self.num_edge_nodes
        
        return np.random.choice(self.num_edge_nodes, p=probabilities)

    def process(self, tasks: List[Dict]) -> List[int]:
        task_features = np.array([list(task['features'].values()) for task in tasks])
        num_tasks = len(tasks)
        
        initial_assignments = list(range(min(self.num_edge_nodes, num_tasks)))
        remaining_tasks = num_tasks - len(initial_assignments)
        
        if remaining_tasks > 0:
            additional_assignments = [self.select_action(tf) for tf in task_features[len(initial_assignments):]]
            assignments = initial_assignments + additional_assignments
        else:
            assignments = np.random.choice(self.num_edge_nodes, num_tasks, replace=False)
        
        for a in assignments:
            self.task_count[a] += 1
        
        self.last_task_features = task_features
        self.last_assignments = assignments
        
        return assignments

    def feedback(self, node_performance: List[Dict[str, float]]):
        performance = np.array([
            [node['accuracy'], node['latency'], node['avg_throughput']]
            for node in node_performance
        ])
        
        for a, perf in enumerate(performance):
            self.update_beliefs(perf, a)
        
        for tf, a in zip(self.last_task_features, self.last_assignments):
            pred = np.dot(self.A[a], tf)
            actual = performance[a]
            self.A[a] += 0.01 * np.outer(actual - pred, tf)
        
        self.A = np.clip(self.A, -10, 10)

        avg_accuracy = np.mean([node['accuracy'] for node in node_performance])
        avg_latency = np.mean([node['latency'] for node in node_performance])
        avg_throughput = np.mean([node['avg_throughput'] for node in node_performance])

        self.accuracy_history.append(avg_accuracy)
        self.performance_history.append({
            'accuracy': avg_accuracy,
            'latency': avg_latency,
            'throughput': avg_throughput
        })

        self.epoch_count += 1
        
        # 动态权重调整
        self.adjust_weights(avg_accuracy, avg_latency, avg_throughput)

        if self.epoch_count >= 500 and np.mean(self.accuracy_history[-500:]) < self.accuracy_threshold:
            self.model_offloading()

    def adjust_weights(self, accuracy, latency, throughput):
        # 动态调整权重
        if accuracy < self.accuracy_threshold:
            self.accuracy_weight += self.weight_adjust_rate
            self.latency_weight -= self.weight_adjust_rate / 2
            self.load_weight -= self.weight_adjust_rate / 2
        elif latency > 0.3:  # 假设0.3是一个可接受的延迟阈值
            self.latency_weight += self.weight_adjust_rate
            self.accuracy_weight -= self.weight_adjust_rate / 2
            self.load_weight -= self.weight_adjust_rate / 2
        else:
            self.load_weight += self.weight_adjust_rate
            self.accuracy_weight -= self.weight_adjust_rate / 2
            self.latency_weight -= self.weight_adjust_rate / 2

        # 确保权重和为1且非负
        total_weight = self.accuracy_weight + self.latency_weight + self.load_weight
        self.accuracy_weight = max(0, self.accuracy_weight / total_weight)
        self.latency_weight = max(0, self.latency_weight / total_weight)
        self.load_weight = max(0, self.load_weight / total_weight)

    def model_offloading(self):
        logging.info("开始模型卸载过程...")
        
        available_models = set(model.id for model in self.model_list)
        offloading_suggestions = []

        for i, mu in enumerate(self.mu):
            best_model = None
            best_score = float('-inf')
            for model in self.model_list:
                if model.id not in available_models:
                    continue
                score = (model.avg_accuracy / mu[0] +
                         mu[1] / model.avg_inference_latency +
                         model.memory_usage -
                         model.transmission_cost)
                if score > best_score:
                    best_score = score
                    best_model = model
            
            if best_model:
                offloading_suggestions.append((i, best_model.id))
                available_models.remove(best_model.id)
            else:
                if available_models:
                    random_model_id = np.random.choice(list(available_models))
                    offloading_suggestions.append((i, random_model_id))
                    available_models.remove(random_model_id)
                else:
                    logging.warning(f"没有足够的不同模型分配给节点 {i}")
        
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
        ModelInfo(3, 0.3, 0.98, 300, 30, 3000000, 15),
        ModelInfo(4, 0.15, 0.92, 150, 15, 1500000, 7),
        ModelInfo(5, 0.25, 0.97, 250, 25, 2500000, 12)
    ]

    accuracy_threshold = 0.8

    allocator = ActiveInferenceTaskAllocation(num_edge_nodes, num_features, model_list, accuracy_threshold)

    for step in range(600):
        print(f"\n步骤 {step + 1}:")
        
        tasks = [
            {
                'id': f"task_{i}",
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

        try:
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
            print("累计任务分配:", allocator.task_count)
            print(f"当前学习率: {allocator.learning_rate:.6f}")
            print(f"当前权重: 准确率={allocator.accuracy_weight:.2f}, 延迟={allocator.latency_weight:.2f}, 负载={allocator.load_weight:.2f}")
        
        except Exception as e:
            logging.error(f"步骤 {step + 1} 发生错误: {str(e)}")
            logging.error(f"当前状态: mu = {allocator.mu}, Sigma = {allocator.Sigma}, A = {allocator.A}")
            break

    # 触发模型卸载
    #allocator.model_offloading()