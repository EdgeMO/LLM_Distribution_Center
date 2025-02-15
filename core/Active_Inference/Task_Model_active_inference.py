import numpy as np
from scipy.special import softmax
from collections import deque

class ModelInfo:
    def __init__(self, model_id, avg_latency, avg_accuracy, memory_usage, energy_consumption, param_size, transfer_cost):
        self.id = model_id
        self.avg_latency = avg_latency
        self.avg_accuracy = avg_accuracy
        self.memory_usage = memory_usage
        self.energy_consumption = energy_consumption
        self.param_size = param_size
        self.transfer_cost = transfer_cost

class ActiveInference:
    def __init__(self, num_edge_nodes, num_features, model_infos, history_size=1000, accuracy_threshold=0.7):
        self.num_edge_nodes = num_edge_nodes
        self.num_features = num_features
        self.model_infos = model_infos
        self.accuracy_threshold = accuracy_threshold
        
        # 任务分配相关参数
        self.task_prior = np.ones((num_edge_nodes, 3)) / 3  # [准确率, 1/时延, 吞吐量]
        self.task_A = np.random.rand(num_features, 3, num_edge_nodes)
        self.task_B = np.eye(num_edge_nodes)
        self.task_C = np.array([0.3, 0.3, 0.3, 0.1])  # 准确率、时延、吞吐量、负载均衡的权重
        self.task_pi = np.ones(3)
        self.task_policy = np.ones(num_edge_nodes) / num_edge_nodes
        
        # 模型卸载相关参数
        self.model_prior = np.ones((num_edge_nodes, len(model_infos))) / len(model_infos)
        self.model_A = np.random.rand(4, len(model_infos), num_edge_nodes)  # 4个特征：准确率、时延、内存、能耗
        self.model_B = np.eye(num_edge_nodes)
        self.model_C = np.array([0.3, 0.3, 0.2, 0.2])  # 准确率、时延、内存、能耗的权重
        self.model_pi = np.ones(4)
        self.model_policy = np.ones((num_edge_nodes, len(model_infos))) / len(model_infos)
        
        self.edge_node_models = [None] * num_edge_nodes
        self.edge_node_memory = np.ones(num_edge_nodes) * 1000  # 假设每个节点初始有1000单位内存
        self.task_counts = np.zeros(num_edge_nodes)
        self.history = deque(maxlen=history_size)
        self.epoch_accuracy = []

    def process(self, time_slice_data):
        task_allocation = {}
        total_accuracy = 0

        # 任务分配
        for task in time_slice_data['tasks']:
            task_features = [task['vocab_complexity'], task['syntax_complexity'], task['task_type'],
                             task['context_dependency'], task['ambiguity'], task['info_density'],
                             task['special_char_ratio']]
            assigned_node = self.allocate_task(task_features)
            task_allocation[task['id']] = assigned_node

        # 更新边缘节点状态和计算总体准确率
        for node, node_data in enumerate(time_slice_data['edge_nodes']):
            total_accuracy += node_data['avg_accuracy']
            self.task_prior[node] = [node_data['avg_accuracy'], 1/node_data['avg_latency'], node_data['avg_throughput']]
            self.edge_node_memory[node] = node_data['remaining_memory']
            self.task_counts[node] = node_data['avg_throughput']

        # 计算平均准确率并检查是否需要模型卸载
        avg_accuracy = total_accuracy / self.num_edge_nodes
        self.epoch_accuracy.append(avg_accuracy)
        
        model_allocation = None
        if len(self.epoch_accuracy) >= 500 and np.mean(self.epoch_accuracy) < self.accuracy_threshold:
            model_allocation = self.model_offloading(time_slice_data['edge_nodes'])
            self.epoch_accuracy = []

        return {
            'task_allocation': task_allocation,
            'model_allocation': model_allocation
        }

    def allocate_task(self, task_features):
        observations = np.array(task_features)
        self.update_task_beliefs(observations)
        self.update_task_policy(observations)
        self.task_transition()
        
        temperature = 1.0
        adjusted_policy = softmax(np.clip(np.log(self.task_policy + 1e-10) / temperature, -100, 100))
        
        if np.isnan(adjusted_policy).any() or np.sum(adjusted_policy) == 0:
            adjusted_policy = np.ones(self.num_edge_nodes) / self.num_edge_nodes
        else:
            adjusted_policy /= np.sum(adjusted_policy)
        
        assigned_node = np.argmax(self.task_policy)
        self.task_counts[assigned_node] += 1
        
        return assigned_node

    def update_task_beliefs(self, observations):
        likelihood = np.exp(np.clip(np.einsum('f,fna->na', observations, self.task_A), -100, 100))
        posterior = self.task_prior * likelihood
        self.task_prior = posterior / (np.sum(posterior, axis=1, keepdims=True) + 1e-10)

    def update_task_policy(self, observations):
        free_energies = np.array([self.calculate_task_free_energy(observations, i) for i in range(self.num_edge_nodes)])
        self.task_policy = softmax(-np.clip(free_energies, -100, 100))
        epsilon = 0.1  # 探索率
        if np.random.random() < epsilon:
            self.task_policy = np.random.dirichlet(np.ones(self.num_edge_nodes))

    def task_transition(self):
        self.task_prior = np.dot(self.task_B, self.task_prior)

    def calculate_task_free_energy(self, observations, node):
        expected_obs = np.dot(observations, self.task_A[:, :, node])
        entropy = -np.sum(self.task_prior[node] * np.log(self.task_prior[node] + 1e-10))
        load_balance_factor = self.task_counts[node] / (np.sum(self.task_counts) + 1e-10)
        return 0.5 * np.sum(self.task_pi * (expected_obs - self.task_prior[node])**2) - entropy + load_balance_factor

    def model_offloading(self, edge_nodes):
        model_allocation = {}
        
        for node in range(self.num_edge_nodes):
            node_data = edge_nodes[node]
            observations = np.array([
                node_data['avg_accuracy'],
                1 / node_data['avg_latency'],
                node_data['remaining_memory'] / 1000,  # 归一化内存
                1 / node_data['avg_throughput']  # 用吞吐量的倒数作为能耗的近似
            ])
            
            self.update_model_beliefs(observations, node)
            self.update_model_policy(observations, node)
            self.model_transition(node)
            
            # 使用 Thompson 采样来平衡探索和利用
            sampled_policy = np.random.dirichlet(self.model_policy[node] * 10)
            chosen_model_index = np.argmax(sampled_policy)
            chosen_model = self.model_infos[chosen_model_index]
            
            # 考虑内存约束和模型性能
            if chosen_model.memory_usage <= node_data['remaining_memory']:
                expected_performance = (
                    chosen_model.avg_accuracy / chosen_model.avg_latency *
                    (node_data['remaining_memory'] / chosen_model.memory_usage)
                )
                current_performance = node_data['avg_accuracy'] / node_data['avg_latency']
                
                if expected_performance > current_performance:
                    model_allocation[node] = chosen_model.id
                    self.edge_node_models[node] = chosen_model
                    self.edge_node_memory[node] -= chosen_model.memory_usage
                else:
                    model_allocation[node] = None  # 保持当前模型
            else:
                model_allocation[node] = None  # 内存不足，无法加载新模型
        
        return model_allocation
    def update_model_beliefs(self, observations, node):
        likelihood = np.exp(np.clip(np.einsum('f,fna->na', observations, self.model_A[:, :, node]), -100, 100))
        posterior = self.model_prior[node] * likelihood
        self.model_prior[node] = posterior / (np.sum(posterior) + 1e-10)

    def update_model_policy(self, observations, node):
        free_energies = np.array([self.calculate_model_free_energy(observations, node, i) for i in range(len(self.model_infos))])
        
        # 考虑模型的内存使用
        memory_factors = np.array([model.memory_usage for model in self.model_infos])
        normalized_memory_factors = memory_factors / np.max(memory_factors)
        
        # 结合自由能和内存因子
        combined_energies = free_energies + self.memory_weight * normalized_memory_factors
        
        self.model_policy[node] = softmax(-np.clip(combined_energies, -100, 100))
        
        # 动态调整探索率
        uncertainty = np.std(self.model_policy[node])
        epsilon = min(0.1, uncertainty)  # 最大探索率为0.1
        if np.random.random() < epsilon:
            self.model_policy[node] = np.random.dirichlet(np.ones(len(self.model_infos)))

    def model_transition(self, node):
        self.model_prior[node] = np.dot(self.model_B, self.model_prior[node])

    def calculate_model_free_energy(self, observations, node, model_index):
        expected_obs = np.dot(observations, self.model_A[:, model_index, node])
        entropy = -np.sum(self.model_prior[node] * np.log(self.model_prior[node] + 1e-10))
        return 0.5 * np.sum(self.model_pi * (expected_obs - self.model_prior[node, model_index])**2) - entropy

# 使用示例
if __name__ == "__main__":
    num_edge_nodes = 3
    num_features = 7
    model_infos = [
        ModelInfo(1, 10, 0.9, 200, 50, 1000000, 100),
        ModelInfo(2, 5, 0.85, 150, 40, 500000, 80),
        ModelInfo(3, 15, 0.95, 250, 60, 2000000, 120)
    ]
    
    allocator = ActiveInference(num_edge_nodes, num_features, model_infos, accuracy_threshold=0.9)

    # 模拟多个时间帧
    num_frames = 100  # 模拟10个时间帧
    tasks_per_frame = 5  # 每帧5个任务

    for frame in range(num_frames):
        print(f"\n--- 时间帧 {frame + 1} ---")

        # 生成时间帧数据
        time_slice_data = {
            'tasks': [
                {
                    'id': i,
                    'vocab_complexity': np.random.random(),
                    'syntax_complexity': np.random.random(),
                    'task_type': np.random.random(),
                    'context_dependency': np.random.random(),
                    'ambiguity': np.random.random(),
                    'info_density': np.random.random(),
                    'special_char_ratio': np.random.random()
                } for i in range(tasks_per_frame)
            ],
            'edge_nodes': [
                {
                    'avg_accuracy': np.random.uniform(0.6, 0.9),
                    'avg_latency': np.random.uniform(60, 180),
                    'avg_throughput': np.random.uniform(6, 14),
                    'remaining_memory': np.random.uniform(500, 1000)
                } for _ in range(num_edge_nodes)
            ]
        }

        # 处理时间帧数据
        result = allocator.process(time_slice_data)

        # 打印任务分配结果
        print("任务分配结果:")
        for task_id, node in result['task_allocation'].items():
            print(f"任务 {task_id} 分配给边缘节点 {node}")

        # 打印模型分配建议（如果有）
        if result['model_allocation']:
            print("\n模型分配建议:")
            for node, model_id in result['model_allocation'].items():
                print(f"边缘节点 {node}: 建议加载模型 ID {model_id}" if model_id is not None else f"边缘节点 {node}: 无合适模型")
        else:
            print("\n本时间帧未触发模型卸载")

        # 打印当前系统状态
        print("\n当前系统状态:")
        for node in range(num_edge_nodes):
            print(f"边缘节点 {node}:")
            print(f"  任务计数: {allocator.task_counts[node]}")
            print(f"  剩余内存: {allocator.edge_node_memory[node]:.2f}")
            if allocator.edge_node_models[node]:
                print(f"  当前模型: ID {allocator.edge_node_models[node].id}")
            else:
                print("  当前无加载模型")

    # 打印最终统计信息
    print("\n--- 最终统计信息 ---")
    print("任务分配总计:")
    for node in range(num_edge_nodes):
        print(f"边缘节点 {node}: {allocator.task_counts[node]} 任务")

    print("\n最终模型分配:")
    for node in range(num_edge_nodes):
        if allocator.edge_node_models[node]:
            print(f"边缘节点 {node}: 模型 ID {allocator.edge_node_models[node].id}")
        else:
            print(f"边缘节点 {node}: 无模型")
