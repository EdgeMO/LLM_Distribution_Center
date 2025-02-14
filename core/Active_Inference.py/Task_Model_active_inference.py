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

class ActiveInferenceTaskAndModelAllocator:
    def __init__(self, num_edge_nodes, num_features, model_infos, history_size=1000, accuracy_threshold=0.7):
        self.num_edge_nodes = num_edge_nodes
        self.num_features = num_features
        self.model_infos = model_infos  # 存储所有模型的固有属性
        self.accuracy_threshold = accuracy_threshold
        
        # 初始化其他属性
        self.prior = np.ones((num_edge_nodes, 3)) / 3  # [准确率, 1/时延, 吞吐量]
        self.A = np.random.rand(num_features, 3, num_edge_nodes)
        self.B = np.eye(num_edge_nodes)
        self.C = np.array([0.3, 0.3, 0.3, 0.1])  # 准确率、时延、吞吐量、负载均衡的权重
        self.pi = np.ones(3)
        self.policy = np.ones(num_edge_nodes) / num_edge_nodes
        self.task_counts = np.zeros(num_edge_nodes)
        self.edge_node_models = [None] * num_edge_nodes
        self.edge_node_memory = np.ones(num_edge_nodes) * 1000  # 假设每个节点初始有1000单位内存
        self.history = deque(maxlen=history_size)
        self.epoch_accuracy = []

    def process(self, time_slice_data):
        """
        处理单一时间片的数据，给出任务分配和模型卸载的建议。

        :param time_slice_data: 包含单一时间片数据的字典
        {
            'tasks': [
                {
                    'id': int,
                    'vocab_complexity': float,
                    'syntax_complexity': float,
                    'task_type': float,
                    'context_dependency': float,
                    'ambiguity': float,
                    'info_density': float,
                    'special_char_ratio': float
                },
                ...
            ],
            'edge_nodes': [
                {
                    'avg_accuracy': float,
                    'avg_latency': float,
                    'avg_throughput': float,
                    'remaining_memory': float
                },
                ...
            ]
        }
        :return: 包含任务分配和模型卸载建议的字典
        """
        task_allocation = {}
        total_accuracy = 0

        # 任务分配
        for task in time_slice_data['tasks']:
            task_features = [
                task['vocab_complexity'], task['syntax_complexity'], task['task_type'],
                task['context_dependency'], task['ambiguity'], task['info_density'],
                task['special_char_ratio']
            ]
            assigned_node = self.allocate_task(task_features)
            task_allocation[task['id']] = assigned_node

        # 更新边缘节点状态和计算总体准确率
        for node, node_data in enumerate(time_slice_data['edge_nodes']):
            total_accuracy += node_data['avg_accuracy']
            self.prior[node] = [node_data['avg_accuracy'], 1/node_data['avg_latency'], node_data['avg_throughput']]
            self.edge_node_memory[node] = node_data['remaining_memory']
            self.task_counts[node] = node_data['avg_throughput']

        # 计算平均准确率并检查是否需要模型卸载
        avg_accuracy = total_accuracy / self.num_edge_nodes
        self.epoch_accuracy.append(avg_accuracy)
        
        model_allocation = None
        if len(self.epoch_accuracy) >= 500 and np.mean(self.epoch_accuracy) < self.accuracy_threshold:
            model_allocation = self.model_offloading()
            self.epoch_accuracy = []

        return {
            'task_allocation': task_allocation,
            'model_allocation': model_allocation
        }

    def allocate_task(self, task_features):
        observations = np.array(task_features)
        self.update_beliefs(observations)
        self.update_policy(observations)
        self.transition()
        
        temperature = 1.0
        adjusted_policy = softmax(np.clip(np.log(self.policy + 1e-10) / temperature, -100, 100))
        
        if np.isnan(adjusted_policy).any() or np.sum(adjusted_policy) == 0:
            adjusted_policy = np.ones(self.num_edge_nodes) / self.num_edge_nodes
        else:
            adjusted_policy /= np.sum(adjusted_policy)
        
        assigned_node = np.random.choice(self.num_edge_nodes, p=adjusted_policy)
        self.task_counts[assigned_node] += 1
        
        return assigned_node

    def update_beliefs(self, observations):
        likelihood = np.exp(np.clip(np.einsum('f,fna->na', observations, self.A), -100, 100))
        posterior = self.prior * likelihood
        self.prior = posterior / (np.sum(posterior, axis=1, keepdims=True) + 1e-10)

    def update_policy(self, observations):
        free_energies = np.array([self.calculate_free_energy(observations, i) for i in range(self.num_edge_nodes)])
        self.policy = softmax(-np.clip(free_energies, -100, 100))

    def transition(self):
        self.prior = np.dot(self.B, self.prior)

    def calculate_free_energy(self, observations, node):
        expected_obs = np.dot(observations, self.A[:, :, node])
        entropy = -np.sum(self.prior[node] * np.log(self.prior[node] + 1e-10))
        load_balance_factor = self.task_counts[node] / (np.sum(self.task_counts) + 1e-10)
        return 0.5 * np.sum(self.pi * (expected_obs - self.prior[node])**2) - entropy + load_balance_factor

    def model_offloading(self):
        model_allocation = {}
        for node in range(self.num_edge_nodes):
            best_model = None
            best_score = float('-inf')
            
            for model in self.model_infos:
                if model.memory_usage <= self.edge_node_memory[node]:
                    score = (model.avg_accuracy / model.avg_latency) * (self.edge_node_memory[node] / model.memory_usage)
                    if score > best_score:
                        best_score = score
                        best_model = model

            if best_model:
                model_allocation[node] = best_model.id
                self.edge_node_models[node] = best_model
                self.edge_node_memory[node] -= best_model.memory_usage
            else:
                model_allocation[node] = None

        return model_allocation

# 使用示例
if __name__ == "__main__":
    num_edge_nodes = 3
    num_features = 7
    model_infos = [
        ModelInfo(1, 10, 0.9, 200, 50, 1000000, 100),
        ModelInfo(2, 5, 0.85, 150, 40, 500000, 80),
        ModelInfo(3, 15, 0.95, 250, 60, 2000000, 120)
    ]
    
    allocator = ActiveInferenceTaskAndModelAllocator(num_edge_nodes, num_features, model_infos, accuracy_threshold=0.75)

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
            print(f"  信念: {allocator.prior[node]}")
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

    print(f"\n最终偏好权重: {allocator.C}")