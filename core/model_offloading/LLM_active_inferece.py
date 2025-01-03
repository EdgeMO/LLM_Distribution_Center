import numpy as np
from scipy.special import softmax
"""
主要的变更在 select_action 方法中：

使用匈牙利算法（通过 scipy.optimize.linear_sum_assignment）来优化模型分配。这确保了在可能的情况下，每个节点都会分配到不同的模型。

处理了节点数量可能多于模型数量的情况：

首先为不同的节点分配唯一的模型。
如果还有剩余节点，它们会被分配到剩余的模型或选择最佳的已分配模型。
这种方法的优点包括：

最大化模型多样性：系统会尽可能地在不同节点上部署不同的模型。
优化整体性能：通过使用匈牙利算法，我们确保了总体的预期自由能最小化。
灵活性：即使在节点数量多于模型数量的情况下，系统也能做出合理的分配。
要进一步优化这个系统，你可以考虑：

实现一个机制来平衡模型多样性和性能优化，例如在某些情况下允许重复分配高性能模型。
添加模型切换的成本考虑，避免频繁的模型更换。
实现长期性能跟踪，以评估不同模型分配策略的长期效果。
考虑节点的硬件限制，确保分配的模型不会超出节点的能力范围。
这个更新后的实现应该能够在保证模型多样性的同时，仍然基于主动推理原则优化整体系统性能。
"""
class ActiveInferenceModelDistributor:
    def __init__(self, num_nodes, num_models, num_task_types, model_sizes, learning_rate=0.01):
        self.num_nodes = num_nodes
        self.num_models = num_models
        self.num_task_types = num_task_types
        self.model_sizes = model_sizes
        self.learning_rate = learning_rate

        # 初始化生成模型
        self.A = self.initialize_A()  # 观察模型
        self.B = self.initialize_B()  # 转移模型
        self.C = self.initialize_C()  # 偏好模型
        self.D = self.initialize_D()  # 先验信念

        # 当前信念状态 (准确率, 时延, 内存, 模型分配)
        self.beliefs = np.ones((num_nodes, 5, 5, 5, num_models)) / (5**3 * num_models)

        # 模型与任务类型的适配性矩阵
        self.model_task_compatibility = np.random.rand(num_models, num_task_types)

    def initialize_A(self):
        # 简化的观察模型
        return np.eye(5) * 0.8 + np.ones((5, 5)) * 0.04

    def initialize_B(self):
        # 简化的转移模型
        B = np.eye(5) * 0.6
        for i in range(5):
            if i > 0:
                B[i-1, i] = 0.2
            if i < 4:
                B[i+1, i] = 0.2
        return B

    def initialize_C(self):
        # 偏好模型：高准确率，低时延，高内存剩余
        C = np.zeros((5, 5, 5, self.num_models))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for m in range(self.num_models):
                        C[i,j,k,m] = i - j + k
        return softmax(C.ravel()).reshape(C.shape)

    def initialize_D(self):
        # 均匀先验
        return np.ones((5, 5, 5, self.num_models)) / (5**3 * self.num_models)

    def update_beliefs(self, observations):
        for node, obs in enumerate(observations):
            accuracy, latency, memory, current_model = obs
            acc_state = min(int(accuracy * 5), 4)
            lat_state = min(int((1 - latency) * 5), 4)
            mem_state = min(int(memory * 5), 4)

            likelihood = (self.A[acc_state, :, np.newaxis, np.newaxis, np.newaxis] * 
                          self.A[lat_state, np.newaxis, :, np.newaxis, np.newaxis] * 
                          self.A[mem_state, np.newaxis, np.newaxis, :, np.newaxis])
            
            self.beliefs[node] *= likelihood
            self.beliefs[node] /= self.beliefs[node].sum()

    def select_action(self, task_types):
        model_assignments = np.zeros(self.num_nodes, dtype=int)
        expected_free_energy = np.zeros((self.num_nodes, self.num_models))

        for node in range(self.num_nodes):
            for model in range(self.num_models):
                q_s = np.tensordot(self.B, self.beliefs[node], axes=([1],[0]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[1]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[2]))
                
                # 考虑模型大小对内存的影响
                mem_impact = self.model_sizes[model] / max(self.model_sizes)
                q_s[:,:,max(0, int((1-mem_impact)*5)):,:] = 0
                
                expected_free_energy[node, model] = np.sum(q_s * (np.log(q_s + 1e-10) - np.log(self.D + 1e-10) - np.log(self.C + 1e-10)))

                # 考虑模型与任务类型的适配性
                task_compatibility = np.mean([self.model_task_compatibility[model, t] for t in task_types])
                expected_free_energy[node, model] -= task_compatibility

        # 使用匈牙利算法来优化模型分配，确保每个节点分配不同的模型
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(expected_free_energy)
        
        model_assignments = col_ind

        # 如果节点数量多于模型数量，为剩余节点分配模型
        if self.num_nodes > self.num_models:
            remaining_nodes = set(range(self.num_nodes)) - set(row_ind)
            remaining_models = list(set(range(self.num_models)) - set(col_ind))
            for node in remaining_nodes:
                if remaining_models:
                    model_assignments[node] = remaining_models.pop(0)
                else:
                    # 如果没有剩余的唯一模型，选择最佳的已分配模型
                    model_assignments[node] = np.argmin(expected_free_energy[node])

        return model_assignments

    def update_model(self, assignments, performance):
        for node, model in enumerate(assignments):
            acc, lat, mem = performance[node]
            self.model_task_compatibility[model] += self.learning_rate * (acc - self.model_task_compatibility[model])

    def run(self, observations, task_types):
        self.update_beliefs(observations)
        assignments = self.select_action(task_types)
        return assignments

# 示例用法
def process(num_steps=100):
    num_nodes = 5
    num_models = 8
    num_task_types = 4
    model_sizes = np.random.uniform(1, 10, num_models)  # 模型大小（GB）
    
    distributor = ActiveInferenceModelDistributor(num_nodes, num_models, num_task_types, model_sizes)
    
    for step in range(num_steps):
        # 模拟观察数据 [准确率, 时延, 内存剩余量, 当前模型]
        observations = [
            [np.random.uniform(0.7, 1.0), np.random.uniform(0, 0.5), np.random.uniform(0.3, 1.0), np.random.randint(num_models)]
            for _ in range(num_nodes)
        ]
        
        # 模拟任务类型
        task_types = np.random.randint(0, num_task_types, 10)
        
        assignments = distributor.run(observations, task_types)
        
        # 模拟性能反馈
        performance = [
            [np.random.uniform(0.7, 1.0), np.random.uniform(0, 0.5), np.random.uniform(0.3, 1.0)]
            for _ in range(num_nodes)
        ]
        
        distributor.update_model(assignments, performance)
        
        if step % 10 == 0:
            print(f"Step {step}:")
            for node in range(num_nodes):
                print(f"  Node {node}: Model {assignments[node]}")
            print(f"  Task Types: {task_types}")
            print(f"  Observations: {observations}")
            print()

# 运行模拟
if __name__ == "__main__":
    process()