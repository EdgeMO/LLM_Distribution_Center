import numpy as np
from scipy.special import softmax


"""主要的变更和添加包括：

在 __init__ 方法中添加了 observation_counts 和 transition_counts，用于在线学习。

新增 update_observation_model 方法：

这个方法在每次观察后更新观察模型（A矩阵）。
它使用计数器来累积观察数据，然后更新A矩阵。
新增 update_transition_model 方法：

这个方法在每次任务分配后更新转移模型（B矩阵）。
它记录状态转换，并使用这些信息来更新B矩阵。
在 update_beliefs 方法中，每次更新信念后调用 update_observation_model。

在 select_action 方法中，每次选择动作后调用 update_transition_model。

修改了 process 函数，每100步打印一次A矩阵和B矩阵，以便观察它们如何随时间变化。

这些改变实现了在线学习机制，使系统能够：

根据实际观察动态调整其观察模型，提高对环境的理解。
根据实际的状态转换调整其转移模型，更好地预测行动的结果。
随着时间的推移，不断优化其决策过程，适应环境的变化。
通过这种方式，系统可以在长期运行中不断学习和改进，提高任务分配的效率和准确性。你可以通过观察A矩阵和B矩阵的变化来了解系统如何适应环境。

要进一步优化这个在线学习机制，你可以考虑：

实现学习率的动态调整，使系统在早期学习快速，后期更稳定。
添加正则化项，防止过拟合。
实现周期性的模型重置或调整，以应对环境的突变。
使用更复杂的学习算法，如贝叶斯更新或梯度下降。
这个增强版的实现提供了一个更加灵活和自适应的任务分发系统，能够随时间学习和改善其性能。
"""
    
    
class ActiveInferenceTaskDistributorOnlineLearning:
    def __init__(self, num_nodes, batch_num, num_states=5, learning_rate=0.01):
        self.num_nodes = num_nodes
        self.batch_num = batch_num
        self.num_states = num_states
        self.learning_rate = learning_rate

        # 初始化生成模型
        self.A = self.initialize_A()  # 观察模型
        self.B = self.initialize_B()  # 转移模型
        self.C = self.initialize_C()  # 偏好模型
        self.D = self.initialize_D()  # 先验信念

        # 当前信念状态 (准确度, 时延, 内存)
        self.beliefs = np.ones((num_nodes, num_states, num_states, num_states)) / (num_states**3)

        # 用于在线学习的计数器
        self.observation_counts = np.ones_like(self.A) * 0.1
        self.transition_counts = np.ones_like(self.B) * 0.1

    def initialize_A(self):
        A = np.eye(self.num_states) * 0.8
        noise = np.ones((self.num_states, self.num_states)) * 0.2 / self.num_states
        A += noise
        return A / A.sum(axis=0, keepdims=True)

    def initialize_B(self):
        B = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            B[i, i] = 0.6
            B[min(i+1, self.num_states-1), i] = 0.2
            B[max(0, i-1), i] = 0.2
        return B

    def initialize_C(self):
        C = np.zeros((self.num_states, self.num_states, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                for k in range(self.num_states):
                    C[i,j,k] = i - j + k  # 高准确度，低时延，高内存剩余
        return softmax(C.ravel()).reshape(C.shape)

    def initialize_D(self):
        return np.ones((self.num_states, self.num_states, self.num_states)) / (self.num_states**3)
    def update_beliefs(self, observations):
        for node, obs in enumerate(observations):
            _, accuracy, latency, memory = obs
            acc_state = int(accuracy * (self.num_states - 1))
            lat_state = int((1 - latency) * (self.num_states - 1))
            mem_state = int(memory * (self.num_states - 1))

            likelihood = (self.A[:, acc_state][:, np.newaxis, np.newaxis] * 
                          self.A[:, lat_state][np.newaxis, :, np.newaxis] * 
                          self.A[:, mem_state][np.newaxis, np.newaxis, :])
            
            self.beliefs[node] *= likelihood
            self.beliefs[node] /= self.beliefs[node].sum()

            # 在线学习：更新观察模型
            self.update_observation_model(node, (acc_state, lat_state, mem_state))

    def update_observation_model(self, node, observed_state):
        for dim, state in enumerate(observed_state):
            self.observation_counts[:, state] += self.beliefs[node].sum(axis=tuple(i for i in range(3) if i != dim))
        
        # 使用计数更新A矩阵
        self.A = self.observation_counts / self.observation_counts.sum(axis=0, keepdims=True)

    def update_transition_model(self, node, previous_state, current_state):
        for dim in range(3):
            self.transition_counts[current_state[dim], previous_state[dim]] += 1

        # 使用计数更新B矩阵
        self.B = self.transition_counts / self.transition_counts.sum(axis=0, keepdims=True)

    def select_action(self):
        assignments = np.zeros(self.batch_num, dtype=int)
        for task in range(self.batch_num):
            expected_free_energy = np.zeros(self.num_nodes)
            for node in range(self.num_nodes):
                q_s = np.tensordot(self.B, self.beliefs[node], axes=([1],[0]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[0]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[0]))
                expected_free_energy[node] = np.sum(q_s * (np.log(q_s + 1e-10) - np.log(self.D + 1e-10) - np.log(self.C + 1e-10)))
            
            selected_node = np.argmin(expected_free_energy)
            assignments[task] = selected_node

            # 更新选中节点的信念
            previous_state = np.unravel_index(np.argmax(self.beliefs[selected_node]), self.beliefs[selected_node].shape)
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] /= self.beliefs[selected_node].sum()
            
            # 在线学习：更新转移模型
            current_state = np.unravel_index(np.argmax(self.beliefs[selected_node]), self.beliefs[selected_node].shape)
            self.update_transition_model(selected_node, previous_state, current_state)

        return assignments

    def run(self, observations):
        self.update_beliefs(observations)
        assignments = self.select_action()
        return assignments

# 模拟边缘计算场景
def process(num_steps=1000):
    distributor = ActiveInferenceTaskDistributorOnlineLearning(num_nodes=3, batch_num=10)
    
    for step in range(num_steps):
        # 模拟观察数据 [节点索引, 准确率, 时延, 内存剩余量]
        observations = [
            [0, np.random.uniform(0.8, 1.0), np.random.uniform(0, 0.5), np.random.uniform(0.3, 1.0)],
            [1, np.random.uniform(0.75, 0.95), np.random.uniform(0.1, 0.6), np.random.uniform(0.4, 1.0)],
            [2, np.random.uniform(0.85, 1.0), np.random.uniform(0, 0.4), np.random.uniform(0.5, 1.0)]
        ]
        
        assignments = distributor.run(observations)
        
        if step % 100 == 0:
            print(f"Step {step}:")
            for node in range(distributor.num_nodes):
                tasks = np.where(assignments == node)[0]
                print(f"  Node {node}: Tasks {tasks}")
            print(f"  Observations: {observations}")
            #print(f"  A matrix:\n{distributor.A}")
            #print(f"  B matrix:\n{distributor.B}")
            print()

# 运行模拟
if __name__ == "__main__":
    process()