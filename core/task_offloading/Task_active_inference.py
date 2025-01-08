import numpy as np
from scipy.special import softmax
"""
用主动推理的方法设计一个任务分发系统。现在需要完成主动推理的算法代码的编写。
在这个系统中，中心节点会收到一批任务，这一批任务可能有10个请求，基于主动推理的算法，想要把这批任务分发到不同的边缘节点上。每个边缘节点上完成对任务的准确率统计以及对所接收到的任务统计时延，以及边缘节点的内存剩余量，会返回给中心节点。最终通过主动推理得到每次一运行的时候给出 同一批任务里面哪一部分被分发到哪一个边缘节点的编号上。
"""

"""这个实现具有以下特点：

ActiveInferenceTaskDistributor 类实现了主动推理算法：

初始化包括观察模型(A)、转移模型(B)、偏好模型(C)和先验信念(D)
update_beliefs 方法根据观察更新信念状态。
select_action 方法为每个任务选择最优的边缘节点。
观察数据格式为 [节点索引, 准确率, 时延, 内存剩余量]，符合您的需求。

在每次运行时，系统会为10个任务分别选择最优的边缘节点。

process 函数模拟了整个过程，包括生成模拟观察数据和运行任务分发算法。

输出显示了每个步骤中各节点被分配的任务以及观察数据。

要进一步优化这个系统，您可以考虑：

调整 num_states 参数来改变状态空间的粒度。
修改 initialize_C 方法中的偏好计算，以根据您的具体需求调整系统的目标函数。
实现在线学习机制，动态更新 A 和 B 矩阵。
添加更复杂的探索策略，如ε-贪心或上限置信区间（UCB）算法。
根据实际系统的特性，调整观察数据的生成方式。
增加性能指标的计算和记录，以评估系统的长期表现。

"""
    
    
class ActiveInferenceTaskDistributor:
    def __init__(self, num_nodes, num_tasks, num_states=5, learning_rate=0.01):
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.num_states = num_states
        self.learning_rate = learning_rate

        # 初始化生成模型
        self.A = self.initialize_A()  # 观察模型
        self.B = self.initialize_B()  # 转移模型
        self.C = self.initialize_C()  # 偏好模型
        self.D = self.initialize_D()  # 先验信念

        # 当前信念状态 (准确度, 时延, 内存)
        self.beliefs = np.ones((num_nodes, num_states, num_states, num_states)) / (num_states**3)

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

    def select_action(self):
        assignments = np.zeros(self.num_tasks, dtype=int)
        for task in range(self.num_tasks):
            expected_free_energy = np.zeros(self.num_nodes)
            for node in range(self.num_nodes):
                q_s = np.tensordot(self.B, self.beliefs[node], axes=([1],[0]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[0]))
                q_s = np.tensordot(self.B, q_s, axes=([1],[0]))
                expected_free_energy[node] = np.sum(q_s * (np.log(q_s + 1e-10) - np.log(self.D + 1e-10) - np.log(self.C + 1e-10)))
            
            selected_node = np.argmin(expected_free_energy)
            assignments[task] = selected_node

            # 更新选中节点的信念
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] = np.tensordot(self.B, self.beliefs[selected_node], axes=([1],[0]))
            self.beliefs[selected_node] /= self.beliefs[selected_node].sum()

        return assignments

    def run(self, observations):
        self.update_beliefs(observations)
        assignments = self.select_action()
        return assignments

# 模拟计算场景
def process(num_steps=100):
    distributor = ActiveInferenceTaskDistributor(num_nodes=3, num_tasks=10)
    
    for step in range(num_steps):
        # 模拟观察数据 [节点索引, 准确率, 时延, 内存剩余量]
        observations = [
            [0, np.random.uniform(0.8, 1.0), np.random.uniform(0, 0.5), np.random.uniform(0.3, 1.0)],
            [1, np.random.uniform(0.75, 0.95), np.random.uniform(0.1, 0.6), np.random.uniform(0.4, 1.0)],
            [2, np.random.uniform(0.85, 1.0), np.random.uniform(0, 0.4), np.random.uniform(0.5, 1.0)]
        ]
        
        assignments = distributor.run(observations)
        
        print(f"Step {step}:")
        for node in range(distributor.num_nodes):
            tasks = np.where(assignments == node)[0]
            print(f"  Node {node}: Tasks {tasks}")
        print(f"  Observations: {observations}")
        print()

# 运行模拟
if __name__ == "__main__":
    process()