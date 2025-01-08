import numpy as np
"""主动推理的元素：

系统维护了一个内部模型（状态和权重）。
它根据观察更新这个模型。
它使用这个模型来指导动作选择（任务分配）。
系统从经验中学习，不断调整其模型（权重更新）。
与典型主动推理的差异：

没有明确的生成模型（如观察模型A和状态转移模型B）。
没有显式的自由能最小化过程。
没有使用变分推断来更新信念状态。
动作选择更多地基于当前状态，而不是对未来状态的预测。
这个实现的特点：

使用了一个简化的模型，直接通过权重来表示不同因素（延迟、准确率、内存）的重要性。
采用了一种启发式的方法来选择动作，使用softmax函数将得分转换为概率分布。
实现了一个简单的在线学习机制，通过调整权重来适应观察到的性能。
虽然这个实现不是标准的主动推理算法，但它确实体现了一些主动推理的核心思想，如维护内部模型、基于模型做决策、从经验中学习等。它可以被视为一个简化的、启发式的主动推理方法。

要使其更接近典型的主动推理，你可以考虑以下改进：

引入明确的生成模型，包括观察模型和状态转移模型。
实现变分推断来更新信念状态。
使用自由能最小化作为动作选择的标准。
加入对未来状态的预测和评估。
尽管如此，当前的实现对于许多实际应用可能已经足够有效，特别是在计算资源有限或需要快速决策的情况下。它提供了一个在计算效率和实现简单性之间的良好平衡。

Returns:
    _type_: _description_
"""
class SimpleInference:
    def __init__(self, num_nodes, num_tasks, learning_rate=0.1):
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate

        # 初始化状态
        self.delay = np.zeros(num_nodes)
        self.accuracy = np.zeros(num_nodes)
        self.memory = np.zeros(num_nodes)

        # 初始化模型参数
        self.delay_weight = np.ones(num_nodes)
        self.accuracy_weight = np.ones(num_nodes)
        self.memory_weight = np.ones(num_nodes)

    def update_state(self, observations):
        """
        更新系统状态
        :param observations: 列表的列表，每个内部列表包含 [节点索引, 时延, 准确率, 内存剩余量]
        """
        for node, delay, accuracy, memory in observations:
            self.delay[node] = delay
            self.accuracy[node] = accuracy
            self.memory[node] = memory

    def select_action(self):
        # 计算每个节点的得分
        scores = (self.accuracy * self.accuracy_weight +
                  self.memory * self.memory_weight -
                  self.delay * self.delay_weight)

        # 使用softmax来计算任务分配概率
        probabilities = np.exp(scores) / np.sum(np.exp(scores))

        # 为每个任务分配节点
        assignments = np.random.choice(self.num_nodes, size=self.num_tasks, p=probabilities)

        return assignments

    def update_model(self, assignments):
        """
        更新模型权重
        """
        for node in range(self.num_nodes):
            num_assigned = np.sum(assignments == node)
            if num_assigned > 0:
                # 更新权重
                self.delay_weight[node] += self.learning_rate * self.delay[node] * num_assigned / self.num_tasks
                self.accuracy_weight[node] += self.learning_rate * self.accuracy[node] * num_assigned / self.num_tasks
                self.memory_weight[node] += self.learning_rate * self.memory[node] * num_assigned / self.num_tasks

        # 正则化权重
        total_weight = self.delay_weight + self.accuracy_weight + self.memory_weight
        self.delay_weight /= total_weight
        self.accuracy_weight /= total_weight
        self.memory_weight /= total_weight

    def run(self, observations):
        self.update_state(observations)
        assignments = self.select_action()
        self.update_model(assignments)
        return assignments

# 示例用法
def process(num_steps=100):
    distributor = SimpleInference(num_nodes=3, num_tasks=10)
    
    for step in range(num_steps):
        # 从真实系统获取观察数据
        # 格式: [节点索引, 时延, 准确率, 内存剩余量]
        observations = [
            [0, np.random.exponential(1), np.random.uniform(0.8, 1.0), np.random.uniform(0.3, 1.0)],
            [1, np.random.exponential(1.2), np.random.uniform(0.75, 0.95), np.random.uniform(0.4, 1.0)],
            [2, np.random.exponential(0.8), np.random.uniform(0.85, 1.0), np.random.uniform(0.5, 1.0)]
        ]
        
        # 运行主动推理算法
        assignments = distributor.run(observations)
        
        # 输出结果
        print(f"Step {step}:")
        for node in range(distributor.num_nodes):
            tasks = np.where(assignments == node)[0]
            print(f"  Node {node}: Tasks {tasks}")
        print(f"  Observations: {observations}")
        print(f"  Weights: Delay={distributor.delay_weight}, Accuracy={distributor.accuracy_weight}, Memory={distributor.memory_weight}")
        print()

# 运行模拟
if __name__ == "__main__":
    process()