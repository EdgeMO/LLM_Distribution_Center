import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np
import time

from algorithms.base_allocator import BaseAllocator
# 配置TensorFlow以更好地利用多核CPU
physical_devices = tf.config.list_physical_devices('CPU')
if len(physical_devices) > 0:
    tf.config.threading.set_intra_op_parallelism_threads(8)  # 设置操作内并行线程数
    tf.config.threading.set_inter_op_parallelism_threads(8)  # 设置操作间并行线程数
    print(f"TensorFlow configured to use up to 8 threads for parallel operations")
class DQNTaskAllocator(BaseAllocator):
    def __init__(self, edge_nodes, models, network_env, feature_dim, 
                 memory_size=2000, batch_size=32):
        """
        初始化基于DQN的任务分配器
        
        参数:
        - edge_nodes: 边缘节点列表
        - models: 可用模型列表
        - network_env: 网络环境
        - feature_dim: 任务特征维度
        - memory_size: 经验回放缓冲区大小
        - batch_size: 训练批次大小
        """
        super().__init__(edge_nodes, models, network_env)
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # DQN参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # 创建经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        
        # 动作空间 - (节点,模型)对的所有可能组合
        self.action_space = [(n.node_id, m.model_id) for n in edge_nodes for m in models]
        self.action_space_size = len(self.action_space)
        
        # 创建DQN模型 - 输出为每个(节点,模型)对的Q值
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self.fitted_scaler = False
    
    def _build_model(self):
        """构建DQN模型"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.feature_dim,)))  # 使用 Input 层作为第一层
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_space_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """更新目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state, action, reward, next_state):
        """将经验存储到回放缓冲区"""
        # 确保动作是元组而不是NumPy数组
        if isinstance(action, np.ndarray):
            action = tuple(action)
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state, current_bandwidth):
        """选择动作 - 返回(节点ID,模型ID)对"""
        # 探索 vs 利用
        if np.random.random() <= self.epsilon:
            return random.choice(self.action_space)
        
        # 预测每个动作的Q值
        act_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # 调整Q值以考虑带宽和模型部署状态
        adjusted_values = act_values.copy()
        for i, (node_id, model_id) in enumerate(self.action_space):
            # 如果模型未部署，考虑传输时间的影响
            if model_id not in self.model_deployment_cache.get(node_id, set()):
                model = next((m for m in self.models if m.model_id == model_id), None)
                if model:
                    # 计算传输时间
                    bandwidth = self.network_env.get_bandwidth(node_id)
                    transfer_time = model.size_mb / bandwidth
                    
                    # 传输时间越长，Q值越低
                    adjusted_values[i] -= transfer_time * 10
        
        # 返回动作空间中对应索引的动作
        best_action_index = np.argmax(adjusted_values)
        return self.action_space[best_action_index]
    
    def replay(self):
        """从记忆中批量学习"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([m[0] for m in minibatch])
        actions = [m[1] for m in minibatch]  # 不转换为NumPy数组
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        
        # 将动作索引转换为动作空间索引
        action_indices = []
        for a in actions:
            # 查找动作在动作空间中的索引
            for i, action in enumerate(self.action_space):
                # 将元组动作与动作空间中的元组进行比较
                if isinstance(a, tuple) and a[0] == action[0] and a[1] == action[1]:
                    action_indices.append(i)
                    break
        
        action_indices = np.array(action_indices)
        
        # 预测当前状态和下一状态的Q值
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # 更新目标Q值
        for i in range(self.batch_size):
            target[i][action_indices[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # 训练模型
        self.model.fit(states, target, epochs=1, verbose=0)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
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
            # 缩放特征
            if not self.fitted_scaler:
                self.scaler.fit([task_features])
                self.fitted_scaler = True
            
            scaled_features = self.scaler.transform([task_features])[0]
            
            # 选择动作 (节点,模型)
            node_id, model_id = self.act(scaled_features, current_bandwidth)
            
            # 计算传输时间
            transfer_time = self._calculate_model_transfer_time(model_id, node_id)
            
            # 尝试部署模型
            deploy_success = self._deploy_model(model_id, node_id)
            
            if deploy_success:
                allocations.append({
                    "task_features": task_features,
                    "node_id": node_id,
                    "model_id": model_id,
                    "transfer_time": transfer_time
                })
            else:
                # 如果部署失败，选择已部署的模型
                deployed_models = list(self.model_deployment_cache.get(node_id, set()))
                if deployed_models:
                    model_id = deployed_models[0]  # 简单选择第一个
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node_id,
                        "model_id": model_id,
                        "transfer_time": 0  # 已部署，无需传输
                    })
                else:
                    # 如果节点没有模型，选择最小的模型
                    smallest_model = min(self.models, key=lambda m: m.size_mb)
                    transfer_time = self._calculate_model_transfer_time(smallest_model.model_id, node_id)
                    self._deploy_model(smallest_model.model_id, node_id)
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node_id,
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
            task_features = alloc["task_features"]
            node_id = alloc["node_id"]
            model_id = alloc["model_id"]
            transfer_time = alloc["transfer_time"]
            accuracy = alloc["result"]["accuracy"]
            latency = alloc["result"]["latency"]
            resource = alloc["result"]["resource_usage"]
            
            # 考虑传输时间
            total_latency_with_transfer = latency + transfer_time * 1000  # 转换为ms
            
            # 缩放特征
            if not self.fitted_scaler:
                self.scaler.fit([task_features])
                self.fitted_scaler = True
            
            scaled_features = self.scaler.transform([task_features])[0]
            
            # 计算奖励 - 高准确率、低延迟、低资源使用为佳
            reward = accuracy - 0.01 * total_latency_with_transfer - 0.5 * resource
            
            # 存储经验 - 简化处理，使用相同状态作为下一状态
            action = (node_id, model_id)
            self.memorize(scaled_features, action, reward, scaled_features)
            
            # 累计性能指标
            total_accuracy += accuracy
            total_latency += total_latency_with_transfer
            total_resource += resource
            count += 1
        
        # 训练模型
        self.replay()
        
        # 定期更新目标网络
        if len(self.performance_history) % 10 == 0:
            self.update_target_model()
        
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