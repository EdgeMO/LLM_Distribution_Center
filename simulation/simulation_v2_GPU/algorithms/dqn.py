import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler

from algorithms.base_allocator import BaseAllocator

# 配置 TensorFlow 以使用 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"找到 {len(physical_devices)} 个 GPU:")
    for device in physical_devices:
        print(f"  名称: {device.name}, 类型: {device.device_type}")
    # 配置 GPU 内存增长
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用 GPU 内存动态增长")
        
        # 启用混合精度训练以提高性能
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("已启用混合精度训练")
        
        # 启用XLA加速
        tf.config.optimizer.set_jit(True)
        print("已启用XLA编译加速")
    except RuntimeError as e:
        print(f"GPU 配置错误: {e}")
else:
    print("未找到 GPU，将使用 CPU 运行")

class DQNTaskAllocator(BaseAllocator):
    def __init__(self, edge_nodes, models, network_env, feature_dim, 
                 memory_size=2000, batch_size=64):
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
        
        # 优化 GPU 使用
        self.using_gpu = len(physical_devices) > 0
        if self.using_gpu:
            # 使用分布式策略以更好地利用GPU
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"使用 {self.strategy.num_replicas_in_sync} 个设备进行分布式训练")
        else:
            self.strategy = None
        
        # 创建DQN模型 - 输出为每个(节点,模型)对的Q值
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self.fitted_scaler = False
        
        # 批量预测缓存
        self.state_batch_size = 128  # 批量预测的大小
    
    def _build_model(self):
        """构建DQN模型"""
        if self.using_gpu and self.strategy:
            with self.strategy.scope():
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.feature_dim,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(self.action_space_size, activation='linear', dtype='float32')
                ])
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                model.compile(loss='mse', optimizer=optimizer, jit_compile=True)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.feature_dim,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.action_space_size, activation='linear')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(loss='mse', optimizer=optimizer)
        
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
        
        # 预测每个动作的Q值 - 使用GPU加速
        if self.using_gpu:
            # 转换为TensorFlow张量以使用GPU
            state_tensor = tf.convert_to_tensor(np.array([state]), dtype=tf.float32)
            act_values = self.model(state_tensor, training=False).numpy()[0]
        else:
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
        
        # 增大批次大小以更好地利用GPU
        actual_batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, actual_batch_size)
        
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
        
        if self.using_gpu:
            # 使用TensorFlow张量和GPU进行批量计算
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
            
            # 预测当前状态和下一状态的Q值
            target = self.model(states_tensor, training=False).numpy()
            target_next = self.target_model(next_states_tensor, training=False).numpy()
            
            # 更新目标Q值
            for i in range(actual_batch_size):
                target[i][action_indices[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
            
            # 转换为TensorFlow张量
            target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
            
            # 训练模型
            self.model.fit(states_tensor, target_tensor, epochs=1, verbose=0, batch_size=actual_batch_size)
        else:
            # CPU版本
            target = self.model.predict(states, verbose=0)
            target_next = self.target_model.predict(next_states, verbose=0)
            
            # 更新目标Q值
            for i in range(actual_batch_size):
                target[i][action_indices[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
            
            # 训练模型
            self.model.fit(states, target, epochs=1, verbose=0, batch_size=actual_batch_size)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def predict_batch(self, states):
        """批量预测状态的Q值"""
        if self.using_gpu:
            # 使用GPU进行批量预测
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            return self.model(states_tensor, training=False).numpy()
        else:
            return self.model.predict(states, verbose=0)
    
    def allocate_tasks(self, task_batch, current_bandwidth):
        """分配任务到边缘节点并选择模型"""
        # 更新网络环境带宽
        for node in self.edge_nodes:
            self.network_env.set_bandwidth(node.node_id, current_bandwidth)
        
        # 如果使用GPU，批量预测所有任务的Q值
        if self.using_gpu and len(task_batch) > 1:
            return self._allocate_tasks_batch(task_batch, current_bandwidth)
        else:
            return self._allocate_tasks_individual(task_batch, current_bandwidth)
    
    def _allocate_tasks_batch(self, task_batch, current_bandwidth):
        """使用GPU批量分配任务"""
        allocations = []
        
        # 构建需要使用DQN决策的任务列表
        exploit_tasks = []
        exploit_indices = []
        
        for i, task_features in enumerate(task_batch):
            # 探索 vs 利用
            if np.random.random() <= self.epsilon:
                # 随机选择节点和模型
                node = np.random.choice(self.edge_nodes)
                model = np.random.choice(self.models)
                
                # 尝试部署模型
                transfer_time = self._calculate_model_transfer_time(model.model_id, node.node_id)
                deploy_success = self._deploy_model(model.model_id, node.node_id)
                
                if deploy_success:
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node.node_id,
                        "model_id": model.model_id,
                        "transfer_time": transfer_time
                    })
            else:
                # 收集需要使用DQN决策的任务
                exploit_tasks.append(task_features)
                exploit_indices.append(i)
        
        if exploit_tasks:
            # 批量预测Q值
            states = np.array(exploit_tasks)
            q_values = self.predict_batch(states)
            
            # 为每个任务选择最佳动作
            for idx, task_idx in enumerate(exploit_indices):
                task_features = task_batch[task_idx]
                
                # 调整Q值以考虑带宽和模型部署状态
                adjusted_values = q_values[idx].copy()
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
                
                # 选择最佳动作
                best_action_index = np.argmax(adjusted_values)
                node_id, model_id = self.action_space[best_action_index]
                
                # 尝试部署模型
                transfer_time = self._calculate_model_transfer_time(model_id, node_id)
                deploy_success = self._deploy_model(model_id, node_id)
                
                if deploy_success:
                    allocations.append({
                        "task_features": task_features,
                        "node_id": node_id,
                        "model_id": model_id,
                        "transfer_time": transfer_time
                    })
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return allocations
    
    def _allocate_tasks_individual(self, task_batch, current_bandwidth):
        """逐个分配任务（原始方法）"""
        allocations = []
        
        for task_features in task_batch:
            # 选择动作 (节点ID, 模型ID)
            node_id, model_id = self.act(task_features, current_bandwidth)
            
            # 计算传输时间
            transfer_time = self._calculate_model_transfer_time(model_id, node_id)
            
            # 尝试部署模型
            deploy_success = self._deploy_model(model_id, node_id)
            
            if deploy_success:
                # 记录分配
                allocations.append({
                    "task_features": task_features,
                    "node_id": node_id,
                    "model_id": model_id,
                    "transfer_time": transfer_time
                })
                
                # 更新经验回放
                state = task_features
                action = (node_id, model_id)
                
                # 奖励将在update_feedback中计算
                next_state = task_features  # 简化：使用相同特征作为下一状态
                
                # 存储经验
                self.memorize(state, action, 0, next_state)  # 初始奖励设为0
        
        return allocations
    
    def update_feedback(self, executed_allocations):
        """更新来自边缘节点的反馈"""
        if not executed_allocations:
            return
        
        # 收集性能数据
        performance_sum = 0
        
        for alloc in executed_allocations:
            task_features = alloc["task_features"]
            node_id = alloc["node_id"]
            model_id = alloc["model_id"]
            transfer_time = alloc["transfer_time"]
            
            # 计算性能指标
            accuracy = alloc["result"]["accuracy"]
            latency = alloc["result"]["latency"] + transfer_time * 1000  # 转换为ms
            resource = alloc["result"]["resource_usage"]
            
            # 计算加权性能得分
            # 高准确率好，低延迟好，低资源使用好
            performance = accuracy - 0.01 * latency - 0.5 * resource
            performance_sum += performance
            
            # 更新经验回放
            state = task_features
            action = (node_id, model_id)
            reward = performance
            next_state = task_features  # 简化：使用相同特征作为下一状态
            
            # 更新奖励
            for i, experience in enumerate(self.memory):
                if np.array_equal(experience[0], state) and experience[1] == action:
                    self.memory[i] = (state, action, reward, next_state)
                    break
        
        # 从经验回放中学习
        self.replay()
        
        # 定期更新目标网络
        if hasattr(self, 'update_counter'):
            self.update_counter += 1
        else:
            self.update_counter = 0
            
        if self.update_counter % 10 == 0:
            self.update_target_model()
        
        # 记录性能
        avg_performance = performance_sum / len(executed_allocations)
        self.performance_history.append(avg_performance)
        
        # 检查收敛性
        if len(self.performance_history) >= 50:
            recent_avg = np.mean(self.performance_history[-50:])
            if not self.converged and len(self.performance_history) >= 100:
                prev_avg = np.mean(self.performance_history[-100:-50])
                if abs(recent_avg - prev_avg) < 0.01:
                    self.converged = True
                    self.convergence_time = time.time() - self.start_time