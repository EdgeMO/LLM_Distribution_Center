class EdgeNode:
    def __init__(self, node_id, compute_power, memory_capacity, initial_models=None):
        """
        初始化边缘节点
        
        参数:
        - node_id: 节点唯一标识符
        - compute_power: 计算能力 (0-1)
        - memory_capacity: 内存容量 (MB)
        - initial_models: 初始预装的模型列表
        """
        self.node_id = node_id
        self.compute_power = compute_power
        self.memory_capacity = memory_capacity
        self.deployed_models = {} if initial_models is None else {model.model_id: model for model in initial_models}
        self.used_memory = sum(model.size_mb for model in self.deployed_models.values())
        
        # 任务执行历史
        self.task_history = []
        
        # 当前负载状态
        self.current_load = 0.0
    
    def can_deploy_model(self, model):
        """检查是否可以部署新模型"""
        if model.model_id in self.deployed_models:
            return True  # 模型已部署
        
        # 检查剩余内存是否足够
        return self.used_memory + model.size_mb <= self.memory_capacity
    
    def deploy_model(self, model):
        """部署模型到节点"""
        if model.model_id in self.deployed_models:
            return True  # 模型已存在
        
        if not self.can_deploy_model(model):
            return False  # 内存不足
        
        self.deployed_models[model.model_id] = model
        self.used_memory += model.size_mb
        return True
    
    def remove_model(self, model_id):
        """移除模型"""
        if model_id in self.deployed_models:
            self.used_memory -= self.deployed_models[model_id].size_mb
            del self.deployed_models[model_id]
            return True
        return False
    
    def execute_task(self, task, model):
        """执行任务并返回性能指标"""
        if model.model_id not in self.deployed_models:
            raise ValueError(f"Model {model.model_id} not deployed on node {self.node_id}")
        
        # 考虑节点计算能力对推理时间的影响
        effective_inference_time = model.inference_time / self.compute_power
        
        # 考虑当前负载对推理时间的影响
        load_factor = 1 + self.current_load
        effective_inference_time *= load_factor
        
        # 更新负载
        self.current_load = min(1.0, self.current_load + model.resource_usage * 0.1)
        
        # 记录任务执行
        self.task_history.append((task, model.model_id))
        
        # 返回执行结果
        return {
            "accuracy": model.accuracy,
            "latency": effective_inference_time,
            "resource_usage": model.resource_usage
        }
    
    def update_load(self, decay_factor=0.95):
        """负载自然衰减"""
        self.current_load *= decay_factor