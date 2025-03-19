class Model:
    def __init__(self, model_id, accuracy, inference_time, resource_usage, size_mb):
        """
        初始化模型
        
        参数:
        - model_id: 模型唯一标识符
        - accuracy: 模型准确率 (0-1)
        - inference_time: 推理时间 (ms)
        - resource_usage: 资源使用率 (0-1)
        - size_mb: 模型大小 (MB)
        """
        self.model_id = model_id
        self.accuracy = accuracy
        self.inference_time = inference_time
        self.resource_usage = resource_usage
        self.size_mb = size_mb
    
    def __repr__(self):
        return f"Model({self.model_id}, acc={self.accuracy:.2f}, time={self.inference_time}ms, res={self.resource_usage:.2f}, size={self.size_mb}MB)"