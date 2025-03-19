class NetworkEnvironment:
    def __init__(self, default_bandwidth=10.0):
        """
        初始化网络环境
        
        参数:
        - default_bandwidth: 默认带宽 (MB/s)
        """
        self.default_bandwidth = default_bandwidth
        self.node_bandwidths = {}  # 中心节点到各边缘节点的带宽
    
    def set_bandwidth(self, node_id, bandwidth):
        """设置到特定节点的带宽"""
        self.node_bandwidths[node_id] = bandwidth
    
    def get_bandwidth(self, node_id):
        """获取到特定节点的带宽"""
        return self.node_bandwidths.get(node_id, self.default_bandwidth)
    
    def calculate_transfer_time(self, model, node_id):
        """计算模型传输时间 (秒)"""
        bandwidth = self.get_bandwidth(node_id)
        return model.size_mb / bandwidth