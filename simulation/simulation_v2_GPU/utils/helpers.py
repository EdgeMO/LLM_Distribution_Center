import numpy as np
import time

def measure_execution_time(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def calculate_weighted_score(accuracy, latency, resource_usage, weights=(1.0, 0.01, 0.5)):
    """
    计算加权综合得分
    
    参数:
    - accuracy: 准确率 (越高越好)
    - latency: 延迟 (越低越好)
    - resource_usage: 资源使用 (越低越好)
    - weights: 各指标的权重 (准确率权重, 延迟权重, 资源使用权重)
    """
    return weights[0] * accuracy - weights[1] * latency - weights[2] * resource_usage

def smooth_data(data, window_size=10):
    """
    使用滑动平均平滑数据
    
    参数:
    - data: 输入数据列表
    - window_size: 滑动窗口大小
    """
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed