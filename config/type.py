from enum import Enum
import json
class TaskType(Enum):
    TC = 0
    NER = 1
    QA = 2
    TL = 3
    SG = 4

class DistributionType(Enum):
    TASK = 0  # 表示是任务下发
    MODEL = 1 # 表示是模型下发
    
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return json.JSONEncoder.default(self, obj)

class ModelInfo:
    def __init__(self, id, disk_space, parameter_count, load_time, token_processing_time, perplexity,model_name):
        """
        模型信息类
        
        Args:
            id: 模型ID
            disk_space: 模型占用的磁盘空间(MB)
            parameter_count: 模型参数量(百万)
            load_time: 模型加载时间(秒)
            token_processing_time: 每个token的处理时间(毫秒)
            perplexity: 模型困惑度(越低越好)
        """
        self.id = id
        self.disk_space = disk_space
        self.parameter_count = parameter_count
        self.load_time = load_time
        self.token_processing_time = token_processing_time
        self.perplexity = perplexity
        self.model_name = model_name

    