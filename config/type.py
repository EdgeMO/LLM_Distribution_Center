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
    