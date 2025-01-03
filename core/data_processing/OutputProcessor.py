import json
# 基于算法的输出 (模型和任务)，生成对应的字节流
class OutputProcessor:
    """
    generate output data and transmission config well formated for networking transmission 
    use queue for task and model distribution
    """
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def process(self, switch):
        if self.switch == 0:
            return "task transmission"
        elif self.switch == 1:
            return "model transmission"
        return 