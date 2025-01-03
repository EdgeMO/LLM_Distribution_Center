# 基于本地数据，生成算法适配的随机批输入, 并且提供数据查询接口，返回基于 task id 的具体数据

class InputProcessor:
    """
    regulate the input data from different sources and generate output data compatible for the algorithm input
    """
    def __init__(self, data):
        self.data = data

    def process(self):
        return self.data