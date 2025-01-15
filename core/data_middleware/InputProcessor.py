# 基于本地数据，生成算法适配的随机批输入, 并且提供数据查询接口，返回基于 task id 的具体数据

class InputProcessor:
    """
    regulate the input data from different sources and generate output data compatible for the algorithm input
    """
    def __init__(self, input_type, file_path = None):
        """
        Args:
            input_type (_type_): indicates the type of the input data  0 : local_file, 1 : tcp transmission
            file_path (_type_): local file path, activates when input_type is 0
            
            1 : TBD
        """
        self.input_type = input_type
        self.file_path = file_path
        self.task_set_size = 10

    def generate_task_set_for_each_timestamp(self, task_num):
        """
        
        randomly select task_num tasks from data iterator

        Args:
            task_num (_type_): task_set size
        """
        task_set = []
        
        return task_set
    
    def similarity_assemble_for_each_timestamp(self, task_set):
        """
        calculate the similarity of same batch ,  used for observation space
        """
        
        
        pass
    def process(self):
        """
        generate input for each timestamp
        """
        res = []
        input_set = self.generate_task_set_for_each_timestamp(task_num = self.task_set_size)
        res = self.similarity_assemble_for_each_timestamp(task_set= input_set)
        
        
        return res
        