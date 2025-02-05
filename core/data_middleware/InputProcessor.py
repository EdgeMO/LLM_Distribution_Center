# 基于本地数据，生成算法适配的随机批输入, 并且提供数据查询接口，返回基于 task id 的具体数据

import os
import sys
import pandas as pd

current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType
class InputProcessor:
    """
    regulate the input data from different sources and generate output data compatible for the algorithm input
    """
    def __init__(self, file_path = None):
        """
        Args:
            input_type (_type_): indicates the type of the input data  0 : local_file, 1 : tcp transmission
            file_path (_type_): local file path, activates when input_type is 0
            
            1 : TBD
        """
        self.file_path = file_path
        self.task_set_size = 10
        self.random_state = 930
        self.df = pd.read_csv(file_path)
    def generate_task_set_for_each_timestamp(self, task_num):
        """
        
        randomly select task_num tasks from data iterator

        Args:
            task_num (_type_): task_set size
        """
        sampled_task_set = self.df.sample(n = task_num, random_state = self.random_state)
        task_dicts = sampled_task_set.to_dict(orient = 'records')
                
        return task_dicts
    
    def generate_query_word_from_task_type(self, TASK_TYPE):
        """
        generate query word from task type, fixed query mode, without considering the output format
        """
        TASK_TYPE  = TaskType(TASK_TYPE)
        if TASK_TYPE == TaskType.TC:
            res = f"Please classify the sentiment of the following text into one of the categories: sad, happy, love, angry, fear, surprise \n\n Text:"
            return res
        elif TASK_TYPE == TaskType.NER:
            res = f"Please identify the named entities in the following text. Classify entities into categories such as Person, Location, Organization, Miscellaneous \n\n Text:"
            return res
        elif TASK_TYPE == TaskType.QA:
            res = f"please answer the following question based on the text provided \n\n Question:"
            return res
        elif TASK_TYPE == TaskType.TL:
            res = "please translate the following text into English \n\n text:"
            return res
        elif TASK_TYPE == TaskType.SG:
            res = "Please summarize the following text \n\n Text:"
            return res
        else:
            return ""


if __name__ == "__main__":
    input = InputProcessor('/mnt/data/workspace/LLM_Distribution_Center/data/example.csv')
    res = input.generate_task_set_for_each_timestamp(task_num = 10)
    pass