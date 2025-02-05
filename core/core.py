import os
import sys
import pandas as pd

current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType
from transmission.Center import EdgeCommunicator
from task_offloading.Task_active_inference import  ActiveInferenceTaskDistributor
from data_middleware.InputProcessor import InputProcessor
from data_middleware.OutputProcessor import OutputProcessor
# 负责整体节点的调度


class Core:
    def __init__(self):
        self.input_generator = InputProcessor(file_path = '/mnt/data/workspace/LLM_Distribution_Center/data/example.csv')
        self.task_offloading_generator = ActiveInferenceTaskDistributor(num_nodes=3, num_tasks=10)
        self.center = EdgeCommunicator(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        self.ouput_processor = OutputProcessor(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        pass
    def process(self):
        # 建立连接
        self.center.establish_connection()
        for i in range(10):
            # 生成当前时刻下的任务集合
            task_set = self.input_generator.generate_task_set_for_each_timestamp(10)
            
            # 生成当前环境观测量
            observation = self.center.get_observation()
        
            # 基于观测量，运行算法，生成任务下发的决策
            algorithem_output = self.task_offloading_generator.run(observation, task_set)
            
            # 基于任务下发的决策，调用 output_processor 生成对应的消息格式
            task_message = self.ouput_processor.generate_formated_algorithm_output_for_task_offloading(algorithem_output)
            # 调用 center 发送消息
            self.center.send_task_message_to_client(task_message)
            # 统计相关指标，并且更新下本地的环境观测量
            
        pass
if __name__ == "__main__":
    core = Core()
    core.process()