import os
import sys
import pandas as pd

current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType
from transmission.Center import EdgeCommunicator
from Active_Inference.Task_Model_active_inference import ActiveInference,ModelInfo
from data_middleware.InputProcessor import InputProcessor
from data_middleware.OutputProcessor import OutputProcessor
# 负责整体节点的调度


class Core:
    def __init__(self):
        model_infos = [
            ModelInfo(1, 10, 0.9, 200, 50, 1000000, 100),
            ModelInfo(2, 5, 0.85, 150, 40, 500000, 80),
            ModelInfo(3, 15, 0.95, 250, 60, 2000000, 120)
        ]
        self.input_generator = InputProcessor(file_path = '/mnt/data/workspace/LLM_Distribution_Center/data/example.csv')
        self.center = EdgeCommunicator(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        self.ouput_processor = OutputProcessor(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        self.allocator = ActiveInference(num_edge_nodes = 3, num_features = 7, model_infos = model_infos, accuracy_threshold=0.9)
        pass
    def process(self):
        # 建立连接
        self.center.establish_connection()
        for i in range(10):
            # 生成当前时刻下的任务集合
            # single momvement
            task_set = self.input_generator.generate_task_set_for_each_timestamp(10)
            formated_task_set = []
            for task in task_set:
                # 生成对输入的观测量,传给算法模块
                task_type = task.get('task_type', 0)
                task_id = task.get('task_id', 0)
                task_token = task.get('task_token', '')
                task_reference_value = task.get('task_reference_value', '')
                temp_input_features = self.input_generator.calculate_input_features(task_token)
                temp_input_features['task_type'] = task_type
                temp_input_features['task_id'] = task_id
                formated_task_set.append(temp_input_features)
            
            
            pass
                
                
                
                
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