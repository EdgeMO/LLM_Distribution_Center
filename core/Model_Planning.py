import os
import sys
import pandas as pd
import time
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType, DistributionType
from transmission.Center import EdgeCommunicator
from Active_Inference.Model_Planning import ActiveInferenceTaskAllocation,ModelInfo
from data_middleware.InputProcessor import InputProcessor
from data_middleware.OutputProcessor import OutputProcessor
# 负责整体节点的调度
import logging

# 在文件开头配置日志记录
logging.basicConfig(filename='core_process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Core:
    def __init__(self):
        self.num_edge_nodes = 1
        self.input_num_features = 7
        self.model_list = [
            ModelInfo(1, 0.1, 0.9, 100, 10, 1000000, 5),
            ModelInfo(2, 0.2, 0.95, 200, 20, 2000000, 10),
            ModelInfo(3, 0.3, 0.98, 300, 30, 3000000, 15)
        ]

        self.accuracy_threshold = 0.8
        self.input_generator = InputProcessor(file_path = '/mnt/data/workspace/LLM_Distribution_Center/data/example.csv')
        self.center = EdgeCommunicator(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        self.ouput_processor = OutputProcessor(config_file_path='/mnt/data/workspace/LLM_Distribution_Center/config/Running_config.json')
        self.allocator = ActiveInferenceTaskAllocation(self.num_edge_nodes, self.input_num_features,model_list=self.model_list, accuracy_threshold=self.accuracy_threshold)
    def wrapper_for_task_distribution(self,assignments, formated_input_for_algorithm, task_set):
        
        # 构建任务分配的字典
        edge_id_to_task_id_dict = {}
        for i, task_individual in enumerate(formated_input_for_algorithm):
            print(f"任务 {task_individual['id']} 分配给节点: {assignments[i]}")
            if str(assignments[i]) in edge_id_to_task_id_dict:
                edge_id_to_task_id_dict[str(assignments[i])].append(task_individual['id'])
            else:
                edge_id_to_task_id_dict[str(assignments[i])]= [task_individual['id']]
                
        # 对算法的输出进行后处理，得到消息格式
        message_list = self.input_generator.generate_offloading_message(edge_id_to_task_id_dict, task_set)
        
        return message_list
        
        
    def process(self):
        # 建立连接
        self.center.establish_connection()
        for sequence in range(10):
            # 生成当前时刻下的任务集合
            task_set = self.input_generator.generate_task_set_for_each_timestamp(4)
            formated_input_for_algorithm = []
            """            
            {
                    'id': i,
                    'features': {
                        'vocabulary_complexity': np.random.uniform(0, 1),
                        'syntactic_complexity': np.random.uniform(0, 1),
                        'task_type': np.random.uniform(0, 1),
                        'context_dependency': np.random.uniform(0, 1),
                        'ambiguity_level': np.random.uniform(0, 1),
                        'information_density': np.random.uniform(0, 1),
                        'special_symbol_ratio': np.random.uniform(0, 1)
                    }
            }
            """
            print("开始处理任务,生成任务观测量")
            for task in task_set:
                # 生成对输入的观测量,传给算法模块
                temp_input_features = {}
                task_type = task.get('task_type', 0)
                task_id = task.get('task_id', 0)
                task_token = task.get('task_token', '')
                task_reference_value = task.get('task_reference_value', '')
                temp_input_features['features'] = self.input_generator.calculate_input_features(task_token)
                temp_input_features['features']['task_type'] = task_type
                temp_input_features['id'] = task_id
                formated_input_for_algorithm.append(temp_input_features)
            
            # 算法给出任务卸载的结果
            assignments = self.allocator.process(formated_input_for_algorithm)
                    
            # 对算法的输出进行后处理，得到消息格式
            message_list = self.wrapper_for_task_distribution(assignments, formated_input_for_algorithm, task_set)
            print("给客户端发送任务")
            # 给边缘端发送文本任务
            for message in message_list:
                # 有几个 for 循环 就有几个 edge 节点
                edge_id = int(message.get('edge_id', 0))
                # 组织一下当前消息的格式
                message["mode"] = DistributionType.TASK
                message['sequence'] = sequence
                message['timestamp'] = time.time()
                self.center.send_task_message_to_client(edge_id, message)
            
            # 生成当前环境观测量，只有当所有边缘节点都返回了 sequence 之后才能进行下一步
            observation = self.center.get_overall_observation(sequence)
            print(f"第 {sequence} 个时间戳的观测结果 {observation}:")
            # 生成算法适配的观测量数据结构
            update_system_observation = []
            for key,value in observation.items():
                # 遍历每个edge节点的观测量
                temp_dict = {}
                edge_id = int(key)
                edge_observation = value['observation']
                for key,value in edge_observation.items():
                    # 为算法包装更新量
                    if 'time' in key:
                        temp_dict['accuracy'] = value
                    if 'accuracy' in key:
                        temp_dict['latency'] = value
                    if 'disk' in key:
                        temp_dict['avg_throughput'] = value
                update_system_observation.append(temp_dict)
            
            self.allocator.feedback(update_system_observation)
            
            
        pass
if __name__ == "__main__":
    core = Core()
    core.process()