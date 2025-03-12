import os
import sys
import time
current_working_directory = os.getcwd()
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import TaskType, DistributionType
from transmission.Center import EdgeCommunicator
from Active_Inference.Model_Planning import ActiveInferenceTaskAllocation,ModelInfo
from data_middleware.InputProcessor import InputProcessor
from data_middleware.OutputProcessor import OutputProcessor
# 负责整体节点的调度
import logging
from config.type import ModelInfo
# 在文件开头配置日志记录
logging.basicConfig(filename='core_process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Core:
    def __init__(self):
        self.num_edge_nodes = 1
        self.input_num_features = 7

        self.input_generator = InputProcessor(file_path = 'data/example.csv')
        self.center = EdgeCommunicator(config_file_path='config/Running_config.json')
        self.ouput_processor = OutputProcessor(config_file_path='config/Running_config.json')
        self.allocator = ActiveInferenceTaskAllocation(self.num_edge_nodes, self.input_num_features)
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
        for sequence in range(300):
            # 生成当前时刻下的任务集合
            task_set = self.input_generator.generate_task_set_for_each_timestamp(2)
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
                temp_input_features['features'] = self.input_generator.calculate_input_features(task_token)
                temp_input_features['features']['task_type'] = task_type
                temp_input_features['id'] = task_id
                for key, value in temp_input_features['features'].items():
                    try:
                        temp_input_features['features'][key] = float(value)
                    except (ValueError, TypeError):
                        print(f"警告: 特征 '{key}' 的值 '{value}' 不是数值，将使用默认值 0")
                        temp_input_features['features'][key] = 0.0
                # ID 可以保持为字符串，因为它不会用作特征
                temp_input_features['id'] = task_id
                formated_input_for_algorithm.append(temp_input_features)
            logging.info(f"sequence {sequence} formated_input_for_algorithm = {formated_input_for_algorithm}")
            
            # 定义预期的特征名称和顺序
            feature_names = [
                'vocabulary_complexity', 'syntactic_complexity', 'context_dependency', 
                'ambiguity_level', 'information_density', 'special_symbol_ratio', 'task_type'
            ]
            
            # 将字典列表转换为特征数组
            feature_arrays = []
            for task_dict in formated_input_for_algorithm:
                feature_values = []
                for feature_name in feature_names:
                    # 获取特征值，默认为0.0
                    value = task_dict['features'].get(feature_name, 0.0)
                    # 已经确保了值是浮点数，但再次转换以确保安全
                    feature_values.append(float(value))
                
                feature_arrays.append(feature_values)
            
            # 转换为 NumPy 数组
            task_features_array = np.array(feature_arrays, dtype=float)
            
            # 打印调试信息
            print(f"转换后的特征数组形状: {task_features_array.shape}, 类型: {task_features_array.dtype}")
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
                message["mode"] = DistributionType.TASK.value
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
                    if 'throughput' in key:
                        temp_dict['avg_throughput'] = value
                update_system_observation.append(temp_dict)
            
            self.allocator.feedback(update_system_observation)
            
            
if __name__ == "__main__":
    core = Core()
    core.process()