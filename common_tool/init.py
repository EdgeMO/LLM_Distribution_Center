import json
import os
import sys
current_working_directory = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
#from common_tool.plot import *
from config.type import ModelInfo
        
        
def output_processor_config_init(config_path):
    with open(file = config_path, mode= 'r') as file:
        res = json.load(file)
        center_model_config = res.get('center_model_config',None)
        if center_model_config != None :
            model_directory = center_model_config.get('model_directory', None)
        output_processor_task_source_type = res.get('output_processor_task_source_type', None)
        if output_processor_task_source_type == 0:
            # local task read
            task_input_directory = res.get("local_input_directory", None)
        return center_model_config, model_directory, output_processor_task_source_type, task_input_directory
def client_server_config_init(config_path):
    with open(file = config_path, mode= 'r') as file:
        res = json.load(file)
        server_ip_config = res.get('server_ip_config',None)
        return server_ip_config
def client_model_util_init(config_path):
    with open(file = config_path, mode= 'r') as file:
        res = json.load(file)
        edge_deployment = res.get('edge_deployment',None)
        return edge_deployment
def visualization_config_init(config_path):
    with open(file = config_path, mode= 'r') as file:
        res = json.load(file)
        visualization_config = res.get('visualization',None)
        return visualization_config

def create_model_info_from_dict(model_dict, id_value):
    """
    从模型字典创建 ModelInfo 对象
    
    Args:
        model_dict: 包含模型信息的字典
        id_value: 模型ID值
        
    Returns:
        ModelInfo 对象
    """
    # 提取所需信息并进行转换
    disk_space = model_dict['model_size_gb'] * 1024  # 转换为MB
    parameter_count = model_dict['model_params'] / (10 * 9)  # 按要求处理参数量
    load_time = model_dict['load_time_ms'] / 1000  # 转换为秒
    token_processing_time = model_dict['total_time_ms'] / model_dict['total_tokens']  # 每个token的处理时间(毫秒)
    perplexity = model_dict['ppl']  # 困惑度
    model_name = model_dict['model_name']
    # 创建并返回ModelInfo对象
    return ModelInfo(
        id=id_value,
        disk_space=disk_space,
        parameter_count=parameter_count,
        load_time=load_time,
        token_processing_time=token_processing_time,
        perplexity=perplexity,
        model_name= model_name
    )

# def model_init(model_perpelexity_log_path='metrics/model_perplexity_metric.log'):
#     """ init model list from perplexity_log file"""
#     with open(model_perpelexity_log_path, 'r', encoding='utf-8') as f:
#         log_content = f.read()

#     models_data = parse_log_file(log_content)
#     model_list = []
#     for id,model_dict in enumerate(models_data,start=1):
#         model_info = create_model_info_from_dict(model_dict, id_value=id)
#         model_list.append(model_info)
#     return model_list
    pass

if __name__ == "__main__":
    #model_init()
    
    pass
    