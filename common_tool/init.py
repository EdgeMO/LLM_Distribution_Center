import json


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