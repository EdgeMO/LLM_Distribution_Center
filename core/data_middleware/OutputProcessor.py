import os
import json
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common_tool.init import *

# 基于算法的输出 (模型和任务)，生成对应的待传输设备
class OutputProcessor:
    """
    generate output data and transmission config well formated for networking transmission 
    use queue for task and model distribution
    """
    def __init__(self, config_file_path):
        self.center_model_config, self.model_directory, self.output_processor_task_source_type, self.task_input_directory = output_processor_config_init(config_file_path)
        print("output processor init sucess")
    
    

        
    def process(self, output_type_switch, task_id_list = None, model_id_list = None, output_processor_task_source_type = 0):
        """
        
        for each time stamp, generate well formated output for transmission

        Args:
            output_type_switch (_type_): task or model output.
            task_id_list (_type_, optional): list(int). Defaults to None.
            model_id_list (_type_, optional): list(int). Defaults to None.
            output_processor_task_source_type (int, optional): local read or remote read. Defaults to 0.

        Returns:
            _type_: 
            0 : list(str) : task token list 
            1 : list(str) : model directory
            
        """
        output = {}
        output['type'] = output_type_switch
        if output_type_switch == 0:
            # task output generation
            if output_processor_task_source_type == 0: # local_read
                output_data = self.generate_task_list_from_task_id_list(task_id_list = task_id_list)
                
        if output_type_switch == 1:
            # model output generation
            output_data = self.generate_model_directory_from_model_id(model_id_list)
        output['data'] = output_data
        return output
    
            
            
            
            
    def generate_model_directory_from_model_id(self, model_id_list ):
        res = []
        model_id_list = [0,1]
        model_params = self.center_model_config.get('model_parameters', None)
        base_model_path = self.center_model_config.get('model_directory', None)
        for model_id in model_id_list:
            temp_path = ""
            model_name = model_params[str(model_id)].get('model_name', None)
            temp_path = base_model_path + f"/{model_name}" + f"/{model_name}.gguf"
            res.append(temp_path)
        return res
    
    def generate_formated_algorithm_output_for_task_offloading(self,algorithm_output):
        
        pass
    
    def generate_task_list_from_task_id_list(self, task_id_list):
        res = []
        pass
if __name__ == "__main__":
    ouput_processor = OutputProcessor()
    ouput_processor.process(output_type_switch = 1)