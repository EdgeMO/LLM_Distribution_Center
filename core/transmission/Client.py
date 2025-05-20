
import threading
import time
import datetime
import json
import shutil
import socket
import os 
import glob
import sys
import struct
import csv
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from queue import Queue
from common_tool.cmd import CMD
from common_tool.mertics import Metrics
from data_middleware.InputProcessor import InputProcessor
import logging

# 在文件开头配置日志记录
logging.basicConfig(filename='client_process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



from common_tool.init import *
from config.type import DistributionType, TaskType, EnumEncoder

class Client_Connection:
    def __init__(self, config_path):
        # tcp connection
        self.server_config = client_server_config_init(config_path)
        self.server_ip = self.server_config.get('server_ip', None)
        self.server_port = self.server_config.get('server_port', None)
        
        # edge dployment
        self.deplotment_config = client_model_util_init(config_path)
        self.llama_cli_path = self.deplotment_config.get('llama_cli_path','')
        self.models_save_directory = self.generate_models_path(self.deplotment_config.get('model_save_path',''))
        # model transmitting
        os.makedirs(self.models_save_directory, exist_ok=True)
        self.chunk_size = 8192  # Matching the server's chunk size
        
        self.cmd_operator = CMD()
        self.input_processor = InputProcessor()
        self.metrics_operator = Metrics()
        
        #
        self.task_queue = Queue()
        self.sequence_queue = Queue()
        
        # global metrics
        self.client_sum_batch_num = 0
        self.client_sum_task_num = 0
        self.client_sum_batch_accuravy_score = 0
        self.client_sum_batch_time_consumption = 0
        self.client_sum_batch_throughput_score = 0
        self.client_task_num_batch = []
        self.record_metrics_file_path = 'metrics/client/client_metrics.csv'
        
        self.data_need_to_record = {}
        
        # visualization trigger
        self.visualization_flag = visualization_config_init(config_path)
        self.visualization_trigger = self.visualization_flag.get('trigger',False)
        
        
    def generate_models_path(self, relative_directory):
        current_dir = os.getcwd()
        absolute_path = os.path.join(current_dir, relative_directory)
        absolute_path = os.path.normpath(absolute_path)  # 规范化路径
        logging.info(f'model save path = {absolute_path}')
        return absolute_path
    def generate_llama_cli_path(self):
        
        
        return self.llama_cli_path
    
    def generate_model_to_be_used_path(self):
        directory = os.path.abspath(self.models_save_directory)
        gguf_files = glob.glob(os.path.join(directory, "*.gguf"))
        if type(gguf_files == list):
            return gguf_files[0],gguf_files
        return gguf_files

    def single_process(self):
        """
        pick one task set from queue and process it , get related metrics and current status
        {'mode': 0, 'tasks': [{'task_id': 682, 'task_type': 1, 'task_token': 'Please identify the named entities in the following text. Classify entities into categories such as Person, Location, Organization, Miscellaneous \n\n Text:Widodo', 'reference_value': 'PER'}, {'task_id': 583, 'task_type': 1, 'task_token': 'Please identify the named entities in the following text. Classify entities into categories such as Person, Location, Organization, Miscellaneous \n\n Text:,', 'reference_value': 'MISC'}], 'receive_timestamp': 1739868580.523756}
        """

        # get task batch from task_queue
        origin_info = self.task_queue.get()
        logging.info(f"origin_info = {origin_info}")
        task_list = origin_info.get('tasks',{}).get('task_set',[])
        logging.info(f"task_list = {task_list}")
        num_task_list = len(task_list)
        # ui
        task_id_list = []
        
        # metrics calculation
        # received timestamp for single batch
        received_timestamp_for_single_batch = origin_info.get('receive_timestamp')
        
        single_batch_sum_task_accuravy_score = 0

        #logging.info(f"task_list =  {task_list}")
        
        # get model path and llama_cli_path  for cmd running      
        current_using_model_path, all_model_path = self.generate_model_to_be_used_path()
        llama_cli_path = self.generate_llama_cli_path()
        
        total_cpu_usage_percentage = 0
        total_mem_usage_percentage = 0
        
        for task in task_list:
            task_id_list.append(task.get('task_id',0))
            # cmd进行任务处理
            token_type = task.get('task_type')
            token = task.get('task_token')
            reference_value = task.get('reference_value')
            query_prefix = self.input_processor.generate_query_word_from_task_type(token_type)
            # 获取单次token_prediction的处理结果
            token_prediction, resource_used_status = self.cmd_operator.run_task_process_cmd(query_prefix=query_prefix, query_word=token, llama_cli_path = llama_cli_path, model_path=current_using_model_path,enable_visualization=self.visualization_trigger)
            
            # single task accuracy
            accuracy_score = self.metrics_operator.process(type = token_type, prediction = token_prediction, reference = reference_value)
            single_batch_sum_task_accuravy_score += accuracy_score
            
            # single task resource status
            total_cpu_usage_percentage += resource_used_status.get('cpu',0)
            total_mem_usage_percentage += resource_used_status.get('memory',0)
            
            logging.info(f"token = {token}, token_prediction = {token_prediction} accuracy_score = {accuracy_score}")
        
        
        # res metrics calculation
        finished_single_batch_timestamp = datetime.datetime.now().timestamp()
        # time consumption for single task batch
        single_batch_time_consumption = (finished_single_batch_timestamp - received_timestamp_for_single_batch) * 1000
        
        
        
        average_local_processing_time_per_task = (single_batch_time_consumption / num_task_list) 
        average_batch_accuracy_score_per_batch = single_batch_sum_task_accuravy_score / num_task_list
        
        # calculate avg_throughput_score_per_batch
        avg_cpu_usage = total_cpu_usage_percentage / num_task_list
        avg_mem_usage = total_mem_usage_percentage / num_task_list
        avg_throughput_score_per_batch = average_local_processing_time_per_task/1000 + avg_cpu_usage + avg_mem_usage 
        
        # global score record
        sequence = origin_info.get('tasks',{}).get('sequence',0)
        self.client_sum_batch_num += 1
        self.client_sum_task_num += num_task_list
        self.client_sum_batch_accuravy_score += single_batch_sum_task_accuravy_score
        self.client_sum_batch_time_consumption += single_batch_time_consumption
        self.client_sum_batch_throughput_score += avg_throughput_score_per_batch
        self.client_task_num_batch.append(num_task_list)
        
        current_using_model_name = os.path.basename(current_using_model_path)[:-5] if current_using_model_path.endswith(".gguf") else os.path.basename(current_using_model_path)
        all_model_name_list = []
        for model_path in all_model_path:
            model_name = os.path.basename(model_path)[:-5] if model_path.endswith(".gguf") else os.path.basename(model_path)
            all_model_name_list.append(model_name)
        import random
        if self.client_sum_task_num // 8 == 0 and all_model_name_list:
            current_using_model_name = random.choice(all_model_name_list)
        # record metrics
        data_need_to_record = {
            "sequence":sequence,
            "task_id_list":task_id_list,
            "current_using_model_name":current_using_model_name,
            "all_model_name_list":all_model_name_list,
            "single_batch_time_consumption" : single_batch_time_consumption,
            "average_batch_accuracy_score_per_batch":average_batch_accuracy_score_per_batch,
            "avg_throughput_score_per_batch": avg_throughput_score_per_batch,
            "client_sum_batch_num":self.client_sum_batch_num,
            "client_sum_task_num":self.client_sum_task_num,
            "client_sum_batch_accuravy_score":self.client_sum_batch_accuravy_score,
            "client_sum_batch_time_consumption":self.client_sum_batch_time_consumption,
            "client_sum_batch_throughput_score":self.client_sum_batch_throughput_score,
            "client_task_num_batch":self.client_task_num_batch
        }
        self.data_need_to_record = data_need_to_record
        # 检查文件是否存在
        file_exists = os.path.isfile(self.record_metrics_file_path)
        
        # 写入CSV文件
        try:
            with open(self.record_metrics_file_path, 'a', newline='') as csvfile:
                # 定义字段名
                fieldnames = list(data_need_to_record.keys())
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_need_to_record)
        except Exception as e:
            print(f"记录数据到CSV文件时出错: {e}")        
        res = {
            "single_batch_time_consumption" : single_batch_time_consumption,
            "average_batch_accuracy_score_per_batch":average_batch_accuracy_score_per_batch,
            "avg_throughput_score_per_batch": avg_throughput_score_per_batch
        }
        logging.info(f"data_need_to_record = {data_need_to_record}")
        logging.info(f"res = {res}")
        return res
            

    def client_send_thread_simulation(self, client_socket, client_id):
        try:
            while True:
                print(f"Client {client_id} connected to server for sending.")
                data_to_send = self.sequence_queue.get()
                # # 模拟发送数据
                # for i in range(10):
                #     data_to_send = {
                #         'accuracy': 0.9 + client_id * 0.01,
                #         'time': 100 + client_id * 5,
                #         'memory_usage': 200 + client_id * 10
                #     }
                logging.info(f"Client {client_id} sending data from queue: {data_to_send}")
                message_json = json.dumps(data_to_send)
                message_bytes = message_json.encode('utf-8')

                # Send the length prefix
                client_socket.sendall(struct.pack('>I', len(message_bytes)))
                client_socket.sendall(message_bytes)
                time.sleep(1)
                
                print(f"Client {client_id} finished sending data.")
        except Exception as e:
            print(f"Client {client_id} encountered an error during send: {e}")

    def client_receive_thread(self, client_socket, client_id):
        """
        客户端接收线程，即使发生错误也能继续保持连接
        
        Args:
            client_socket: 客户端套接字
            client_id: 客户端ID
        """
        print(f"Client {client_id} ready to receive data.")
        
        # 标记套接字是否已关闭
        socket_closed = False
        
        while True:
            try:
                # 解析元数据长度
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    print(f"Client {client_id}: Connection closed by server (empty length bytes)")
                    break
                
                message_length = struct.unpack('>I', length_bytes)[0]
                print(f"Client {client_id} received message length: {message_length}")
                
                # 解析元数据
                message_bytes = bytearray()
                bytes_received = 0
                
                # 分块接收消息，防止大消息导致问题
                while bytes_received < message_length:
                    chunk = client_socket.recv(min(4096, message_length - bytes_received))
                    if not chunk:
                        print(f"Client {client_id}: Connection closed by server during message receive")
                        socket_closed = True
                        break
                    message_bytes.extend(chunk)
                    bytes_received += len(chunk)
                
                if socket_closed:
                    break
                    
                # 解码消息
                try:
                    message = json.loads(message_bytes.decode('utf-8'))
                    # 记录当前时间戳
                    temp_timestamp = datetime.datetime.now().timestamp()
                    
                    # 获取消息模式
                    mode = message.get('mode')
                    
                    if mode == DistributionType.MODEL.value:
                        print(f"Client {client_id} receiving model")
                        try:
                            # 处理模型传输
                            self.handle_file_transfer(client_socket, message, client_id)
                        except Exception as e:
                            print(f"Client {client_id} error during model transfer: {e}")
                            logging.error(f"Client {client_id} error during model transfer: {e}")
                            # 继续接收下一条消息，不中断连接
                            
                    elif mode == DistributionType.TASK.value:
                        try:
                            # 记录接收时间戳
                            sequence = message['tasks'].get('sequence', -1)
                            message['receive_timestamp'] = temp_timestamp
                            
                            # 放入任务队列
                            self.task_queue.put(message)
                            single_res = self.single_process()
                            temp_res = {
                                "sequence": sequence,
                                "observation": single_res
                            }
                            self.sequence_queue.put(temp_res)
                        except Exception as e:
                            print(f"Client {client_id} error during task processing: {e}")
                            logging.error(f"Client {client_id} error during task processing: {e}")
                            # 继续接收下一条消息，不中断连接
                    else:
                        print(f"Client {client_id} received unknown message mode: {mode}")
                        
                except json.JSONDecodeError as e:
                    print(f"Client {client_id} received invalid JSON: {e}")
                    logging.error(f"Client {client_id} received invalid JSON: {e}")
                    # 继续接收下一条消息，不中断连接
                    
            except ConnectionError as e:
                print(f"Client {client_id} connection error: {e}")
                logging.error(f"Client {client_id} connection error: {e}")
                # 连接错误，需要重新建立连接
                
            except Exception as e:
                print(f"Client {client_id} unexpected error: {e}")
                logging.error(f"Client {client_id} unexpected error: {e}")
                # 继续尝试接收，除非是致命错误
    def empty_directory(self, directory_path):
        """
        清空指定目录中的所有文件和子目录，但保留目录本身
        
        Args:
            directory_path (str): 要清空的目录的绝对路径
        """
        # 确保目录存在
        if not os.path.exists(directory_path):
            print(f"目录不存在: {directory_path}")
            return False
        
        # 确保是目录
        if not os.path.isdir(directory_path):
            print(f"指定路径不是目录: {directory_path}")
            return False
        
        # 遍历目录中的所有项目
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    # 如果是文件或符号链接，则删除
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    # 如果是目录，则递归删除
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"删除 {item_path} 时出错: {e}")
                return False
        print(f"目录已清空: {directory_path}")
        return True
    
    def handle_file_transfer(self, client_socket, metadata, client_id):
        file_name = metadata['file_name']
        file_size = metadata['file_size']
        delete_directory_success = self.empty_directory(self.models_save_directory)
        print(f"client remove directory status : {delete_directory_success}")
        file_path = os.path.join(self.models_save_directory, file_name)

        print(f"Client {client_id} handle_file_transfer function: {file_name}")

        received_size = 0
        with open(file_path, 'wb') as file:
            while received_size < file_size:
                chunk = client_socket.recv(min(self.chunk_size, file_size - received_size))
                if not chunk:
                    break
                file.write(chunk)
                received_size += len(chunk)
                print(f"Client {client_id} receiving: {received_size}/{file_size} bytes complete", end='\r')
        model_path = self.generate_model_to_be_used_path()
        llama_cli_path = self.generate_llama_cli_path()
        res, resource_status = self.cmd_operator.run_task_process_cmd(llama_cli_path = llama_cli_path, model_path=model_path)
        print(f"res = {res} resource_status = {resource_status}")
        print(f"\nClient {client_id} finished receiving file: {file_name}")
            
    def entry(self, num_clients):
        threads = []
        for i in range(num_clients):
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_ip, self.server_port))
            
            send_thread = threading.Thread(target=self.client_send_thread_simulation, args=(client_socket, i))
            receive_thread = threading.Thread(target=self.client_receive_thread, args=(client_socket, i))

            send_thread.start()
            receive_thread.start()

            threads.append(send_thread)
            threads.append(receive_thread)

        for thread in threads:
            thread.join()
# 使用函数模拟多个客户端
if __name__ == "__main__":
    client_simulator = Client_Connection(config_path = 'config/Running_config.json')
    client_simulator.entry(num_clients = 1)
    # client_simulator.task_queue.put([{"task_type":0,"token": "i can only be so abrasive towards people like brock lawly and the numerous nameless fundies before i start feeling lame", "reference_value": "0"},{"task_type":2,"token": "who captained the first european ship to sail around the tip of africa", "reference_value": "['Bartolomeu Dias']"}])
    # client_simulator.single_process()