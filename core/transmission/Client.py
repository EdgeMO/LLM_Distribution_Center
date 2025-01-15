import threading
import time
import json
import socket
import os 
import sys
import struct
import csv
from queue import Queue
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common_tool.init import *
from config.type import DistributionType, TaskType, EnumEncoder

class Client_Connection:
    def __init__(self, config_path):
        self.server_config = client_server_config_init(config_path)
        self.server_ip = self.server_config.get('server_ip', None)
        self.server_port = self.server_config.get('server_port', None)
        self.task_queue = Queue()
        self.models_directory = "downloaded_models"
        os.makedirs(self.models_directory, exist_ok=True)
        self.csv_path = 'metrics.csv'        # 确保模型目录存在
        self.chunk_size = 8192  # Matching the server's chunk size
    
    def client_send_thread(self, client_socket, client_id):
        try:
            print(f"Client {client_id} connected to server for sending.")
            
            last_position = 0
            while True:
                with open(self.csv_path, 'r') as f:
                    f.seek(last_position)
                    reader = csv.DictReader(f)
                    for row in reader:
                        accuracy = float(row['accuracy'])
                        time_delay = float(row['time'])
                        memory_usage = float(row['memory_usage'])

                        data_to_send = {
                            'accuracy': accuracy,
                            'time': time_delay,
                            'memory_usage': memory_usage
                        }
                        data_json = json.dumps(data_to_send)
                        client_socket.sendall(data_json.encode('utf-8'))

                        print(f"Client {client_id} sent data: {data_to_send}")
                        time.sleep(1)  # 控制发送频率

                    last_position = f.tell()  # 记录文件当前位置以便下次读取

                time.sleep(1)  # 控制文件检查频率
            
            print(f"Client {client_id} finished sending data.")
        except Exception as e:
            print(f"Client {client_id} encountered an error during send: {e}")
            
    def client_send_thread_simulation(self, client_socket, client_id):
        try:
            print(f"Client {client_id} connected to server for sending.")

            # 模拟发送数据
            for i in range(10):
                data_to_send = {
                    'accuracy': 0.9 + client_id * 0.01,
                    'time': 100 + client_id * 5,
                    'memory_usage': 200 + client_id * 10
                }
                data_json = json.dumps(data_to_send)
                client_socket.sendall(data_json.encode('utf-8'))
                time.sleep(1)
            
            print(f"Client {client_id} finished sending data.")
        except Exception as e:
            print(f"Client {client_id} encountered an error during send: {e}")

    def client_receive_thread(self, client_socket, client_id):
        try:
            print(f"Client {client_id} ready to receive data.")
            
            while True:
                # parse meta data
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                message_length = struct.unpack('>I', length_bytes)[0]
                # Now, receive the full message
                print(f"Client {client_id} received message length: {message_length}")
                message_bytes = client_socket.recv(message_length)
                message = json.loads(message_bytes.decode('utf-8'))
                # get the mode of the message
                mode = message.get('mode')
                
                if mode == DistributionType.MODEL.value:
                    # model processing 
                    self.handle_file_transfer(client_socket, message, client_id)
                elif mode == DistributionType.TASK.value:
                    # task processing
                    # example: Client 1 received : {'mode': 0, 'task_data': [{'mode': 0, 'task_type': 1, 'token': 'token1', 'true_value': 'value1'}, {'mode': 0, 'task_type': 0, 'token': 'token2', 'true_value': 'value2'}]}
                    print(f"Client {client_id} received : {message}")
                    self.task_queue.put(message)
                        

        except Exception as e:
            print(f"Client {client_id} encountered an error during receive: {e}")
        finally:
            client_socket.close()
            print(f"Client {client_id} finished receiving data and disconnected.")
            
    def handle_file_transfer(self, client_socket, metadata, client_id):
        file_name = metadata['file_name']
        file_size = metadata['file_size']
        file_path = os.path.join(self.models_directory, file_name)

        print(f"Client {client_id} preparing to receive file: {file_name}")

        received_size = 0
        with open(file_path, 'wb') as file:
            while received_size < file_size:
                chunk = client_socket.recv(min(self.chunk_size, file_size - received_size))
                if not chunk:
                    break
                file.write(chunk)
                received_size += len(chunk)
                print(f"Client {client_id} receiving: {received_size}/{file_size} bytes complete", end='\r')

        print(f"\nClient {client_id} finished receiving file: {file_name}")
            
    def process(self, num_clients):
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
    client_simulator.process(num_clients = 5)
