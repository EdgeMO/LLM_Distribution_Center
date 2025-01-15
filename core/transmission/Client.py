import threading
import time
import json
import socket
import os 
import sys
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
        self.models_directory = '/models'
        self.csv_path = 'metrics.csv'
        os.makedirs(self.models_directory, exist_ok=True)  # 确保模型目录存在
        
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
                data = client_socket.recv(1024)
                if not data:
                    break

                messages = json.loads(data.decode('utf-8'))
                print(f"Client {client_id} received messages: {messages}")
                for message in messages:
                    message_type = message.get('mode')

                    if message_type == DistributionType.TASK:
                        # Handle text task
                        self.task_queue.put(message)
                        print(f"Client {client_id} received text task: {message}")
                    elif message_type == DistributionType.MODEL:
                        # Handle model data
                        model_name = message.get('model_name', f'model_{client_id}.mdl')
                        model_data = message.get('model_data', '')
                        model_path = os.path.join(self.models_directory, model_name)
                        
                        with open(model_path, 'w') as model_file:
                            model_file.write(model_data)
                        
                        print(f"Client {client_id} stored model at {model_path}")
        except Exception as e:
            print(f"Client {client_id} encountered an error during receive: {e}")
        finally:
            client_socket.close()
            print(f"Client {client_id} finished receiving data and disconnected.")

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
