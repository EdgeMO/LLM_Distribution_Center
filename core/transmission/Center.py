# author : junyuan wu

"""

center core workspace


"""

import socket
import json
import threading
import os
import sys
import queue
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config.type import DistributionType, TaskType, EnumEncoder

class EdgeCommunicator:
    def __init__(self, config_file_path, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port    
        self.config_file_path = config_file_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients_index_ip_dict = {} # 用于存储所有的边缘节点的观测量
        self.lock = threading.Lock()  # 创建一个线程锁
        self.message_queues = {}  # 用来存储每个客户端的消息队列

    def establish_connection(self):
        
        # 负责建立网路连接
        threading.Thread(target=self.server_accept_all_connections).start()
        time.sleep(5)  # 等待服务器线程启动
    

    def receive_from_client_(self, client_socket, index):
        """每有来自终端的信息的时候，更新本地的观测观测量

        Args:
            client_socket (_type_): _description_
            index (_type_): _description_
        """
        while True:
            try:
                #print(f"Waiting for data from client {index}...")
                data = client_socket.recv(1024)
                if not data:
                    print(f"Client {index} disconnected.")
                    break

                message = json.loads(data.decode('utf-8'))
                if 'accuracy' in message and 'time' in message and 'memory_usage' in message:
                    # 使用锁保护对共享字典的访问, 这里用来更新观测量
                    with self.lock:
                        self.clients_index_ip_dict[str(index)]['accuracy'] = message['accuracy']
                        self.clients_index_ip_dict[str(index)]['time'] = message['time']
                        self.clients_index_ip_dict[str(index)]['memory_usage'] = message['memory_usage']
                    
                    # 打印接收到的数据
                    print(f"Received data from client {index}: {self.clients_index_ip_dict[str(index)]['accuracy']}")

            except ConnectionResetError:
                print(f"Client {index} forcibly closed the connection.")
                break
            except json.JSONDecodeError:
                print(f"JSON decode error from client {index}.")
    
    def send_message_to_client(self, index, message):
        """ send message to client by index, e.g. insert message into queue

        Args:
            index (_type_): client index
            message (_type_): json format message
            
        """
        if str(index) in self.message_queues:
            self.message_queues[str(index)].put(message)
        else:
            print(f"No client with index {index} is connected.")
  
    def send_to_client_(self, client_socket, index):
        """专门的线程从队列中取消息并发送到客户端"""
        while True:
            try:
                message = self.message_queues[str(index)].get()
                message_json = json.dumps(message, cls=EnumEncoder)
                client_socket.sendall(message_json.encode('utf-8'))  # 假设消息是字符串
            except Exception as e:
                print(f"Error sending message to client {index}: {e}")
                break
    
    def server_accept_all_connections(self):
        """
        thread function
        1. establish connection with all edge nodes
        2. local state observation space update   "self.clients_index_ip_dict"
        3. send task offloading decisions and model distribution to edge nodes
        """
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server is listening on {self.host}:{self.port}")

        index = 0 #边缘节点编号
        while True:
            client_socket, addr = self.server_socket.accept()
            ip, port = addr
            with self.lock:
                self.clients_index_ip_dict[str(index)] = {
                    'ip': ip,
                    'port': port,
                    'socket': client_socket
                }
            self.message_queues[str(index)] = queue.Queue()  # 为每个客户端创建消息队列
            print(f"Accepted connection from {ip}:{port} with index {index}")
            
            # 开启线程，接收来自client 端的消息
            client_thread = threading.Thread(target=self.receive_from_client_, args=(client_socket, index))
            client_thread.daemon = True
            client_thread.start()
            # 开启线程，给client 端发送消息
            sending_thread = threading.Thread(target=self.send_to_client_, args=(client_socket, index))
            sending_thread.daemon = True
            sending_thread.start()

            index += 1
    def get_observation(self):
        """获取所有边缘节点的观测量"""
        with self.lock:
            return self.clients_index_ip_dict




if __name__ == "__main__":
    communicator = EdgeCommunicator(config_file_path='config/Running_config.json')
    communicator.establish_connection()
    # 算法实际的迭代过程 和任务下发有关
    message_list = [
        {"mode":DistributionType.TASK,"task_type": TaskType.QuestionAnswering, "token": "token1", "true_value": "value1"},
        {"mode":DistributionType.TASK,"task_type": TaskType.TextClassification, "token": "token2", "true_value": "value2"}
    ]
    communicator.send_message_to_client(1, message_list)
