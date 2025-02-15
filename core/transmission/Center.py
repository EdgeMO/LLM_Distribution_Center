# author : junyuan wu

"""

center core workspace


"""

import struct
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
import logging

# 在文件开头配置日志记录
logging.basicConfig(filename='center_process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EdgeCommunicator:
    def __init__(self, config_file_path, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port    
        self.config_file_path = config_file_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients_index_ip_dict = {} # 用于存储所有的边缘节点的观测量
        self.lock = threading.Lock()  # 创建一个线程锁
        self.message_queues = {}  # 用来存储每个客户端的消息队列
        self.chunk_size = 8192  # 8KB chunks for file transfer

    def establish_connection(self):
        # 负责建立网路连接
        threading.Thread(target=self.server_accept_all_connections).start()
        time.sleep(10)  # 等待服务器线程启动
    

    def receive_from_client_worker(self, client_socket, index):
        """每有来自终端的信息的时候，更新本地的观测观测量

        Args:
            client_socket (_type_): _description_
            index (_type_): _description_
        """
        while True:
            try:
                #print(f"Waiting for data from client {index}...")
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                message_length = struct.unpack('>I', length_bytes)[0]
                # parse meta data
                print(f"Client {index} received message length: {message_length}")
                message_bytes = client_socket.recv(message_length)
                
                message = json.loads(message_bytes.decode('utf-8'))
                
                single_observation = message.get('observation', None)
                if single_observation:
                    with self.lock:
                        self.clients_index_ip_dict[str(index)]['observation'] = single_observation
                    # 打印接收到的数据
                logging.info(f"Received data from client {index}: type {type(message)} message = {message}")
                logging.info(f"get_observation: {self.get_client_observation_by_index(index)}")

            except ConnectionResetError:
                print(f"Client {index} forcibly closed the connection.")
                break
            except json.JSONDecodeError:
                print(f"JSON decode error from client {index}.")
    
    def send_task_message_to_client(self, index, message):
        """ send message to client by index, e.g. insert message into queue

        Args:
            index (_type_): client index
            message (_type_): json format message
            
        """
        if str(index) in self.message_queues:
            self.message_queues[str(index)].put(message)
        else:
            print(f"No client with index {index} is connected.")
    
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
            # wait for connection, establish new connection for each edge node
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
            client_thread = threading.Thread(target=self.receive_from_client_worker, args=(client_socket, index))
            client_thread.daemon = True
            client_thread.start()
            # 开启线程，给client 端发送消息
            sending_thread = threading.Thread(target=self.send_to_client_worker, args=(client_socket, index))
            sending_thread.daemon = True
            sending_thread.start()

            index += 1
    
    def generate_meta_data(self, enum, file_name = None, file_size = None, task_message = None):
        res = {}
        if enum == DistributionType.MODEL:
            res['mode'] = DistributionType.MODEL
            res['file_name'] = file_name
            res['file_size'] = file_size
        elif enum == DistributionType.TASK:
            res['mode'] = DistributionType.TASK
            res['tasks'] = task_message
        return res
    
    def process_model_file_transmission(self, client_index, file_path):
        """
        send model file to client by index, for model transmission finishied , the task offloading can continue
        """
        print(" get in process model function")
        if str(client_index) not in self.clients_index_ip_dict:
            print(f"No client with index {client_index}")
            return

        client_socket = self.clients_index_ip_dict[str(client_index)]['socket']

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            return
        
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        print("find file path {file_path} , file size {file_size}  and file name {file_name}")
        # 发送文件元数据
        metadata = self.generate_meta_data(DistributionType.MODEL, file_name = file_name, file_size=file_size)
        print(f"model transmission meta data : {metadata}")
        metadata = json.dumps(metadata,cls=EnumEncoder).encode('utf-8')
        # format in 4 bytes
        client_socket.sendall(struct.pack('>I', len(metadata)))
        client_socket.sendall(metadata)

        # 发送文件内容
        with open(file_path, 'rb') as file:
            sent = 0
            while sent < file_size:
                chunk = file.read(self.chunk_size)
                if not chunk:
                    break
                client_socket.sendall(chunk)
                sent += len(chunk)
                print(f"Sent {sent}/{file_size} bytes to client {client_index}")

        print(f"File {file_name} sent successfully to client {client_index}")

    def process_task_message_transmission(self, client_socket, message):
        # send meta data
        task_message = self.generate_meta_data(DistributionType.TASK, task_message=message)

        # Serialize the message
        message_json = json.dumps(task_message, cls=EnumEncoder)
        message_bytes = message_json.encode('utf-8')

        # Send the length prefix
        client_socket.sendall(struct.pack('>I', len(message_bytes)))

        # Send the actual message
        client_socket.sendall(message_bytes)

        
    
    def send_to_client_worker(self, client_socket, index):
        """ 
        worker function for sending message to client from message queue

        Args:
            client_socket (_type_): socket when connection established
            index (_type_): edge node index 
        """
        while True:
            try:
                message = self.message_queues[str(index)].get()
                print("server message", message)
                
                mode = message.get('mode')
                print(f"Sending message to client {index} with mode {mode} type of node {type(mode)}")
                if mode == DistributionType.MODEL:
                    print("send model file to index {index}")
                    # model transmission
                    self.process_model_file_transmission(index, message['file_path'])
                elif mode == DistributionType.TASK:
                    # task_transmission
                    print(f"Sent TASK message to client {index}")
                    self.process_task_message_transmission(client_socket, message['data'])
            except Exception as e:
                print(f"Error sending message to client {index}: {e}")
                break
        
        
    def get_client_observation_by_index(self,index):
        """获取所有边缘节点的观测量"""
        with self.lock:
            return self.clients_index_ip_dict[str(index)]['observation']




if __name__ == "__main__":
    communicator = EdgeCommunicator(config_file_path='config/Running_config.json')
    communicator.establish_connection()
    # 算法实际的迭代过程 和任务下发有关
    message_list = {"sequence":0,"mode":DistributionType.TASK, "data" : [{"task_type":0,"token": "i can only be so abrasive towards people like brock lawly and the numerous nameless fundies before i start feeling lame", "reference_value": "0"},{"task_type":2,"token": "who captained the first european ship to sail around the tip of africa", "reference_value": "['Bartolomeu Dias']"}]}
    model_message = {
        "mode": DistributionType.MODEL,
        "file_path": "/mnt/data/workspace/LLM_Distribution_Center/model/models/distilBert/distilgpt2.IQ3_M.gguf"
    }
    communicator.send_task_message_to_client(0, message_list)
