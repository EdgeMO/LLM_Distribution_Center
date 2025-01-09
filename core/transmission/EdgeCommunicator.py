# author : junyuan wu
# description: 用于与边缘设备通信的类，包括发送任务消息和发送模型文件
"""

支持两种连接方式，

一种是本地读取配置文件，读取固定的ip地址

另一种是通过socket连接，接受所有的边缘节点的连接

"""

import socket
import json
import threading
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

class EdgeCommunicator:
    def __init__(self, config_file_path, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port    
        self.config_file_path = config_file_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients_index_ip_dict = {}
        self.lock = threading.Lock()  # 创建一个线程锁

    def process(self, local_config_file_read=False):
        threading.Thread(target=self.server_accept_all_connections, args=(local_config_file_read,)).start()
        
    def handle_client(self, client_socket, index):
        while True:
            try:
                #print(f"Waiting for data from client {index}...")
                data = client_socket.recv(1024)
                if not data:
                    print(f"Client {index} disconnected.")
                    break

                message = json.loads(data.decode('utf-8'))
                if 'accuracy' in message and 'time' in message and 'memory_usage' in message:
                    # 使用锁保护对共享字典的访问
                    with self.lock:
                        self.clients_index_ip_dict[str(index)]['accuracy'] = message['accuracy']
                        self.clients_index_ip_dict[str(index)]['time'] = message['time']
                        self.clients_index_ip_dict[str(index)]['memory_usage'] = message['memory_usage']
                    
                    # 打印接收到的数据
                    print(f"Received data from client {index}: {message}")

            except ConnectionResetError:
                print(f"Client {index} forcibly closed the connection.")
                break
            except json.JSONDecodeError:
                print(f"JSON decode error from client {index}.")
        
    def server_accept_all_connections(self, local_config_file_read):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server is listening on {self.host}:{self.port}")

        if not local_config_file_read:
            index = 0
            while True:
                client_socket, addr = self.server_socket.accept()
                ip, port = addr
                with self.lock:
                    self.clients_index_ip_dict[str(index)] = {
                        'ip': ip,
                        'port': port,
                        'socket': client_socket
                    }
                print(f"Accepted connection from {ip}:{port} with index {index}")

                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, index))
                client_thread.daemon = True
                client_thread.start()
                index += 1
                # TBD  data receiving logic

        if local_config_file_read == True:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
                ip_config = config.get('edge_ip_config',None)
                self.clients_index_ip_dict = ip_config
                print(ip_config)



if __name__ == "__main__":
    communicator = EdgeCommunicator(config_file_path='config/Running_config.json')
    communicator.process(local_config_file_read = False)
