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
    def __init__(self, config_file_path , host = '0.0.0.0', port = 12345):
        self.host = host
        self.port = port    
        # 保存配置文件的路径
        self.config_file_path = config_file_path
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = {}

    def process(self, local_config_file_read = False):
        # 开启一个线程，用于接受所有端节点的连接
        threading.Thread(target=self.server_accept_all_connections, args=(local_config_file_read,)).start()
    
    def server_accept_all_connections(self, local_config_file_read = False):
        """
            起一个服务器, 记录所有连接的客户端的socket和地址, 并且更新连接的状态
        """
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server is listening on {self.host}:{self.port}")

        if local_config_file_read == False:
            while True:
                # 记录所有连接的客户端的socket和地址
                client_socket, addr = self.server_socket.accept()
                ip, port = addr
                print(f"Connected by {ip}:{port}")
                self.clients[ip] = client_socket    
        if local_config_file_read == True:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
                
                print(config)
            
    def state_monitor(self):
        """
        基于收到的客户端，更新本地的状态
        1. 接收到边缘节点传回的信息
        2. 基于核心函数的计算方式, 更新主动推理的空间状态
        """



if __name__ == "__main__":
    communicator = EdgeCommunicator(config_file_path='config/Running_config.json')
    communicator.process(local_config_file_read = True)
