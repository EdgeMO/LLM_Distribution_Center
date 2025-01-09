import threading
import time
import json
import socket

def client_thread(server_ip, server_port, client_id):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        print(f"Client {client_id} connected to server.")

        # 模拟发送数据
        for i in range(10):  # 每个客户端发送10次数据
            data_to_send = {
                'accuracy': 0.9 + client_id * 0.01,  # 模拟不同的准确率
                'time': 100 + client_id * 5,         # 模拟不同的耗时
                'memory_usage': 200 + client_id * 10 # 模拟不同的内存使用
            }
            data_json = json.dumps(data_to_send)
            client_socket.sendall(data_json.encode('utf-8'))
            time.sleep(1)  # 每秒发送一次数据
        
        client_socket.close()
        print(f"Client {client_id} disconnected.")
    except Exception as e:
        print(f"Client {client_id} encountered an error: {e}")

def simulate_multiple_clients(num_clients, server_ip='0.0.0.0', server_port=12345):
    threads = []
    for i in range(num_clients):
        thread = threading.Thread(target=client_thread, args=(server_ip, server_port, i))
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()

# 使用函数模拟多个客户端
simulate_multiple_clients(5)  # 模拟5个客户端连接到服务器