# used for test tcp connection , check whether the firewall is inactive or not


import socket
import threading
import argparse
import time
import sys
from datetime import datetime

def handle_client(client_socket, client_address):
    """处理客户端连接"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 新连接: {client_address}")
    
    try:
        # 发送欢迎消息
        welcome_msg = f"欢迎连接到服务器! 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        client_socket.send(welcome_msg.encode('utf-8'))
        
        # 持续接收客户端消息
        while True:
            # 设置接收超时
            client_socket.settimeout(300)  # 5分钟超时
            
            # 接收数据
            data = client_socket.recv(1024)
            if not data:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端 {client_address} 断开连接")
                break
            
            # 处理接收到的数据
            message = data.decode('utf-8').strip()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 从 {client_address} 收到: {message}")
            
            # 处理特殊命令
            if message.lower() == "ping":
                response = "pong"
            elif message.lower() == "time":
                response = f"服务器时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elif message.lower() == "exit":
                response = "再见!"
                client_socket.send(response.encode('utf-8'))
                break
            else:
                response = f"收到消息: {message}"
            
            # 发送响应
            client_socket.send(response.encode('utf-8'))
            
    except socket.timeout:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端 {client_address} 连接超时")
    except ConnectionResetError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 客户端 {client_address} 连接被重置")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理客户端 {client_address} 时出错: {e}")
    finally:
        client_socket.close()

def start_server(host='0.0.0.0', port=9999):
    """启动 TCP 服务器"""
    try:
        # 创建 socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定地址和端口
        server_socket.bind((host, port))
        
        # 开始监听
        server_socket.listen(5)
        
        local_ip = get_local_ip()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器启动成功!")
        print(f"监听地址: {host}:{port}")
        print(f"本机IP地址: {local_ip}")
        print("等待客户端连接...\n")
        
        while True:
            # 接受客户端连接
            client_socket, client_address = server_socket.accept()
            
            # 为每个客户端创建新线程
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器正在关闭...")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器错误: {e}")
    finally:
        try:
            server_socket.close()
        except:
            pass
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 服务器已关闭")

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个临时socket连接，用于获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "无法获取IP地址"

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TCP 服务器连通性测试")
    parser.add_argument("-p", "--port", type=int, default=9999, help="监听的端口号")
    parser.add_argument("-a", "--address", default="0.0.0.0", help="监听的地址")
    
    args = parser.parse_args()
    
    # 启动服务器
    start_server(args.address, args.port)
    
# python tcp_server.py -p 8888
# python tcp_server.py
# 检查 firewalld 是否运行
# sudo systemctl status firewalld
# # 启动 firewalld
# sudo systemctl start firewalld

# # 设置开机自启
# sudo systemctl enable firewalld

# # 停止 firewalld
# sudo systemctl stop firewalld

# # 禁用 firewalld
# sudo systemctl disable firewalld
# # 开放单个 TCP 端口 (如 9999)
# sudo firewall-cmd --permanent --add-port=9999/tcp

# # 开放端口范围 (如 8000-9000)
# sudo firewall-cmd --permanent --add-port=8000-9000/tcp

# # 应用更改
# sudo firewall-cmd --reload
# # 关闭端口
# sudo firewall-cmd --permanent --remove-port=9999/tcp
# sudo firewall-cmd --reload


# sudo apt install ufw
# # 检查 ufw 状态
# sudo ufw status

# # 启用 ufw (首次启用前确保已添加允许 SSH 的规则)
# sudo ufw enable

# # 禁用 ufw
# sudo ufw disable

# # 开放单个 TCP 端口 (如 9999)
# sudo ufw allow 9999/tcp

# # 开放端口范围
# sudo ufw allow 8000:9000/tcp

# # 删除规则
# sudo ufw delete allow 9999/tcp

# # 安装 ufw
# sudo apt update
# sudo apt install ufw

# # 配置基本规则
# sudo ufw default deny incoming
# sudo ufw default allow outgoing
# sudo ufw allow ssh
# sudo ufw allow 9999/tcp
# sudo ufw enable

# 使用以下命令验证端口是否开放：
# # 使用 netstat 检查本地监听端口
# sudo netstat -tulpn | grep 9999

# # 使用 ss 命令检查 (更现代的工具)
# sudo ss -tulpn | grep 9999

# # 使用 nmap 从另一台机器检查
# nmap -p 9999 服务器IP地址