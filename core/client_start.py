import os
import sys
import threading
import time
import tkinter as tk
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common_tool.client_moniter import create_monitor_window,ClientMonitor
from transmission.Client import Client_Connection

def run_client(client_instance):
    """
    在单独的线程中运行客户端
    
    Args:
        client_instance: Client_Connection实例
    """
    try:
        client_instance.entry(num_clients=1)
    except Exception as e:
        print(f"客户端运行错误: {e}")

if __name__ == "__main__":
    # 创建客户端实例
    client_simulator = Client_Connection(config_path='config/Running_config.json')
    
    # 创建并启动客户端监控窗口
    monitor, root = create_monitor_window(client_simulator)
    
    # 在单独的线程中启动客户端
    client_thread = threading.Thread(target=run_client, args=(client_simulator,))
    client_thread.daemon = True
    client_thread.start()
    
    try:
        # 在主线程中运行Tkinter主循环
        root.mainloop()
    except Exception as e:
        print(f"GUI错误: {e}")
    finally:
        # 停止监控
        monitor.stop_monitoring()