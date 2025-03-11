import socket
import time
import argparse
import sys
from datetime import datetime

def test_connection(server_host, server_port, timeout=5):
    """测试与服务器的连接"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在连接服务器 {server_host}:{server_port}...")
    
    try:
        # 创建 socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)
        
        # 连接服务器
        start_time = time.time()
        client_socket.connect((server_host, server_port))
        connect_time = time.time() - start_time
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 连接成功! 用时: {connect_time:.3f} 秒")
        
        # 接收欢迎消息
        welcome_msg = client_socket.recv(1024).decode('utf-8')
        print(f"服务器消息: {welcome_msg}")
        
        # 进入交互模式
        print("\n===== 连接测试成功 =====")
        print("可用命令:")
        print("  ping - 测试服务器响应")
        print("  time - 获取服务器时间")
        print("  exit - 断开连接")
        print("  quit - 退出客户端")
        print("=====================\n")
        
        while True:
            # 获取用户输入
            message = input("发送消息 (输入 'quit' 退出客户端): ")
            
            # 检查是否退出客户端
            if message.lower() == 'quit':
                break
                
            # 发送消息到服务器
            client_socket.send(message.encode('utf-8'))
            
            # 接收服务器响应
            response = client_socket.recv(1024).decode('utf-8')
            print(f"服务器响应: {response}")
            
            # 如果发送了退出命令，则断开连接
            if message.lower() == 'exit':
                break
        
    except socket.timeout:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 连接超时: 无法连接到 {server_host}:{server_port}")
        return False
    except ConnectionRefusedError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 连接被拒绝: 服务器可能未运行或端口不正确")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 连接错误: {e}")
        return False
    finally:
        try:
            client_socket.close()
        except:
            pass
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 连接已关闭")
    
    return True

def run_ping_test(server_host, server_port, count=5, interval=1):
    """运行多次 ping 测试"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始连续 ping 测试 ({count} 次)...")
    
    try:
        # 创建 socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        
        # 连接服务器
        client_socket.connect((server_host, server_port))
        
        # 跳过欢迎消息
        _ = client_socket.recv(1024)
        
        # 进行多次 ping 测试
        successful_pings = 0
        total_time = 0
        
        for i in range(1, count + 1):
            start_time = time.time()
            
            try:
                # 发送 ping
                client_socket.send(b"ping")
                
                # 接收响应
                response = client_socket.recv(1024).decode('utf-8')
                
                # 计算往返时间
                rtt = (time.time() - start_time) * 1000  # 毫秒
                total_time += rtt
                successful_pings += 1
                
                print(f"Ping #{i}: 响应 = {response}, RTT = {rtt:.2f} ms")
                
            except Exception as e:
                print(f"Ping #{i}: 失败 - {e}")
            
            # 等待指定间隔
            if i < count:
                time.sleep(interval)
        
        # 显示统计信息
        if successful_pings > 0:
            print(f"\n--- {server_host} ping 统计 ---")
            print(f"发送 = {count}, 接收 = {successful_pings}, 丢失 = {count - successful_pings} ({(count - successful_pings) / count * 100:.1f}% 丢失)")
            print(f"平均往返时间 = {total_time / successful_pings:.2f} ms")
        else:
            print("\n所有 ping 请求均失败")
            
    except Exception as e:
        print(f"Ping 测试错误: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TCP 客户端连通性测试")
    parser.add_argument("host", help="服务器主机名或IP地址")
    parser.add_argument("-p", "--port", type=int, default=9999, help="服务器端口号")
    parser.add_argument("-m", "--mode", choices=["interactive", "ping"], default="interactive", 
                        help="测试模式: interactive (交互式) 或 ping (连续ping测试)")
    parser.add_argument("-c", "--count", type=int, default=5, help="ping 测试次数")
    parser.add_argument("-i", "--interval", type=float, default=1, help="ping 测试间隔 (秒)")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        # 交互式测试模式
        while True:
            success = test_connection(args.host, args.port)
            
            if not success or input("\n再次测试连接? (y/n): ").lower() != 'y':
                break
    else:
        # Ping 测试模式
        run_ping_test(args.host, args.port, args.count, args.interval)

# python tcp_client.py 192.168.1.100 -p 8888