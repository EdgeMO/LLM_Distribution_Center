import subprocess
import psutil
import threading
import time
import statistics
import socket
import struct

class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.sampling_interval = 0.1  # 采样间隔，单位：秒
    
    def get_cpu_available_usage(self):
        """获取 CPU 使用率"""
        return psutil.cpu_percent(interval=None)
    
    def get_memory_available_percentage(self):
        """获取内存使用率"""
        memory = psutil.virtual_memory()
        return memory.percent
    
    def _monitor_resources(self):
        """资源监控线程函数"""
        while self.monitoring:
            self.cpu_samples.append(self.get_cpu_available_usage())
            self.memory_samples.append(self.get_memory_available_percentage())
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """开始监控资源"""
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控资源"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def get_resource_stats(self):
        """获取资源统计信息"""
        if not self.cpu_samples or not self.memory_samples:
            return {
                "cpu": 0,
                "memory": 0
            }
        
        return {
            "cpu": statistics.mean(self.cpu_samples),
            "memory": statistics.mean(self.memory_samples)
        }
        
class CMD:
    def __init__(self):
        # 将 start_record 变量移到 CMD 类中
        self.start_record = False
        
    def run_task_process_cmd(self, llama_cli_path, model_path, query_prefix='please translate the sentence in english', query_word="我是谁?", enable_visualization=False):
        """
        运行任务处理命令，执行模型推理
        
        Args:
            llama_cli_path (str): llama 命令行工具路径
            model_path (str): 模型文件路径
            query_prefix (str): 查询前缀
            query_word (str): 查询文本
            enable_visualization (bool): 是否启用可视化监控，默认为 False
            
        Returns:
            tuple: (推理结果, 资源统计信息)
        """
        command = [
            llama_cli_path,
            "-m", model_path,
            "-p", f"{query_prefix} {query_word}",
            "-n", "128"
        ]
        monitor = ResourceMonitor()
        
        # 只有在启用可视化时才创建 socket 连接
        vis_socket = None
        if enable_visualization:
            try:
                vis_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                vis_socket.connect(("localhost", 12346))  # 可视化器应监听此端口
                print("Connected to visualization monitor")
            except Exception as e:
                print(f"Failed to connect to visualizer: {e}")
                vis_socket = None
        
        try:
            monitor.start_monitoring()
            
            # 使用 Popen 代替 run 以实时捕获输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 收集所有输出作为返回值
            full_output = ""
            # 重置记录状态
            self.start_record = False
            
            # 逐行读取输出
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # 打印到控制台
                full_output += line
                if 'start' in line:
                    self.start_record = True
                # 仅当启用可视化且 socket 连接成功时发送到可视化器
                if self.start_record and enable_visualization and vis_socket:
                    try:
                        # 发送带有长度前缀的行
                        message_bytes = line.encode('utf-8')
                        vis_socket.sendall(struct.pack('>I', len(message_bytes)))
                        vis_socket.sendall(message_bytes)
                    except Exception as e:
                        print(f"Failed to send output to visualizer: {e}")
                        vis_socket = None
            
            # 重置记录状态
            self.start_record = False
            
            # 等待进程完成
            process.wait()
            monitor.stop_monitoring()
            resource_stats = monitor.get_resource_stats()
            
            # 关闭可视化器 socket
            if vis_socket:
                vis_socket.close()
            
            # 提取实际结果
            res = self.extract_after_start(full_output)
            return res, resource_stats
            
        except Exception as e:
            # 停止监控
            monitor.stop_monitoring()
            print(f"Error occurred: {e}")
            
            # 关闭可视化器 socket
            if vis_socket:
                vis_socket.close()
                
            return '', monitor.get_resource_stats()
    def extract_after_start(self, text):
        start_marker = "[start]"
        end_marker = "[end]"
        
        # 找到开始标记的位置
        start_pos = text.find(start_marker)
        if start_pos == -1:
            # 没找到开始标记，返回原文本
            return text
        
        # 移动到开始标记之后
        start_pos += len(start_marker)
        
        # 找到结束标记的位置（从开始标记之后开始查找）
        end_pos = text.find(end_marker, start_pos)
        if end_pos == -1:
            # 如果没有找到结束标记，返回开始标记之后的所有内容
            return text[start_pos:]
        else:
            # 返回开始标记到结束标记之间的内容
            return text[start_pos:end_pos]
    
    def run_perplexity_process_cmd(self, perplexity_path, model_path, test_txt_file_path):
        """ for edge node to run perplexity process command

        Args:
            model_path (_type_): path to model
            test_txt_file_path (_type_): path to test txt file
        """
        command = [
            perplexity_path,
            "-m", model_path,
            "-f", f"{test_txt_file_path}",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(result.stdout)
            temp_res = result.stdout
            
            return temp_res
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            with open("perplexity_error.txt", "a+") as file:
                file.write(e.stderr)
            return ''


if __name__ == '__main__':
    cmd = CMD()
    #res = cmd.get_free_disk_space()
    query_words = "please answer the following question based on the text provided without explanation \n\n Question:"
    question = f"what's the final sum of 2 + 2"
    llama_cli_path = "/home/junyuan/workspace/LLM_Distribution_Edge/build/bin/llama-cli"
    model_path = "/home/junyuan/workspace/LLM_Distribution_Center/model/models/Qwen2.5-7B-Instruct.Q3_K_S.gguf"
    res, resource_monitor = cmd.run_task_process_cmd(llama_cli_path=llama_cli_path, model_path=model_path, query_prefix=query_words, query_word=question, enable_visualization=False)
    print(resource_monitor)
    pass