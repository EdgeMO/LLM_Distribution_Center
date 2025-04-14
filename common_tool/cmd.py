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
        
        pass
    def run_task_process_cmd(self, llama_cli_path, model_path, query_prefix='please translate the sentence in english', query_word="我是谁?"):
        command = [
            llama_cli_path,
            "-m", model_path,
            "-p", f"{query_prefix} {query_word}",
            "-n", "128"
        ]
        monitor = ResourceMonitor()
        
        # Create a socket connection to the visualizer
        try:
            vis_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            vis_socket.connect(("localhost", 12346))  # Visualizer should listen on this port
        except Exception as e:
            print(f"Failed to connect to visualizer: {e}")
            vis_socket = None
        
        try:
            monitor.start_monitoring()
            
            # Use Popen instead of run to capture output in real-time
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Collect all output for the return value
            full_output = ""
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print to console
                full_output += line
                
                # Send to visualizer if connected
                if vis_socket:
                    try:
                        # Send the line with a length prefix
                        message_bytes = line.encode('utf-8')
                        vis_socket.sendall(struct.pack('>I', len(message_bytes)))
                        vis_socket.sendall(message_bytes)
                    except Exception as e:
                        print(f"Failed to send output to visualizer: {e}")
                        vis_socket = None
            
            # Wait for process to complete
            process.wait()
            monitor.stop_monitoring()
            resource_stats = monitor.get_resource_stats()
            
            # Close the visualizer socket
            if vis_socket:
                vis_socket.close()
            
            # Extract the actual result
            res = self.extract_after_start(full_output)
            return res, resource_stats
            
        except Exception as e:
            # Stop monitoring
            monitor.stop_monitoring()
            print(f"Error occurred: {e}")
            
            # Close the visualizer socket
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
    def run_perplexity_process_cmd(self,perplexity_path,model_path,test_txt_file_path):
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
            result = subprocess.run(command,capture_output=True, text=True, check=True)
            print(result.stdout)
            temp_res = result.stdout
            
            return temp_res
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            with open("perplexity_error.txt", "a+") as file:
                file.write(e.stderr)
            return ''
# 定义要执行的命令和参数

        
if __name__ == '__main__':
    cmd = CMD()
    #res = cmd.get_free_disk_space()
    query_words = "please answer the following question based on the text provided without explanation \n\n Question:"
    question = f"what's the final sum of 2 + 2"
    llama_cli_path = "/home/wu/workspace/LLM_Distribution_Edge/build/bin/llama-cli"
    model_path = "/home/wu/workspace/qwen2.5-7b-instruct-q4_0.gguf"
    res,resource_monitor = cmd.run_task_process_cmd(query_words,question,llama_cli_path,model_path)
    print(resource_monitor)
    pass