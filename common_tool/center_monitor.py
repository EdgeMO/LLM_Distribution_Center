import os
import time
import json
import pandas as pd
import threading
import queue
import socket
import struct
from flask import Flask, render_template_string, jsonify, Response, send_file
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，无需显示器
import matplotlib.pyplot as plt
import io
import base64
import logging
import datetime
import ast
import sys
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 配置日志记录 - 更详细的日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

def find_free_port():
    """找到一个可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))  # 端口 0 表示让操作系统选择一个可用端口
        return s.getsockname()[1]
def parse_double_quoted_json(json_str):
    """
    解析CSV中特殊格式的JSON字段，处理双引号嵌套问题
    
    Args:
        json_str: 包含双引号的JSON字符串
    
    Returns:
        解析后的Python对象
    """
    try:
        # 步骤1: 去除字符串可能存在的外部引号
        if (json_str.startswith('"') and json_str.endswith('"')) or \
           (json_str.startswith("'") and json_str.endswith("'")):
            json_str = json_str[1:-1]
        
        # 步骤2: 替换双引号为单引号
        json_str = json_str.replace('""', '"')
        
        # 步骤3: 尝试解析JSON
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"问题字符串: {json_str}")
        
        # 检查是否有未配对的引号或括号
        quotes_count = json_str.count('"')
        if quotes_count % 2 != 0:
            print(f"警告: 引号数量不成对 ({quotes_count})")
        
        # 尝试手动修复常见问题
        if json_str.startswith('[') and ']' not in json_str:
            json_str += ']'
        elif json_str.startswith('{') and '}' not in json_str:
            json_str += '}'
        
        # 再次尝试解析
        try:
            return json.loads(json_str)
        except:
            # 如果还是失败，可以尝试使用ast.literal_eval
            try:
                import ast
                return ast.literal_eval(json_str)
            except:
                # 最后的备选方案：用正则表达式提取信息
                if '[' in json_str and ']' in json_str:  # 可能是任务列表
                    task_data = []
                    # 提取task_id, edge_id和task_type
                    edge_ids = re.findall(r'edge_id"?\s*:?\s*"([^"]+)"', json_str)
                    task_ids = re.findall(r'task_id"?\s*:?\s*(\d+)', json_str)
                    task_types = re.findall(r'task_type"?\s*:?\s*(\d+)', json_str)
                    
                    if task_ids:
                        for i in range(len(task_ids)):
                            edge_id = edge_ids[i] if i < len(edge_ids) else "0"
                            task_type = int(task_types[i]) if i < len(task_types) else 1
                            task_data.append({
                                "edge_id": edge_id, 
                                "task_id": int(task_ids[i]), 
                                "task_type": task_type
                            })
                        return task_data
                elif '{' in json_str and '}' in json_str:  # 可能是边缘观测
                    edge_data = {}
                    # 提取accuracy, latency和avg_throughput
                    edge_id_match = re.search(r'"(\d+)"', json_str)
                    edge_id = edge_id_match.group(1) if edge_id_match else "0"
                    
                    accuracy_match = re.search(r'accuracy"?\s*:?\s*([\d\.]+)', json_str)
                    latency_match = re.search(r'latency"?\s*:?\s*([\d\.]+)', json_str)
                    throughput_match = re.search(r'avg_throughput"?\s*:?\s*([\d\.]+)', json_str)
                    
                    edge_data[edge_id] = {
                        "accuracy": float(accuracy_match.group(1)) if accuracy_match else 0.0,
                        "latency": float(latency_match.group(1)) if latency_match else 0.0,
                        "avg_throughput": float(throughput_match.group(1)) if throughput_match else 0.0
                    }
                    return edge_data
    
    # 如果所有方法都失败，返回空对象
    if json_str.startswith('['):
        return []  # 对于任务分配
    elif json_str.startswith('{'):
        return {}  # 对于边缘观测
    return None
class CSVFileHandler(FileSystemEventHandler):
    """监控CSV文件变化的处理器"""
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = 0
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            current_time = time.time()
            # 防止在短时间内多次触发
            if current_time - self.last_modified > 1:
                self.last_modified = current_time
                self.callback(event.src_path)

class InferenceMonitor:
    def __init__(self, web_port=None, socket_port=None, csv_path="metrics/client/client_metrics.csv"):
        """
        初始化推理监控系统
        
        Args:
            web_port: Web服务器端口，如果为None则自动查找
            socket_port: Socket服务器端口，如果为None则自动查找
            csv_path: CSV文件路径，用于监控和读取数据
        """
        # 自动查找可用端口
        self.web_port = web_port if web_port is not None else find_free_port()
        self.socket_port = socket_port if socket_port is not None else find_free_port()
        
        # 确保两个端口不同
        if self.web_port == self.socket_port:
            self.socket_port = find_free_port()
            
        # CSV文件路径
        self.csv_path = os.path.abspath(csv_path)
        
        self.sequences = []  # 存储序列号
        self.inference_times = []  # 存储推理时间
        self.task_allocations = {}  # 按序列号存储任务分配信息
        self.edge_metrics = {}  # 存储边缘节点指标
        self.output_queue = queue.Queue()  # 存储日志输出
        
        # 新增: 历史指标数据
        self.historical_latency = []      # 历史延迟数据
        self.historical_accuracy = []     # 历史准确度数据
        self.historical_utility = []      # 历史效用分数数据
        
        # 创建锁以保护共享数据
        self.data_lock = threading.Lock()
        
        # 检查CSV文件目录是否存在，如果不存在则创建
        csv_dir = os.path.dirname(self.csv_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            self.log_message(f"Created directory: {csv_dir}")
        
        # 设置文件监视器
        self.setup_file_observer()
        
        # 创建 Flask 应用
        self.app = Flask(__name__)
        
        # 设置路由
        self.setup_routes()
        
        # 创建 socket 服务器
        self.socket_server = None
        self.setup_socket_server()
        
        # 尝试初始加载CSV文件
        self.try_load_csv_file()

    def setup_file_observer(self):
        """设置文件监视器来监控CSV文件变化"""
        self.log_message(f"Setting up file observer for {self.csv_path}")
        event_handler = CSVFileHandler(self.on_csv_changed)
        self.observer = Observer()
        
        # 监视文件所在的目录
        directory = os.path.dirname(self.csv_path)
        if os.path.exists(directory):
            self.observer.schedule(event_handler, directory, recursive=False)
            self.observer.start()
            self.log_message(f"File observer started for directory: {directory}")
        else:
            self.log_message(f"Warning: CSV directory does not exist: {directory}")

    def on_csv_changed(self, file_path):
        """当CSV文件发生变化时调用"""
        if file_path == self.csv_path:
            self.log_message(f"CSV file changed: {file_path}")
            self.try_load_csv_file()

    def try_load_csv_file(self):
        """尝试加载CSV文件"""
        try:
            if os.path.exists(self.csv_path):
                self.log_message(f"Loading CSV file: {self.csv_path}")
                
                # 直接读取文件内容
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    self.log_message("CSV file is empty")
                    return
                
                # 处理CSV数据
                self.process_csv_data(content)
            else:
                self.log_message(f"CSV file does not exist: {self.csv_path}")
        except Exception as e:
            self.log_message(f"Error loading CSV file: {e}")
            import traceback
            self.log_message(traceback.format_exc())

    def process_csv_data(self, content):
        """处理CSV格式数据，针对我们的特定格式进行优化"""
        lines = content.strip().split('\n')
        processed_count = 0
        
        # 跳过头行
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('sequence,'):
                continue
            
            try:
                # 使用更健壮的CSV解析方法
                parts = self._split_csv_line_robust(line)
                
                # 确保至少有三个字段
                if len(parts) < 3:
                    self.log_message(f"Line doesn't have enough fields: {line}")
                    continue
                
                # 解析序列和推理时间
                try:
                    sequence = int(parts[0])
                    inference_time = float(parts[1])
                except ValueError as e:
                    self.log_message(f"Error parsing sequence or inference time: {e}")
                    continue
                
                # 解析任务分配 - 使用专用的解析函数
                task_allocation = []
                if len(parts) > 2:
                    task_str = parts[2]
                    task_allocation = parse_double_quoted_json(task_str)
                    if task_allocation:
                        self.log_message(f"Successfully parsed {len(task_allocation)} tasks")
                    else:
                        self.log_message(f"No tasks parsed for sequence {sequence}")
                
                # 解析边缘观测 - 使用专用的解析函数
                edge_observations = {}
                if len(parts) > 3:
                    edge_str = parts[3]
                    edge_observations = parse_double_quoted_json(edge_str)
                    if edge_observations:
                        self.log_message(f"Successfully parsed edge observations with {len(edge_observations)} edges")
                    else:
                        self.log_message(f"No edge observations parsed for sequence {sequence}")
                
                # 更新数据
                with self.data_lock:
                    if sequence not in self.sequences:
                        self.sequences.append(sequence)
                        self.inference_times.append(inference_time)
                        # 添加推理时间到历史延迟数据
                        self.historical_latency.append(inference_time)  # 这里直接使用 inference_time
                        self.task_allocations[sequence] = task_allocation
                        self.process_edge_metrics(sequence, edge_observations)
                        processed_count += 1
                    
                    # 保持数据量合理
                    if len(self.sequences) > 100:
                        oldest_seq = self.sequences.pop(0)
                        self.inference_times.pop(0)
                        if oldest_seq in self.task_allocations:
                            del self.task_allocations[oldest_seq]
                        if oldest_seq in self.edge_metrics:
                            del self.edge_metrics[oldest_seq]
                
                self.log_message(f"Processed sequence {sequence} with latency {inference_time:.2f}ms")
                
            except Exception as e:
                self.log_message(f"Error processing line: {e}")
                import traceback
                self.log_message(traceback.format_exc())
        
        self.log_message(f"Processed {processed_count} new data points. Total data: {len(self.sequences)}")
    def _split_csv_line_robust(self, line):
        """
        更健壮地分割CSV行，正确处理引号内的逗号
        """
        if not line:
            return []
        
        result = []
        field_start = 0
        in_quotes = False
        
        for i, char in enumerate(line):
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                result.append(line[field_start:i])
                field_start = i + 1
        
        # 添加最后一个字段
        result.append(line[field_start:])
        
        return result
    def parse_csv_line(self, line):
        """使用正则表达式解析CSV行，正确处理引号内的逗号"""
        if not line:
            return []
        
        # 模式: 找到引号外的逗号或引号内的所有内容
        pattern = r'(?:"[^"]*")|(?:[^,]+)'
        matches = re.findall(pattern, line)
        
        # 处理匹配结果
        result = []
        buffer = ""
        in_quotes = False
        
        for i, part in enumerate(matches):
            if part.startswith('"') and part.endswith('"') and len(part) > 1:
                # 如果是完整的引号字段，则直接添加
                result.append(part)
            elif part.startswith('"'):
                # 开始一个带引号的字段
                in_quotes = True
                buffer = part
            elif part.endswith('"') and in_quotes:
                # 结束一个带引号的字段
                in_quotes = False
                buffer += "," + part
                result.append(buffer)
                buffer = ""
            elif in_quotes:
                # 引号内部的逗号
                buffer += "," + part
            else:
                # 普通字段
                result.append(part)
        
        # 如果最后还有缓冲内容，添加它
        if buffer:
            result.append(buffer)
        
        # 如果行最后有逗号，可能会得到一个空字符串
        if result and not result[-1]:
            result.pop()
            
        # 去除字段两端的引号
        for i in range(len(result)):
            if result[i].startswith('"') and result[i].endswith('"'):
                result[i] = result[i][1:-1]
                
        return result

    def calculate_historical_averages(self):
        """计算历史数据平均值"""
        avg_latency = sum(self.historical_latency) / len(self.historical_latency) if self.historical_latency else 0
        avg_accuracy = sum(self.historical_accuracy) / len(self.historical_accuracy) if self.historical_accuracy else 0
        avg_utility = sum(self.historical_utility) / len(self.historical_utility) if self.historical_utility else 0
        
        return {
            "avg_latency": avg_latency,
            "avg_accuracy": avg_accuracy,
            "avg_utility": avg_utility
        }

    def setup_routes(self):
        """设置 Flask 路由"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/data')
        def get_data():
            """获取当前数据"""
            with self.data_lock:
                historical_averages = self.calculate_historical_averages()
                data = {
                    'sequences': self.sequences,
                    'inference_times': self.inference_times,
                    'task_allocations': self.task_allocations,
                    'edge_metrics': self.edge_metrics,
                    'socket_port': self.socket_port,  # 添加Socket端口信息
                    'csv_path': self.csv_path,  # 添加CSV文件路径
                    'historical_data': {
                        'latency': self.historical_latency,
                        'accuracy': self.historical_accuracy,
                        'utility': self.historical_utility
                    },
                    'historical_averages': historical_averages
                }
                return jsonify(data)
        
        @self.app.route('/charts')
        def get_charts():
            """获取图表数据"""
            chart_images = self.generate_chart_images()
            return jsonify(chart_images)
        
        @self.app.route('/console_stream')
        def console_stream():
            """使用服务器发送事件流式传输日志输出"""
            def generate():
                while True:
                    try:
                        output = self.output_queue.get(timeout=1)
                        yield f"data: {json.dumps({'text': output})}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                    time.sleep(0.1)
            
            return Response(generate(), mimetype='text/event-stream')
        
        @self.app.route('/reload_csv')
        def reload_csv():
            """手动重新加载CSV文件"""
            self.try_load_csv_file()
            return jsonify({"status": "CSV file reloaded", "path": self.csv_path})
        
        @self.app.route('/reset_data')
        def reset_data():
            """重置所有数据"""
            with self.data_lock:
                self.sequences = []
                self.inference_times = []
                self.task_allocations = {}
                self.edge_metrics = {}
                # 重置历史数据
                self.historical_latency = []
                self.historical_accuracy = []
                self.historical_utility = []
                self.log_message("All data has been reset")
            return jsonify({"status": "All data reset"})
        
        @self.app.route('/test_data')
        def test_data():
            """生成测试数据"""
            test_data = """sequence,task_inference_time,tasks_allocation,edge_observations
0,25.030510425567627,"[{""edge_id"": ""0"", ""task_id"": 682, ""task_type"": 1}, {""edge_id"": ""0"", ""task_id"": 583, ""task_type"": 1}]","{""0"": {""accuracy"": 0.020399680852890015, ""latency"": 1.0, ""avg_throughput"": 54.63656347679008}}"
1,15.018349647521973,"[{""edge_id"": ""0"", ""task_id"": 741, ""task_type"": 1}, {""edge_id"": ""0"", ""task_id"": 1390, ""task_type"": 3}]","{""0"": {""accuracy"": 0.012697242975234986, ""latency"": 0.5, ""avg_throughput"": 50.2977599857191}}"""
            
            self.process_csv_data(test_data)
            return jsonify({"status": "Test data added", "data_count": len(self.sequences)})

    def setup_socket_server(self):
        """设置 socket 服务器以接收实时数据"""
        def socket_server():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('0.0.0.0', self.socket_port))
                server.listen(5)
                self.socket_server = server
                
                self.log_message(f"Socket server started on port {self.socket_port}")
                
                while True:
                    try:
                        client, addr = server.accept()
                        self.log_message(f"Client connected: {addr}")
                        
                        # 启动新线程处理客户端连接
                        client_thread = threading.Thread(target=self.handle_client, args=(client,))
                        client_thread.daemon = True
                        client_thread.start()
                    except Exception as e:
                        if isinstance(e, OSError) and e.errno == 9:  # Bad file descriptor, socket closed
                            break
                        self.log_message(f"Socket server error: {e}")
                        time.sleep(1)
            except OSError as e:
                self.log_message(f"Failed to bind socket to port {self.socket_port}: {e}")
                # 尝试查找另一个可用端口
                old_port = self.socket_port
                self.socket_port = find_free_port()
                self.log_message(f"Socket port {old_port} in use, switching to port {self.socket_port}")
                # 重新调用自身以使用新端口
                socket_server()
        
        # 启动 socket 服务器线程
        server_thread = threading.Thread(target=socket_server)
        server_thread.daemon = True
        server_thread.start()

    def handle_client(self, client):
        """处理客户端连接，接收和解析数据"""
        try:
            buffer = b''
            # 先发送一个欢迎消息
            welcome_msg = "Connected to Inference Monitor. Send data in CSV format.\n"
            client.send(welcome_msg.encode('utf-8'))
            
            while True:
                # 尝试接收数据
                chunk = client.recv(4096)
                if not chunk:
                    self.log_message("Client closed connection")
                    break
                
                buffer += chunk
                
                # 检查是否可以处理完整消息
                if b'\n' in buffer:
                    # 分割消息
                    messages = buffer.split(b'\n')
                    # 保留最后一个可能不完整的消息
                    buffer = messages.pop()
                    
                    # 处理完整的消息
                    for msg in messages:
                        if msg:  # 跳过空消息
                            try:
                                message = msg.decode('utf-8')
                                self.log_message(f"Received message: {message[:100]}...")
                                self.process_csv_data(message)
                            except Exception as e:
                                self.log_message(f"Error processing message: {e}")
                
        except Exception as e:
            self.log_message(f"Error handling client: {e}")
        finally:
            client.close()
            self.log_message("Client disconnected")

    def process_edge_metrics(self, sequence, edge_observations):
        """处理边缘节点指标，计算平均值"""
        if not edge_observations:
            self.log_message(f"No edge observations for sequence {sequence}")
            return
            
        accuracies = []
        latencies = []
        throughputs = []
        
        for edge_id, metrics in edge_observations.items():
            if "accuracy" in metrics:
                accuracies.append(metrics["accuracy"])
            if "latency" in metrics:
                latencies.append(metrics["latency"])
            if "avg_throughput" in metrics:
                throughputs.append(metrics["avg_throughput"])
        
        # 计算平均指标
        avg_metrics = {
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "edge_count": len(edge_observations)
        }
        
        # 存储每个边缘节点的原始指标和平均指标
        self.edge_metrics[sequence] = {
            "averages": avg_metrics,
            "edge_details": edge_observations
        }
        
        # 更新历史准确率和效用分数数据
        # 注意：我们不在这里更新 historical_latency，而是在 process_csv_data 中处理
        self.historical_accuracy.append(avg_metrics["avg_accuracy"])
        self.historical_utility.append(avg_metrics["avg_throughput"])
        
        # 保持历史数据长度合理
        max_history_length = 100
        if len(self.historical_accuracy) > max_history_length:
            self.historical_accuracy = self.historical_accuracy[-max_history_length:]
        if len(self.historical_utility) > max_history_length:
            self.historical_utility = self.historical_utility[-max_history_length:]
        
        self.log_message(f"Processed metrics for sequence {sequence}: accuracy={avg_metrics['avg_accuracy']:.4f}, " +
                         f"latency={avg_metrics['avg_latency']:.2f}, utility score={avg_metrics['avg_throughput']:.2f}")

    def generate_chart_images(self):
        """生成图表"""
        chart_images = {}
        
        with self.data_lock:
            if not self.sequences:
                self.log_message("No data available for chart generation")
                return {"error": "No data available"}
            
            try:
                # 1. 延迟时间图表
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(self.sequences, self.inference_times, 'bo-', label='Latency')
                ax.set_title('Latency by Sequence', fontsize=14)
                ax.set_xlabel('Sequence', fontsize=12)
                ax.set_ylabel('Time (ms)', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                chart_images['inference_time'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                # 2. 边缘节点平均指标图表 - 增大尺寸和字体
                if self.edge_metrics:
                    seq_list = sorted(self.edge_metrics.keys())
                    avg_accuracy = [self.edge_metrics[seq]["averages"]["avg_accuracy"] for seq in seq_list]
                    avg_latency = [self.edge_metrics[seq]["averages"]["avg_latency"] for seq in seq_list]
                    avg_throughput = [self.edge_metrics[seq]["averages"]["avg_throughput"] for seq in seq_list]
                    
                    # 增加图表尺寸，使用更大的图表和字体
                    fig, ax = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
                    
                    # 增加子图之间的空间
                    plt.subplots_adjust(hspace=0.3)
                    
                    # 设置更大的字体和线宽
                    line_width = 1.8
                    marker_size = 6
                    title_size = 14
                    axis_label_size = 10
                    tick_size = 8
                    
                    # 准确度图表
                    ax[0].plot(seq_list, avg_accuracy, 'g-o', linewidth=line_width, markersize=marker_size)
                    ax[0].set_title('Accuracy Across Edge Nodes', fontsize=title_size, fontweight='bold')
                    ax[0].set_xlabel('Sequence', fontsize=axis_label_size)
                    ax[0].set_ylabel('Accuracy', fontsize=axis_label_size)
                    ax[0].tick_params(axis='both', which='major', labelsize=tick_size)
                    ax[0].grid(True)
                    
                    # 延迟图表
                    ax[1].plot(seq_list, avg_latency, 'b-o', linewidth=line_width, markersize=marker_size)
                    ax[1].set_title('Latency Across Edge Nodes', fontsize=title_size, fontweight='bold')
                    ax[1].set_xlabel('Sequence', fontsize=axis_label_size)
                    ax[1].set_ylabel('Latency', fontsize=axis_label_size)
                    ax[1].tick_params(axis='both', which='major', labelsize=tick_size)
                    ax[1].grid(True)
                    
                    # 效用分数图表
                    ax[2].plot(seq_list, avg_throughput, 'r-o', linewidth=line_width, markersize=marker_size)
                    ax[2].set_title('Utility Score Across Edge Nodes', fontsize=title_size, fontweight='bold')
                    ax[2].set_xlabel('Sequence', fontsize=axis_label_size)
                    ax[2].set_ylabel('Utility Score', fontsize=axis_label_size)
                    ax[2].tick_params(axis='both', which='major', labelsize=tick_size)
                    ax[2].grid(True)
                    
                    # 使用更高的DPI保存图表
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120)
                    buf.seek(0)
                    chart_images['avg_metrics'] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                
                # 3. 任务分配图表
                if self.task_allocations:
                    all_edge_ids = set()
                    for tasks in self.task_allocations.values():
                        for task in tasks:
                            all_edge_ids.add(task.get("edge_id"))
                    
                    sequences = sorted(self.task_allocations.keys())
                    task_counts = {edge_id: [0] * len(sequences) for edge_id in all_edge_ids}
                    
                    for i, seq in enumerate(sequences):
                        tasks = self.task_allocations[seq]
                        edge_counts = {}
                        for task in tasks:
                            edge_id = task.get("edge_id")
                            if edge_id not in edge_counts:
                                edge_counts[edge_id] = 0
                            edge_counts[edge_id] += 1
                        
                        for edge_id, count in edge_counts.items():
                            task_counts[edge_id][i] = count
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for edge_id, counts in task_counts.items():
                        ax.plot(sequences, counts, 'o-', label=f'Edge {edge_id}')
                    
                    ax.set_title('Task Allocation by Edge Node', fontsize=14)
                    ax.set_xlabel('Sequence', fontsize=12)
                    ax.set_ylabel('Task Count', fontsize=12)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    ax.legend()
                    ax.grid(True)
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    chart_images['task_allocation'] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                
                self.log_message("Charts generated successfully")
            
            except Exception as e:
                self.log_message(f"Error generating charts: {e}")
                import traceback
                self.log_message(traceback.format_exc())
                return {"error": str(e)}
            
        return chart_images
    def log_message(self, message):
        """记录消息到日志和队列"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        logger.info(message)
        self.output_queue.put(formatted_message)

    def get_html_template(self):
        """获取HTML模板 - 专注于关键指标展示，并显示Socket端口和CSV文件信息"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inference Monitor</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f5f5f5;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .console {
            background-color: #000;
            color: #fff;
            font-family: 'Courier New', monospace;
            height: 200px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 5px;
        }
        .chart-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* 调整右侧图表尺寸，使其不那么大 */
        .metrics-chart-image {
            max-width: 100%;
            min-height: 600px; /* 从800px减小到600px */
            object-fit: contain; /* 保持图像比例 */
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 1rem;
            color: #6c757d;
        }
        .historical-avg {
            font-size: 0.9rem;
            color: #6c757d;
            font-style: italic;
        }
        .task-allocation-table {
            font-size: 0.9rem;
            max-height: 300px;
            overflow-y: auto;
        }
        .page-title {
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5rem;
            color: #333;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .button-group {
            margin-bottom: 20px;
            text-align: center;
        }
        .button-group button {
            margin: 0 5px;
        }
        .main-section {
            display: flex;
            flex-direction: row;
            gap: 20px;
        }
        .left-column {
            flex: 0 0 52%;  /* 左列宽度 */
        }
        .right-column {
            flex: 0 0 46%;  /* 右列宽度 */
        }
        .metrics-card {
            height: 100%;
        }
        /* 减小右侧卡片内容区域的高度 */
        .metrics-card .card-body {
            min-height: 650px; /* 从850px减小到650px */
            padding: 15px; /* 增加内边距使内容居中 */
        }
        @media (max-width: 992px) {
            .main-section {
                flex-direction: column;
            }
            .left-column, .right-column {
                flex: 0 0 100%;
            }
            .metrics-card .card-body {
                min-height: auto;
            }
            .metrics-chart-image {
                min-height: 500px; /* 在小屏幕上进一步减少高度 */
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid"> <!-- 使用container-fluid以利用更多屏幕空间 -->
        <h1 class="page-title">Inference Performance Monitor</h1>
        
        <div class="row button-group">
            <div class="col-12">
                <button id="reload-csv-btn" class="btn btn-primary">Reload CSV File</button>
                <button id="reset-data-btn" class="btn btn-danger">Reset All Data</button>
                <button id="test-data-btn" class="btn btn-success">Load Test Data</button>
                <button id="refresh-btn" class="btn btn-info">Refresh Charts</button>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Latest Performance Metrics</h4>
                    </div>
                    <div class="card-body">
                        <div class="row" id="latest-metrics">
                            <div class="col-md-3 text-center">
                                <div class="metric-value text-primary" id="latest-sequence">-</div>
                                <div class="metric-label">Latest Sequence</div>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="metric-value text-info" id="latest-inference-time">-</div>
                                <div class="metric-label">Latency (ms)</div>
                                <div class="historical-avg" id="avg-latency">Avg: -</div>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="metric-value text-success" id="latest-accuracy">-</div>
                                <div class="metric-label">Accuracy</div>
                                <div class="historical-avg" id="avg-accuracy">Avg: -</div>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="metric-value text-warning" id="latest-throughput">-</div>
                                <div class="metric-label">Utility Score</div>
                                <div class="historical-avg" id="avg-utility">Avg: -</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 修改部分：使用两列布局，右侧有图表 -->
        <div class="main-section">
            <!-- 左侧列：推理时间图和任务分配图 -->
            <div class="left-column">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">Latency Chart</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container" id="inference-time-chart">
                            <div class="spinner-border text-info" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="mb-0">Task Allocation Chart</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container" id="task-allocation-chart">
                            <div class="spinner-border text-success" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h5 class="mb-0">Latest Task Allocation Details</h5>
                    </div>
                    <div class="card-body">
                        <div class="task-allocation-table">
                            <table class="table table-striped table-bordered" id="task-details-table">
                                <thead>
                                    <tr>
                                        <th>Edge ID</th>
                                        <th>Task ID</th>
                                        <th>Task Type</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td colspan="3" class="text-center">No task data available</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 右侧列：边缘节点平均指标，使用适中大小的图表 -->
            <div class="right-column">
                <div class="card metrics-card">
                    <div class="card-header bg-warning text-white">
                        <h5 class="mb-0">Edge Node Average Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container" id="avg-metrics-chart">
                            <div class="spinner-border text-warning" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-dark text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">System Log</h5>
                        <button id="clear-console" class="btn btn-sm btn-light">Clear</button>
                    </div>
                    <div class="card-body p-0">
                        <div id="console" class="console"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let refreshInterval = null;
        
        // DOM elements
        const latestSequenceEl = document.getElementById('latest-sequence');
        const latestInferenceTimeEl = document.getElementById('latest-inference-time');
        const latestAccuracyEl = document.getElementById('latest-accuracy');
        const latestThroughputEl = document.getElementById('latest-throughput');
        const inferenceTimeChartEl = document.getElementById('inference-time-chart');
        const taskAllocationChartEl = document.getElementById('task-allocation-chart');
        const avgMetricsChartEl = document.getElementById('avg-metrics-chart');
        const taskDetailsTableEl = document.getElementById('task-details-table').querySelector('tbody');
        const consoleOutput = document.getElementById('console');
        const refreshBtn = document.getElementById('refresh-btn');
        const clearConsoleBtn = document.getElementById('clear-console');
        const testDataBtn = document.getElementById('test-data-btn');
        const reloadCsvBtn = document.getElementById('reload-csv-btn');
        const resetDataBtn = document.getElementById('reset-data-btn');
        
        // Initialize app
        function initializeApp() {
            // Set up event listeners
            refreshBtn.addEventListener('click', refreshData);
            clearConsoleBtn.addEventListener('click', () => {
                consoleOutput.innerHTML = '';
            });
            testDataBtn.addEventListener('click', loadTestData);
            reloadCsvBtn.addEventListener('click', reloadCsvFile);
            resetDataBtn.addEventListener('click', resetAllData);
            
            // Set up console event stream
            setupConsoleEventSource();
            
            // Start auto refresh
            startAutoRefresh();
            
            // Initial data load
            refreshData();
        }
        
        // Reset all data
        function resetAllData() {
            if (confirm('Are you sure you want to reset all data?')) {
                appendToConsole("Resetting all data...");
                fetch('/reset_data')
                    .then(response => response.json())
                    .then(data => {
                        appendToConsole(`${data.status}`);
                        refreshData();
                    })
                    .catch(error => {
                        console.error('Error resetting data:', error);
                        appendToConsole(`Error resetting data: ${error}`);
                    });
            }
        }
        
        // Reload CSV file
        function reloadCsvFile() {
            appendToConsole("Reloading CSV file...");
            fetch('/reload_csv')
                .then(response => response.json())
                .then(data => {
                    appendToConsole(`CSV file reloaded`);
                    refreshData();
                })
                .catch(error => {
                    console.error('Error reloading CSV file:', error);
                    appendToConsole(`Error reloading CSV file: ${error}`);
                });
        }
        
        // Load test data
        function loadTestData() {
            appendToConsole("Loading test data...");
            fetch('/test_data')
                .then(response => response.json())
                .then(data => {
                    appendToConsole(`Test data loaded: ${data.data_count} data points available now`);
                    refreshData();
                })
                .catch(error => {
                    console.error('Error loading test data:', error);
                    appendToConsole(`Error loading test data: ${error}`);
                });
        }
        
        // Set up Server-Sent Events
        function setupConsoleEventSource() {
            const eventSource = new EventSource('/console_stream');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.text) {
                    appendToConsole(data.text);
                }
            };
            
            eventSource.onerror = function() {
                console.error('EventSource failed, reconnecting in 5 seconds...');
                setTimeout(setupConsoleEventSource, 5000);
            };
        }
        
        // Start auto refresh
        function startAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
            refreshInterval = setInterval(refreshData, 5000);
        }
        
        // Refresh all data
        function refreshData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    updateLatestMetrics(data);
                    updateTaskDetailsTable(data);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    appendToConsole(`Error fetching data: ${error}`);
                });
                
            fetch('/charts')
                .then(response => response.json())
                .then(data => {
                    updateCharts(data);
                })
                .catch(error => {
                    console.error('Error fetching charts:', error);
                    appendToConsole(`Error fetching charts: ${error}`);
                });
        }
        
        // Update latest metrics
        function updateLatestMetrics(data) {
            if (data.sequences && data.sequences.length > 0) {
                const latestIndex = data.sequences.length - 1;
                const latestSequence = data.sequences[latestIndex];
                const latestInferenceTime = data.inference_times[latestIndex];
                
                latestSequenceEl.textContent = latestSequence;
                latestInferenceTimeEl.textContent = latestInferenceTime.toFixed(2);
                
                // Update historical averages
                if (data.historical_averages) {
                    document.getElementById('avg-latency').textContent = 
                        `Avg: ${data.historical_averages.avg_latency.toFixed(2)} ms`;
                    document.getElementById('avg-accuracy').textContent = 
                        `Avg: ${data.historical_averages.avg_accuracy.toFixed(4)}`;
                    document.getElementById('avg-utility').textContent = 
                        `Avg: ${data.historical_averages.avg_utility.toFixed(2)}`;
                }
                
                // Update edge metrics if available
                if (data.edge_metrics && data.edge_metrics[latestSequence]) {
                    const metrics = data.edge_metrics[latestSequence].averages;
                    latestAccuracyEl.textContent = metrics.avg_accuracy.toFixed(4);
                    latestThroughputEl.textContent = metrics.avg_throughput.toFixed(2);
                }
            }
        }
        
        // Update task details table
        function updateTaskDetailsTable(data) {
            if (data.sequences && data.sequences.length > 0 && data.task_allocations) {
                const latestSequence = data.sequences[data.sequences.length - 1];
                const latestTasks = data.task_allocations[latestSequence];
                
                if (latestTasks && latestTasks.length > 0) {
                    // Clear table
                    taskDetailsTableEl.innerHTML = '';
                    
                    // Add rows for each task
                    for (const task of latestTasks) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${task.edge_id}</td>
                            <td>${task.task_id}</td>
                            <td>${task.task_type}</td>
                        `;
                        taskDetailsTableEl.appendChild(row);
                    }
                } else {
                    taskDetailsTableEl.innerHTML = '<tr><td colspan="3" class="text-center">No tasks for this sequence</td></tr>';
                }
            } else {
                taskDetailsTableEl.innerHTML = '<tr><td colspan="3" class="text-center">No task data available</td></tr>';
            }
        }
        
        // Update charts
        function updateCharts(data) {
            if (data.error) {
                console.error('Error in chart data:', data.error);
                appendToConsole(`Error in chart data: ${data.error}`);
                return;
            }
            
            if (data.inference_time) {
                inferenceTimeChartEl.innerHTML = `<img src="data:image/png;base64,${data.inference_time}" class="chart-image" alt="Latency Chart">`;
            } else {
                inferenceTimeChartEl.innerHTML = '<div class="alert alert-info">No latency data available</div>';
            }
            
            if (data.task_allocation) {
                taskAllocationChartEl.innerHTML = `<img src="data:image/png;base64,${data.task_allocation}" class="chart-image" alt="Task Allocation Chart">`;
            } else {
                taskAllocationChartEl.innerHTML = '<div class="alert alert-info">No task allocation data available</div>';
            }
            
            if (data.avg_metrics) {
                // 为边缘节点指标图表使用特殊的CSS类以显示适中大小的图表
                avgMetricsChartEl.innerHTML = `<img src="data:image/png;base64,${data.avg_metrics}" class="metrics-chart-image" alt="Average Metrics Chart">`;
            } else {
                avgMetricsChartEl.innerHTML = '<div class="alert alert-info">No metrics data available</div>';
            }
        }
        
        // Append text to console
        function appendToConsole(text) {
            const line = document.createElement('div');
            line.textContent = text;
            
            // Apply different colors based on content
            if (text.toLowerCase().includes('error')) {
                line.style.color = '#ff5555';
            } else if (text.toLowerCase().includes('warning')) {
                line.style.color = '#ffcc00';
            } else if (text.toLowerCase().includes('processed')) {
                line.style.color = '#55ff55';
            } else if (text.toLowerCase().includes('loaded')) {
                line.style.color = '#55aaff';
            }
            
            consoleOutput.appendChild(line);
            
            // Auto-scroll
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        
        // Initialize the app when DOM is loaded
        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
        """

    def run(self):
        """运行监控系统"""
        try:
            print(f"Starting inference monitor...")
            print(f"Web interface available at http://localhost:{self.web_port}")
            print(f"Socket server running on port {self.socket_port}")
            print(f"Monitoring CSV file: {self.csv_path}")
            self.app.run(host='0.0.0.0', port=self.web_port, debug=False, threaded=True)
        except Exception as e:
            logging.error(f"Error starting inference monitor: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def __del__(self):
        """清理资源"""
        # 停止文件观察器
        if hasattr(self, 'observer'):
            try:
                self.observer.stop()
                self.observer.join()
                self.log_message("File observer stopped")
            except:
                pass
        
        # 关闭socket服务器
        if hasattr(self, 'socket_server') and self.socket_server:
            try:
                self.socket_server.close()
                self.log_message("Socket server closed")
            except:
                pass

if __name__ == "__main__":
    try:
        # 创建并运行推理监控系统，使用相对路径
        monitor = InferenceMonitor(csv_path="metrics/core/core_metrics.csv")
        monitor.run()
    except KeyboardInterrupt:
        print("\nShutting down inference monitor")
    except Exception as e:
        print(f"Error running inference monitor: {e}")
        import traceback
        print(traceback.format_exc())