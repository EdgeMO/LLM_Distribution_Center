import os
import time
import pandas as pd
import ast
import json
import threading
import queue
import socket
import struct
import psutil  # 新增导入
from flask import Flask, render_template_string, jsonify, Response, send_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，无需显示器
import matplotlib.pyplot as plt
import io
import base64

class WebMetricsVisualizer:
    def __init__(self, csv_file_path='metrics/client/client_metrics.csv', port=12347):
        """
        Initialize web-based metrics visualizer
        
        Args:
            csv_file_path: Path to the CSV file
            port: Web server port
        """
        self.csv_file_path = csv_file_path
        self.port = port
        self.data = None
        self.last_modified_time = 0
        self.output_queue = queue.Queue()
        self.auto_refresh = True
        
        # 新增系统资源监控数据存储
        self.system_metrics = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used': [],
            'memory_total': []
        }
        self.sys_metrics_max_points = 60  # 存储最近60个数据点
        
        # 创建 Flask 应用
        self.app = Flask(__name__)
        
        # 设置路由
        self.setup_routes()
        
        # 设置文件监控
        self.setup_file_monitoring()
        
        # 设置命令输出接收器
        self.setup_command_output_receiver()
        
        # 初始加载数据
        self.load_data()
        
        # 启动系统资源监控
        self.start_system_monitoring()

    def setup_routes(self):
        """设置 Flask 路由"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/data')
        def get_data():
            """获取当前数据的 API 端点"""
            if self.data is None or self.data.empty:
                return jsonify({'data': None, 'stats': {}, 'sequence': 0})
            
            # 转换数据为 JSON 格式
            data_json = self.data.tail(20).to_dict(orient='records')
            
            # 计算统计数据
            stats = self.calculate_stats()
            
            return jsonify({
                'data': data_json,
                'stats': stats,
                'sequence': int(self.data['sequence'].iloc[-1]) if not self.data.empty else 0
            })
        
        @self.app.route('/charts')
        def get_charts():
            """获取图表的 API 端点"""
            if self.data is None or self.data.empty:
                return jsonify({'error': 'No data available'})
            
            # 生成所有图表
            chart_images = self.generate_chart_images()
            
            return jsonify(chart_images)
        
        @self.app.route('/system_metrics')
        def get_system_metrics():
            """获取系统资源指标的 API 端点"""
            # 获取当前最新的系统资源指标
            latest_metrics = {
                'cpu_percent': self.system_metrics['cpu_percent'][-1] if self.system_metrics['cpu_percent'] else 0,
                'memory_percent': self.system_metrics['memory_percent'][-1] if self.system_metrics['memory_percent'] else 0,
                'memory_used': round(self.system_metrics['memory_used'][-1], 2) if self.system_metrics['memory_used'] else 0,
                'memory_total': round(self.system_metrics['memory_total'][-1], 2) if self.system_metrics['memory_total'] else 0,
            }
            
            # 返回历史数据和当前指标
            return jsonify({
                'history': self.system_metrics,
                'latest': latest_metrics
            })
        
        @self.app.route('/console_stream')
        def console_stream():
            """使用服务器发送事件流式传输控制台输出"""
            def generate():
                while True:
                    try:
                        # 从队列获取输出，设置超时
                        output = self.output_queue.get(timeout=1)
                        yield f"data: {json.dumps({'text': output})}\n\n"
                    except queue.Empty:
                        # 发送心跳以保持连接
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                    time.sleep(0.1)
            
            return Response(generate(), mimetype='text/event-stream')
        
        @self.app.route('/toggle_refresh')
        def toggle_refresh():
            """切换自动刷新设置"""
            self.auto_refresh = not self.auto_refresh
            return jsonify({'auto_refresh': self.auto_refresh})
        
        @self.app.route('/export_data')
        def export_data():
            """导出数据为 CSV 文件"""
            if self.data is None or self.data.empty:
                return jsonify({'error': 'No data available'})
            
            # 创建临时文件
            temp_file = f'metrics_export_{time.strftime("%Y%m%d_%H%M%S")}.csv'
            self.data.to_csv(temp_file, index=False)
            
            # 发送文件
            return send_file(temp_file, as_attachment=True, download_name=temp_file)

    def calculate_stats(self):
        """计算统计数据"""
        if self.data is None or self.data.empty:
            return {}
        
        last_row = self.data.iloc[-1]
        
        # 计算准确率，避免除以零
        task_num = last_row.get('client_sum_task_num', 0)
        accuracy = 0 if task_num == 0 else last_row.get('client_sum_batch_accuravy_score', 0) / task_num
        
        # 计算平均时间，避免除以零
        batch_num = last_row.get('client_sum_batch_num', 0)
        avg_time = 0 if batch_num == 0 else last_row.get('client_sum_batch_time_consumption', 0) / batch_num
        
        # 获取当前模型名称和所有模型列表
        current_model = last_row.get('current_using_model_name', 'N/A')
        all_models = last_row.get('all_model_name_list', [])
        if isinstance(all_models, str):
            try:
                all_models = ast.literal_eval(all_models)
            except:
                all_models = [all_models]
        
        return {
            'total_batches': int(batch_num),
            'total_tasks': int(task_num),
            'avg_accuracy': round(float(accuracy), 4),
            'avg_time': round(float(avg_time), 4),
            'total_accuracy': round(float(last_row.get('client_sum_batch_accuravy_score', 0)), 2),
            'total_time': round(float(last_row.get('client_sum_batch_time_consumption', 0)), 2),
            'total_throughput': round(float(last_row.get('client_sum_batch_throughput_score', 0)), 2),
            'current_model': current_model,
            'all_models': all_models
        }

    def generate_chart_images(self):
        """生成图表的图像，包括性能指标和系统资源图表"""
        chart_images = {}
        
        try:
            # 创建 1x3 子图，保留原来的性能指标图表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 获取 x 轴数据
            x = self.data['sequence']
            
            # 1. 单批次处理时间
            axes[0].plot(x, self.data['single_batch_time_consumption'], 'b-o')
            axes[0].set_title('Batch Processing Time')
            axes[0].set_xlabel('Sequence')
            axes[0].set_ylabel('Time (ms)')
            
            # 2. 每批次准确率
            axes[1].plot(x, self.data['average_batch_accuracy_score_per_batch'], 'g-o')
            axes[1].set_title('Batch Accuracy Score')
            axes[1].set_xlabel('Sequence')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_ylim([0, 1.1])
            
            # 3. 每批次吞吐量
            axes[2].plot(x, self.data['avg_throughput_score_per_batch'], 'r-o')
            axes[2].set_title('Batch Utility Score')  # 更改标题为Utility Score
            axes[2].set_xlabel('Sequence')
            axes[2].set_ylabel('Score')
            
            # 调整布局
            fig.tight_layout()
            
            # 将图表转换为 base64 编码的图像
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            chart_images['main_chart'] = img_str
            
            # 关闭图表以释放内存
            plt.close(fig)
            
            # 新增: 创建系统资源监控图表
            if self.system_metrics['timestamps']:
                # 创建系统资源图表，包含CPU和内存使用率
                fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
                
                # 时间戳
                x_time = self.system_metrics['timestamps']
                
                # CPU 使用率图表
                axes2[0].plot(x_time, self.system_metrics['cpu_percent'], 'b-o')
                axes2[0].set_title('CPU Usage')
                axes2[0].set_ylabel('Usage (%)')
                axes2[0].set_ylim([0, 100])
                axes2[0].grid(True)
                
                # 内存使用率图表
                axes2[1].plot(x_time, self.system_metrics['memory_percent'], 'r-o')
                axes2[1].set_title('Memory Usage')
                axes2[1].set_xlabel('Time')
                axes2[1].set_ylabel('Usage (%)')
                axes2[1].set_ylim([0, 100])
                axes2[1].grid(True)
                
                # 调整布局
                fig2.tight_layout()
                
                # 将系统资源图表转换为 base64 编码的图像
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='png', dpi=100)
                buf2.seek(0)
                img_str2 = base64.b64encode(buf2.read()).decode('utf-8')
                chart_images['system_chart'] = img_str2
                
                # 关闭图表以释放内存
                plt.close(fig2)
            
        except Exception as e:
            print(f"Error generating chart images: {e}")
            chart_images['error'] = str(e)
        
        return chart_images

    def setup_file_monitoring(self):
        """设置文件监控"""
        try:
            # 创建文件系统事件处理程序
            class FileChangeHandler(FileSystemEventHandler):
                def __init__(self, callback):
                    self.callback = callback
                    
                def on_modified(self, event):
                    if not event.is_directory and event.src_path.endswith(self.callback.csv_file_path):
                        if self.callback.auto_refresh:
                            self.callback.load_data()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            
            # 设置观察者
            self.event_handler = FileChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, path=os.path.dirname(self.csv_file_path), recursive=False)
            self.observer.start()
        except Exception as e:
            print(f"Error setting up file monitoring: {e}")

    def setup_command_output_receiver(self):
        """设置命令输出接收器"""
        def socket_server():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('localhost', 12346))
                server_socket.listen(5)
                print("Command output receiver started, listening on port 12346")
                
                while True:
                    try:
                        # 接受连接
                        client_socket, addr = server_socket.accept()
                        print(f"Connection accepted from {addr}")
                        
                        # 启动线程处理此客户端连接
                        client_thread = threading.Thread(
                            target=self.handle_client_connection,
                            args=(client_socket,),
                            daemon=True
                        )
                        client_thread.start()
                    except Exception as e:
                        print(f"Error accepting connection: {e}")
                        time.sleep(1)  # 避免高 CPU 使用率
            except Exception as e:
                print(f"Socket server error: {e}")
        
        # 启动 socket 服务器线程
        server_thread = threading.Thread(target=socket_server, daemon=True)
        server_thread.start()
        
    def handle_client_connection(self, client_socket):
        """处理客户端连接，接收命令输出"""
        try:
            print("New client connection established")
            while True:
                # 首先接收 4 字节长度前缀
                length_prefix = client_socket.recv(4)
                if not length_prefix:
                    print("Connection closed by client")
                    break  # 连接关闭
                    
                # 解析消息长度
                message_length = struct.unpack('>I', length_prefix)[0]
                print(f"Receiving message of length: {message_length}")
                
                # 接收完整消息
                received = 0
                message_data = b''
                
                while received < message_length:
                    chunk = client_socket.recv(min(4096, message_length - received))
                    if not chunk:
                        break
                    message_data += chunk
                    received += len(chunk)
                
                if received == message_length:
                    # 解码并添加到输出队列
                    try:
                        decoded_message = message_data.decode('utf-8')
                        print(f"Received message: {decoded_message[:50]}...")  # 打印前50个字符
                        self.output_queue.put(decoded_message)
                    except UnicodeDecodeError:
                        print(f"Error decoding message: {message_data}")
        except Exception as e:
            print(f"Error handling client connection: {e}")
        finally:
            print("Client connection closed")
            client_socket.close()
            
    def load_data(self):
        """加载 CSV 数据"""
        try:
            if os.path.exists(self.csv_file_path):
                self.data = pd.read_csv(self.csv_file_path)
                
                # 处理列表格式的列
                list_columns = ['client_task_num_batch', 'task_id_list', 'all_model_name_list']
                for col in list_columns:
                    if col in self.data.columns:
                        self.data[col] = self.data[col].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                
                return True
            else:
                print(f"File does not exist: {self.csv_file_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def start_system_monitoring(self):
        """启动系统资源监控线程"""
        def monitor_system_resources():
            while True:
                try:
                    # 获取CPU使用率 (跨所有CPU核心)
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # 获取内存使用情况
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    memory_used = memory.used / (1024 * 1024)  # MB
                    memory_total = memory.total / (1024 * 1024)  # MB
                    
                    # 记录当前时间戳
                    current_time = time.strftime("%H:%M:%S")
                    
                    # 更新数据存储
                    self.system_metrics['timestamps'].append(current_time)
                    self.system_metrics['cpu_percent'].append(cpu_percent)
                    self.system_metrics['memory_percent'].append(memory_percent)
                    self.system_metrics['memory_used'].append(memory_used)
                    self.system_metrics['memory_total'].append(memory_total)
                    
                    # 限制存储数据点数量
                    if len(self.system_metrics['timestamps']) > self.sys_metrics_max_points:
                        for key in self.system_metrics:
                            self.system_metrics[key] = self.system_metrics[key][-self.sys_metrics_max_points:]
                    
                    # 休眠 5 秒
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"Error monitoring system resources: {e}")
                    time.sleep(5)  # 出错时也等待5秒
        
        # 创建并启动监控线程
        monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
        monitor_thread.start()
        print("System resource monitoring started")

    def get_html_template(self):
        """获取 HTML 模板"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Client Metrics Visualization</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 10px 0;
            background-color: #f5f5f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .container-fluid {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-height: 100vh;
        }
        .card {
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .console {
            background-color: #000;
            color: #fff;
            font-family: 'Courier New', monospace;
            height: 280px;  /* 增加高度以匹配图表 */
            overflow-y: auto;
            padding: 8px;
            border-radius: 5px;
            font-size: 0.9rem;
        }
        .chart-container {
            text-align: center;
        }
        .chart-image {
            max-width: 100%;
            max-height: 280px;  /* 限制图表高度 */
            object-fit: contain;
            border-radius: 5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .task-allocation-table {
            font-size: 0.85rem;
            max-height: 150px;
            overflow-y: auto;
        }
        .table {
            margin-bottom: 0;
        }
        .page-title {
            margin-bottom: 10px;
            text-align: center;
            font-size: 1.8rem;
            color: #333;
            font-weight: bold;
        }
        .button-group {
            margin-bottom: 10px;
            text-align: center;
        }
        .button-group button {
            margin: 0 3px;
            padding: 0.25rem 0.5rem;
            font-size: 0.85rem;
        }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .main-section {
            display: flex;
            flex-direction: row;
            gap: 10px;
        }
        .left-column {
            flex: 0 0 52%;
            display: flex;
            flex-direction: column;
        }
        .right-column {
            flex: 0 0 46%;
            display: flex;
            flex-direction: column;
        }
        .card-header {
            padding: 0.5rem 1rem;
        }
        .card-header h4, .card-header h5 {
            margin-bottom: 0;
            font-size: 1rem;
        }
        .card-body {
            padding: 10px;
        }
        .table>:not(caption)>*>* {
            padding: 0.3rem;
        }
        .console .error { color: #ff5555; }
        .console .warning { color: #ffcc00; }
        .console .success { color: #55ff55; }
        .console .info { color: #55ccff; }
        .console .timing { color: #ff55ff; }
        .data-table {
            font-size: 0.9rem;
        }
        #offline-banner {
            display: none;
            background-color: #ff9800;
            color: white;
            text-align: center;
            padding: 10px;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .model-badge {
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .task-id-list {
            max-height: 100px;
            overflow-y: auto;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-family: monospace;
            font-size: 0.85rem;
        }
        .latest-data {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .metric-card {
            flex: 1 0 180px;
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            background-color: #f8f9fa;
            text-align: center;
        }
        .metric-card h6 {
            margin-bottom: 5px;
            color: #6c757d;
            font-size: 0.9rem;
        }
        .metric-card .value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #343a40;
        }
        .metric-card.highlight {
            background-color: #e2f3ff;
            border-color: #90caf9;
        }
        @media (max-width: 992px) {
            .charts-and-console {
                flex-direction: column;
            }
        }
        
        /* 新增系统资源卡片样式 */
        .system-metrics-container {
            display: flex;
            justify-content: space-between;
            gap: 15px;
        }
        .system-metric-card {
            flex: 1;
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            position: relative;
        }
        .system-metric-title {
            color: #6c757d;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        .system-metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .cpu-metric {
            color: #007bff;
        }
        .memory-metric {
            color: #dc3545;
        }
        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background-color: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .cpu-bar {
            background-color: #007bff;
        }
        .memory-bar {
            background-color: #dc3545;
        }
        .system-metrics-details {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div id="offline-banner">
        <strong>You are viewing cached data in offline mode</strong>
    </div>
    
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                        <h4 class="mb-0">Client Metrics Dashboard</h4>
                        <div>
                            <span class="badge bg-info me-2">Sequence: <span id="sequence-label">N/A</span></span>
                            <button id="refresh-btn" class="btn btn-sm btn-light me-2">Refresh Data</button>
                            <div class="form-check form-switch d-inline-block me-2">
                                <input class="form-check-input" type="checkbox" id="auto-refresh" checked>
                                <label class="form-check-label text-white" for="auto-refresh">Auto Refresh</label>
                            </div>
                            <button id="export-btn" class="btn btn-sm btn-success me-2">Export Data</button>
                            <button id="save-snapshot-btn" class="btn btn-sm btn-warning">Save Snapshot</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <!-- 模型信息卡片 -->
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header bg-info text-white">
                                        <h5 class="mb-0">Model Information</h5>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <h6>Current Model:</h6>
                                                <div id="current-model" class="mb-3 fs-5 fw-bold text-primary">N/A</div>
                                                
                                                <h6>Available Models:</h6>
                                                <div id="all-models" class="mb-3">
                                                    <!-- 模型标签将在这里显示 -->
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <h6>Task IDs:</h6>
                                                <div id="task-id-list" class="task-id-list">
                                                    <!-- 任务ID将在这里显示 -->
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- 性能指标卡片 -->
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header bg-success text-white">
                                        <h5 class="mb-0">Performance Metrics</h5>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-3">
                                                <div class="text-center">
                                                    <h6>Total Batches</h6>
                                                    <h3 id="total-batches">0</h3>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="text-center">
                                                    <h6>Total Tasks</h6>
                                                    <h3 id="total-tasks">0</h3>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="text-center">
                                                    <h6>Avg Accuracy</h6>
                                                    <h3 id="avg-accuracy">0.0000</h3>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="text-center">
                                                    <h6>Avg Time (ms)</h6>
                                                    <h3 id="avg-time">0.0000</h3>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 新增: 系统资源监控卡片 -->
        <div class="row mb-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-danger text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">System Resources</h5>
                        <span class="badge bg-light text-dark" id="system-timestamp">Monitoring...</span>
                    </div>
                    <div class="card-body">
                        <div class="system-metrics-container">
                            <div class="system-metric-card">
                                <div class="system-metric-title">CPU Usage</div>
                                <div class="system-metric-value cpu-metric" id="cpu-percent">0%</div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill cpu-bar" id="cpu-bar" style="width: 0%"></div>
                                </div>
                                <div class="system-metrics-details" id="cpu-details">0% of all cores</div>
                            </div>
                            <div class="system-metric-card">
                                <div class="system-metric-title">Memory Usage</div>
                                <div class="system-metric-value memory-metric" id="memory-percent">0%</div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill memory-bar" id="memory-bar" style="width: 0%"></div>
                                </div>
                                <div class="system-metrics-details" id="memory-details">0 MB / 0 MB</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 最新一次运行数据详细信息 -->
        <div class="row mb-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-info text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Latest Run Details</h5>
                        <span class="badge bg-light text-dark" id="latest-run-timestamp">No Data</span>
                    </div>
                    <div class="card-body">
                        <div class="latest-data" id="latest-metrics">
                            <!-- 最新指标将在这里显示 -->
                            <div class="metric-card">
                                <h6>Sequence</h6>
                                <div class="value" id="latest-sequence">-</div>
                            </div>
                            <div class="metric-card highlight">
                                <h6>Time (ms)</h6>
                                <div class="value" id="latest-time">-</div>
                            </div>
                            <div class="metric-card highlight">
                                <h6>Accuracy</h6>
                                <div class="value" id="latest-accuracy">-</div>
                            </div>
                            <div class="metric-card highlight">
                                <h6>Utility Score</h6>
                                <div class="value" id="latest-throughput">-</div>
                            </div>
                            <div class="metric-card">
                                <h6>Current Tasks</h6>
                                <div class="value" id="latest-tasks">-</div>
                            </div>
                            <div class="metric-card">
                                <h6>Current Batches</h6>
                                <div class="value" id="latest-batches">-</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 修改: 将图表和控制台放在同一行 -->
        <div class="row">
            <div class="col-12">
                <div class="row charts-and-console">
                    <!-- 左侧：关键性能图表 -->
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0">Key Performance Charts</h5>
                            </div>
                            <div class="card-body p-2">
                                <div class="chart-container">
                                    <img id="main-chart" class="chart-image" src="" alt="Loading charts...">
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 右侧：命令输出 -->
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-secondary text-white d-flex justify-content-between align-items-center">
                                <h5 class="mb-0">Command Output</h5>
                                <div>
                                    <button id="clear-console" class="btn btn-sm btn-light me-2">Clear</button>
                                    <div class="form-check form-switch d-inline-block">
                                        <input class="form-check-input" type="checkbox" id="auto-scroll" checked>
                                        <label class="form-check-label text-white" for="auto-scroll">Auto Scroll</label>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body p-0">
                                <div id="console" class="console"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 新增: 系统资源图表 -->
        <div class="row mb-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-danger text-white">
                        <h5 class="mb-0">System Resource Charts</h5>
                    </div>
                    <div class="card-body p-2">
                        <div class="chart-container">
                            <img id="system-chart" class="chart-image" src="" alt="Loading system charts...">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h5 class="mb-0">Data Records</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover table-sm data-table">
                                <thead>
                                    <tr>
                                        <th>Sequence</th>
                                        <th>Model</th>
                                        <th>Time (ms)</th>
                                        <th>Accuracy</th>
                                        <th>Utility Score</th>
                                        <th>Batches</th>
                                        <th>Tasks</th>
                                        <th>Total Acc</th>
                                        <th>Total Time</th>
                                    </tr>
                                </thead>
                                <tbody id="data-table-body">
                                    <!-- Data rows will be inserted here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let currentData = null;
        let currentStats = null;
        let currentSequence = null;
        let isOfflineMode = false;
        let consoleLines = [];
        let autoScroll = true;
        let autoRefresh = true;
        let refreshInterval = null;
        let systemRefreshInterval = null;
        
        // DOM elements
        const offlineBanner = document.getElementById('offline-banner');
        const sequenceLabel = document.getElementById('sequence-label');
        const totalBatches = document.getElementById('total-batches');
        const totalTasks = document.getElementById('total-tasks');
        const avgAccuracy = document.getElementById('avg-accuracy');
        const avgTime = document.getElementById('avg-time');
        const dataTableBody = document.getElementById('data-table-body');
        const consoleOutput = document.getElementById('console');
        const refreshBtn = document.getElementById('refresh-btn');
        const exportBtn = document.getElementById('export-btn');
        const saveSnapshotBtn = document.getElementById('save-snapshot-btn');
        const autoRefreshCheckbox = document.getElementById('auto-refresh');
        const autoScrollCheckbox = document.getElementById('auto-scroll');
        const clearConsoleBtn = document.getElementById('clear-console');
        const mainChart = document.getElementById('main-chart');
        const currentModel = document.getElementById('current-model');
        const allModels = document.getElementById('all-models');
        const taskIdList = document.getElementById('task-id-list');
        
        // 系统资源相关元素
        const systemChart = document.getElementById('system-chart');
        const cpuPercent = document.getElementById('cpu-percent');
        const memoryPercent = document.getElementById('memory-percent');
        const cpuBar = document.getElementById('cpu-bar');
        const memoryBar = document.getElementById('memory-bar');
        const cpuDetails = document.getElementById('cpu-details');
        const memoryDetails = document.getElementById('memory-details');
        const systemTimestamp = document.getElementById('system-timestamp');
        
        // Latest run elements
        const latestRunTimestamp = document.getElementById('latest-run-timestamp');
        const latestSequence = document.getElementById('latest-sequence');
        const latestTime = document.getElementById('latest-time');
        const latestAccuracy = document.getElementById('latest-accuracy');
        const latestThroughput = document.getElementById('latest-throughput');
        const latestTasks = document.getElementById('latest-tasks');
        const latestBatches = document.getElementById('latest-batches');
        
        // Initialize app
        function initializeApp() {
            // Check initial online status
            updateOnlineStatus();
            
            // Set up online/offline event listeners
            window.addEventListener('online', updateOnlineStatus);
            window.addEventListener('offline', updateOnlineStatus);
            
            // Set up event listeners
            refreshBtn.addEventListener('click', refreshData);
            exportBtn.addEventListener('click', exportData);
            saveSnapshotBtn.addEventListener('click', saveSnapshot);
            autoRefreshCheckbox.addEventListener('change', function() {
                autoRefresh = this.checked;
                if (autoRefresh) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
                fetch('/toggle_refresh');
            });
            autoScrollCheckbox.addEventListener('change', function() {
                autoScroll = this.checked;
            });
            clearConsoleBtn.addEventListener('click', function() {
                consoleOutput.innerHTML = '';
                consoleLines = [];
            });
            
            // Set up Server-Sent Events for console output
            setupConsoleEventSource();
            
            // Start auto refresh
            startAutoRefresh();
            
            // Start system metrics refresh
            startSystemMetricsRefresh();
            
            // Initial data load
            refreshData();
            refreshSystemMetrics();
        }
        
        // Update online status
        function updateOnlineStatus() {
            isOfflineMode = !navigator.onLine;
            if (isOfflineMode) {
                offlineBanner.style.display = 'block';
                loadFromCache();
            } else {
                offlineBanner.style.display = 'none';
                refreshData();
            }
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
        
        // Stop auto refresh
        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
        }
        
        // Start system metrics refresh
        function startSystemMetricsRefresh() {
            if (systemRefreshInterval) {
                clearInterval(systemRefreshInterval);
            }
            systemRefreshInterval = setInterval(refreshSystemMetrics, 3000);
        }
        
        // Refresh system metrics
        function refreshSystemMetrics() {
            if (!navigator.onLine) {
                // 离线模式下不刷新系统指标
                return;
            }
            
            fetch('/system_metrics')
                .then(response => response.json())
                .then(data => {
                    // 更新系统资源UI
                    updateSystemMetricsUI(data.latest);
                    
                    // 更新系统资源图表
                    if (data.history && data.history.timestamps && data.history.timestamps.length > 0) {
                        fetch('/charts')
                            .then(response => response.json())
                            .then(chartData => {
                                if (chartData.system_chart) {
                                    systemChart.src = 'data:image/png;base64,' + chartData.system_chart;
                                    localStorage.setItem('cachedSystemChart', chartData.system_chart);
                                }
                            })
                            .catch(error => {
                                console.error('Error fetching system charts:', error);
                                // 尝试加载缓存的图表
                                const cachedChart = localStorage.getItem('cachedSystemChart');
                                if (cachedChart) {
                                    systemChart.src = 'data:image/png;base64,' + cachedChart;
                                }
                            });
                    }
                })
                .catch(error => {
                    console.error('Error fetching system metrics:', error);
                });
        }
        
        // Update system metrics UI
        function updateSystemMetricsUI(metrics) {
            if (!metrics) return;
            
            // 更新 CPU 使用率
            cpuPercent.textContent = `${metrics.cpu_percent.toFixed(1)}%`;
            cpuBar.style.width = `${metrics.cpu_percent}%`;
            cpuDetails.textContent = `${metrics.cpu_percent.toFixed(1)}% of all cores`;
            
            // 更新内存使用率
            memoryPercent.textContent = `${metrics.memory_percent.toFixed(1)}%`;
            memoryBar.style.width = `${metrics.memory_percent}%`;
            memoryDetails.textContent = `${metrics.memory_used.toFixed(0)} MB / ${metrics.memory_total.toFixed(0)} MB`;
            
            // 更新时间戳
            const now = new Date();
            systemTimestamp.textContent = `Last Update: ${now.toLocaleTimeString()}`;
            
            // 当内存或CPU使用率过高时，添加警告样式
            if (metrics.cpu_percent > 80) {
                cpuPercent.style.color = '#dc3545';
            } else {
                cpuPercent.style.color = '#007bff';
            }
            
            if (metrics.memory_percent > 80) {
                memoryPercent.style.color = '#dc3545';
            } else {
                memoryPercent.style.color = '#dc3545';
            }
        }
        
        // Refresh data
        function refreshData() {
            if (!navigator.onLine) {
                loadFromCache();
                return;
            }
            
            // Fetch data
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    if (data.data) {
                        currentData = data.data;
                        currentStats = data.stats;
                        currentSequence = data.sequence;
                        
                        updateDataTable(currentData);
                        updateStats(currentStats);
                        updateModelInfo(currentData[currentData.length - 1], currentStats);
                        updateLatestRunData(currentData[currentData.length - 1]); // Update latest run data
                        sequenceLabel.textContent = currentSequence;
                        
                        // Cache the data
                        cacheCurrentState();
                    }
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    loadFromCache();
                });
            
            // Fetch charts
            fetch('/charts')
                .then(response => response.json())
                .then(data => {
                    if (data.main_chart) {
                        mainChart.src = 'data:image/png;base64,' + data.main_chart;
                        localStorage.setItem('cachedChart', data.main_chart);
                    }
                    
                    if (data.system_chart) {
                        systemChart.src = 'data:image/png;base64,' + data.system_chart;
                        localStorage.setItem('cachedSystemChart', data.system_chart);
                    }
                })
                .catch(error => {
                    console.error('Error fetching charts:', error);
                    // Try to load cached chart
                    const cachedChart = localStorage.getItem('cachedChart');
                    if (cachedChart) {
                        mainChart.src = 'data:image/png;base64,' + cachedChart;
                    }
                    const cachedSystemChart = localStorage.getItem('cachedSystemChart');
                    if (cachedSystemChart) {
                        systemChart.src = 'data:image/png;base64,' + cachedSystemChart;
                    }
                });
        }
        
        // Update latest run data
        function updateLatestRunData(latestData) {
            if (!latestData) return;
            
            // Format timestamp
            const now = new Date();
            const timestamp = now.toLocaleTimeString();
            latestRunTimestamp.textContent = `Last Updated: ${timestamp}`;
            
            // Update metrics
            latestSequence.textContent = latestData.sequence;
            latestTime.textContent = parseFloat(latestData.single_batch_time_consumption).toFixed(2);
            latestAccuracy.textContent = parseFloat(latestData.average_batch_accuracy_score_per_batch).toFixed(4);
            latestThroughput.textContent = parseFloat(latestData.avg_throughput_score_per_batch).toFixed(2);
            latestTasks.textContent = latestData.client_task_num_batch 
                ? (Array.isArray(latestData.client_task_num_batch) 
                   ? latestData.client_task_num_batch.length 
                   : latestData.client_task_num_batch)
                : '0';
            latestBatches.textContent = latestData.client_sum_batch_num || '0';
        }
        
        // Export data
        function exportData() {
            window.location.href = '/export_data';
        }
        
        // Save snapshot
        function saveSnapshot() {
            const name = prompt('Enter a name for this snapshot:', 'Snapshot_' + new Date().toISOString().replace(/[:.]/g, '-'));
            if (!name) return;
            
            try {
                const snapshot = {
                    name: name,
                    date: new Date().toISOString(),
                    data: currentData,
                    stats: currentStats,
                    sequence: currentSequence,
                    chart: localStorage.getItem('cachedChart'),
                    systemChart: localStorage.getItem('cachedSystemChart'),
                    console: consoleLines
                };
                
                // Get existing snapshots
                let snapshots = JSON.parse(localStorage.getItem('snapshots') || '[]');
                
                // Add new snapshot
                snapshots.push(snapshot);
                
                // Save to localStorage
                localStorage.setItem('snapshots', JSON.stringify(snapshots));
                
                alert(`Snapshot "${name}" saved successfully!`);
            } catch (error) {
                console.error('Failed to save snapshot:', error);
                alert('Failed to save snapshot: ' + error.message);
            }
        }
        
        // Update model information
        function updateModelInfo(latestData, stats) {
            // 更新当前模型
            if (stats.current_model) {
                currentModel.textContent = stats.current_model;
            }
            
            // 更新所有可用模型列表
            allModels.innerHTML = '';
            if (stats.all_models && stats.all_models.length > 0) {
                stats.all_models.forEach(model => {
                    const badge = document.createElement('span');
                    badge.className = 'badge bg-secondary model-badge';
                    badge.textContent = model;
                    allModels.appendChild(badge);
                });
            } else {
                allModels.textContent = 'No models available';
            }
            
            // 更新任务 ID 列表
            taskIdList.innerHTML = '';
            if (latestData.task_id_list && latestData.task_id_list.length > 0) {
                latestData.task_id_list.forEach(taskId => {
                    const taskElement = document.createElement('div');
                    taskElement.textContent = taskId;
                    taskIdList.appendChild(taskElement);
                });
            } else {
                taskIdList.textContent = 'No tasks';
            }
        }
        
        // Update data table
        function updateDataTable(data) {
            dataTableBody.innerHTML = '';
            
            // 创建数据副本以避免修改原始数据
            const sortedData = [...data];
            
            // 按序列号倒序排序数据
            sortedData.sort((a, b) => b.sequence - a.sequence);
            
            sortedData.forEach(row => {
                const tr = document.createElement('tr');
                
                tr.innerHTML = `
                    <td>${row.sequence}</td>
                    <td>${row.current_using_model_name || 'N/A'}</td>
                    <td>${parseFloat(row.single_batch_time_consumption).toFixed(2)}</td>
                    <td>${parseFloat(row.average_batch_accuracy_score_per_batch).toFixed(4)}</td>
                    <td>${parseFloat(row.avg_throughput_score_per_batch).toFixed(2)}</td>
                    <td>${row.client_sum_batch_num}</td>
                    <td>${row.client_sum_task_num}</td>
                    <td>${parseFloat(row.client_sum_batch_accuravy_score).toFixed(2)}</td>
                    <td>${parseFloat(row.client_sum_batch_time_consumption).toFixed(2)}</td>
                `;
                
                dataTableBody.appendChild(tr);
            });
        }
        
        // Update statistics
        function updateStats(stats) {
            totalBatches.textContent = stats.total_batches;
            totalTasks.textContent = stats.total_tasks;
            avgAccuracy.textContent = stats.avg_accuracy.toFixed(4);
            avgTime.textContent = stats.avg_time.toFixed(4);
        }
        
        // Append text to console
        function appendToConsole(text) {
            const line = document.createElement('div');
            
            // Apply different colors based on content
            if (text.toLowerCase().includes('error') || text.toLowerCase().includes('failed')) {
                line.className = 'error';
            } else if (text.toLowerCase().includes('warning')) {
                line.className = 'warning';
            } else if (text.toLowerCase().includes('success') || text.toLowerCase().includes('completed')) {
                line.className = 'success';
            } else if (text.toLowerCase().includes('loading') || text.toLowerCase().includes('tokenizing')) {
                line.className = 'info';
            } else if (text.includes('llama_print_timings')) {
                line.className = 'timing';
            }
            
            line.textContent = text;
            consoleOutput.appendChild(line);
            
            // Store console line for caching
            consoleLines.push({
                text: text,
                className: line.className
            });
            
            // Limit console lines to prevent excessive memory usage
            if (consoleLines.length > 1000) {
                consoleLines.shift();
            }
            
            // Auto-scroll if enabled
            if (autoScroll) {
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }
        }
        
        // Cache current state
        function cacheCurrentState() {
            try {
                const state = {
                    timestamp: Date.now(),
                    data: currentData,
                    stats: currentStats,
                    sequence: currentSequence,
                    console: consoleLines
                };
                
                localStorage.setItem('currentState', JSON.stringify(state));
            } catch (error) {
                console.error('Failed to cache state:', error);
            }
        }
        
        // Load from cache
        function loadFromCache() {
            try {
                const cachedState = localStorage.getItem('currentState');
                if (cachedState) {
                    const state = JSON.parse(cachedState);
                    
                    currentData = state.data;
                    currentStats = state.stats;
                    currentSequence = state.sequence;
                    
                    if (currentData && currentData.length > 0) {
                        updateDataTable(currentData);
                        updateLatestRunData(currentData[currentData.length - 1]); // Update latest run data
                        if (currentStats) {
                            updateStats(currentStats);
                            updateModelInfo(currentData[currentData.length - 1], currentStats);
                        }
                    }
                    if (currentSequence) sequenceLabel.textContent = currentSequence;
                    
                    // Try to load cached chart
                    const cachedChart = localStorage.getItem('cachedChart');
                    if (cachedChart) {
                        mainChart.src = 'data:image/png;base64,' + cachedChart;
                    }
                    
                    const cachedSystemChart = localStorage.getItem('cachedSystemChart');
                    if (cachedSystemChart) {
                        systemChart.src = 'data:image/png;base64,' + cachedSystemChart;
                    }
                    
                    // Restore console output
                    if (state.console && state.console.length > 0) {
                        consoleOutput.innerHTML = '';
                        consoleLines = state.console;
                        
                        consoleLines.forEach(line => {
                            const lineElement = document.createElement('div');
                            if (line.className) {
                                lineElement.className = line.className;
                            }
                            lineElement.textContent = line.text;
                            consoleOutput.appendChild(lineElement);
                        });
                        
                        if (autoScroll) {
                            consoleOutput.scrollTop = consoleOutput.scrollHeight;
                        }
                    }
                }
            } catch (error) {
                console.error('Failed to load from cache:', error);
            }
        }
        
        // Initialize the app when DOM is loaded
        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
        """

    def run(self):
        """运行 Web 可视化器"""
        try:
            print(f"Starting web server on port {self.port}...")
            print(f"Open http://localhost:{self.port} in your browser")
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        finally:
            # 确保观察者停止
            if hasattr(self, 'observer'):
                self.observer.stop()
                self.observer.join()


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('metrics/client', exist_ok=True)
    
    # 创建并运行 Web 可视化器
    visualizer = WebMetricsVisualizer(port=12347)
    visualizer.run()