import os
import time
import json
import pandas as pd
import threading
import queue
from flask import Flask, render_template_string, jsonify, Response, send_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，无需显示器
import matplotlib.pyplot as plt
import io
import base64
import logging
import datetime
import ast

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CenterCSVMonitor:
    def __init__(self, csv_file_path='metrics/core/core_metrics.csv', port=12348):
        """
        初始化中心节点CSV监控系统
        
        Args:
            csv_file_path: CSV文件路径
            port: Web服务器端口
        """
        self.csv_file_path = csv_file_path
        self.port = port
        self.data = None
        self.edge_nodes = {}  # 存储边缘节点信息
        self.task_history = []  # 存储任务分发历史
        self.output_queue = queue.Queue()  # 存储日志输出
        self.last_modified_time = 0
        
        # 创建锁以保护共享数据
        self.data_lock = threading.Lock()
        self.edge_nodes_lock = threading.Lock()
        self.task_history_lock = threading.Lock()
        
        # 创建 Flask 应用
        self.app = Flask(__name__)
        
        # 设置路由
        self.setup_routes()
        
        # 设置文件监控
        self.setup_file_monitoring()
        
        # 初始加载数据
        self.load_data()

    def setup_routes(self):
        """设置 Flask 路由"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/data')
        def get_data():
            """获取所有CSV数据"""
            with self.data_lock:
                if self.data is None or self.data.empty:
                    return jsonify({'data': None})
                
                # 转换数据为JSON格式
                data_json = self.data.to_dict(orient='records')
                return jsonify({'data': data_json})
        
        @self.app.route('/edge_nodes')
        def get_edge_nodes():
            """获取边缘节点信息"""
            with self.edge_nodes_lock:
                return jsonify(self.edge_nodes)
        
        @self.app.route('/task_history')
        def get_task_history():
            """获取任务分发历史"""
            with self.task_history_lock:
                return jsonify(self.task_history)
        
        @self.app.route('/charts')
        def get_charts():
            """获取图表数据"""
            if self.data is None or self.data.empty:
                return jsonify({'error': 'No data available'})
            
            # 生成图表
            chart_images = self.generate_chart_images()
            
            return jsonify(chart_images)
        
        @self.app.route('/console_stream')
        def console_stream():
            """使用服务器发送事件流式传输日志输出"""
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
        
        @self.app.route('/export_data')
        def export_data():
            """导出数据为CSV文件"""
            if self.data is None or self.data.empty:
                return jsonify({'error': 'No data available'})
            
            # 创建临时文件
            temp_file = f'center_metrics_export_{time.strftime("%Y%m%d_%H%M%S")}.csv'
            self.data.to_csv(temp_file, index=False)
            
            # 发送文件
            return send_file(temp_file, as_attachment=True, download_name=temp_file)

    def setup_file_monitoring(self):
        """设置CSV文件监控"""
        try:
            # 创建文件系统事件处理程序
            class FileChangeHandler(FileSystemEventHandler):
                def __init__(self, callback):
                    self.callback = callback
                    
                def on_modified(self, event):
                    if not event.is_directory and event.src_path.endswith(self.callback.csv_file_path):
                        self.callback.load_data()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            
            # 设置观察者
            self.event_handler = FileChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, path=os.path.dirname(self.csv_file_path), recursive=False)
            self.observer.start()
            
            self.log_message(f"File monitoring set up for {self.csv_file_path}")
        except Exception as e:
            self.log_message(f"Error setting up file monitoring: {e}")

    def load_data(self):
        """加载CSV数据"""
        try:
            if os.path.exists(self.csv_file_path):
                # 检查文件是否已修改
                current_modified_time = os.path.getmtime(self.csv_file_path)
                if current_modified_time == self.last_modified_time:
                    return  # 文件未修改，无需重新加载
                
                self.last_modified_time = current_modified_time
                
                # 读取CSV文件
                with self.data_lock:
                    self.data = pd.read_csv(self.csv_file_path)
                
                # 处理数据
                self.process_data()
                
                self.log_message(f"Loaded data from {self.csv_file_path}, {len(self.data)} rows")
                return True
            else:
                self.log_message(f"File does not exist: {self.csv_file_path}")
                return False
        except Exception as e:
            self.log_message(f"Error loading data: {e}")
            return False

    def process_data(self):
        """处理CSV数据，提取边缘节点信息和任务分发历史"""
        if self.data is None or self.data.empty:
            return
        
        try:
            # 处理每一行数据
            for _, row in self.data.iterrows():
                # 处理边缘节点观测数据
                self.process_edge_observations(row)
                
                # 处理任务分发历史
                self.process_task_allocation(row)
        except Exception as e:
            self.log_message(f"Error processing data: {e}")

    def process_edge_observations(self, row):
        """处理边缘节点观测数据"""
        try:
            sequence = row['sequence']
            edge_observations_str = row['edge_observations']
            
            # 解析JSON字符串
            edge_observations = json.loads(edge_observations_str.replace('""', '"'))
            
            with self.edge_nodes_lock:
                for edge_id, observation in edge_observations.items():
                    # 如果节点不存在，创建它
                    if edge_id not in self.edge_nodes:
                        self.edge_nodes[edge_id] = {
                            "id": edge_id,
                            "status": "connected",
                            "last_seen": time.time(),
                            "observations": [],
                            "current_observation": {}
                        }
                    
                    # 添加观测数据，包含序列号
                    observation_with_sequence = observation.copy()
                    observation_with_sequence['sequence'] = sequence
                    observation_with_sequence['timestamp'] = time.time()
                    
                    # 更新当前观测数据
                    self.edge_nodes[edge_id]["current_observation"] = observation
                    self.edge_nodes[edge_id]["last_seen"] = time.time()
                    
                    # 添加到历史观测数据
                    self.edge_nodes[edge_id]["observations"].append(observation_with_sequence)
                    
                    # 限制历史记录数量
                    if len(self.edge_nodes[edge_id]["observations"]) > 100:
                        self.edge_nodes[edge_id]["observations"] = self.edge_nodes[edge_id]["observations"][-100:]
        except Exception as e:
            self.log_message(f"Error processing edge observations: {e}")

    def process_task_allocation(self, row):
        """处理任务分发历史"""
        try:
            sequence = row['sequence']
            task_inference_time = row['task_inference_time']
            tasks_allocation_str = row['tasks_allocation']
            
            # 解析JSON字符串
            tasks_allocation = json.loads(tasks_allocation_str.replace('""', '"'))
            
            # 按边缘节点ID组织任务
            tasks_by_edge = {}
            for task in tasks_allocation:
                edge_id = task['edge_id']
                if edge_id not in tasks_by_edge:
                    tasks_by_edge[edge_id] = []
                tasks_by_edge[edge_id].append(task)
            
            with self.task_history_lock:
                for edge_id, tasks in tasks_by_edge.items():
                    # 创建任务分发记录
                    record = {
                        "timestamp": time.time(),
                        "edge_id": edge_id,
                        "sequence": sequence,
                        "task_count": len(tasks),
                        "task_inference_time": task_inference_time,
                        "tasks": [{"id": task["task_id"], "type": task["task_type"]} for task in tasks]
                    }
                    
                    self.task_history.append(record)
                
                # 限制历史记录数量
                if len(self.task_history) > 100:
                    self.task_history = self.task_history[-100:]
        except Exception as e:
            self.log_message(f"Error processing task allocation: {e}")

    def generate_chart_images(self):
        """生成图表"""
        chart_images = {}
        
        try:
            # 为每个边缘节点生成性能图表
            with self.edge_nodes_lock:
                for edge_id, node_data in self.edge_nodes.items():
                    if not node_data["observations"]:
                        continue
                    
                    # 提取数据
                    observations = node_data["observations"]
                    sequences = [obs["sequence"] for obs in observations]
                    accuracies = [obs["accuracy"] for obs in observations]
                    latencies = [obs["latency"] for obs in observations]
                    throughputs = [obs["avg_throughput"] for obs in observations]
                    
                    # 创建图表
                    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
                    
                    # 准确率图表
                    ax[0].plot(sequences, accuracies, 'g-o', label='Accuracy')
                    ax[0].set_title(f'Edge Node {edge_id} - Accuracy')
                    ax[0].set_xlabel('Sequence')
                    ax[0].set_ylabel('Accuracy')
                    ax[0].grid(True)
                    
                    # 延迟图表
                    ax[1].plot(sequences, latencies, 'b-o', label='Latency')
                    ax[1].set_title(f'Edge Node {edge_id} - Latency')
                    ax[1].set_xlabel('Sequence')
                    ax[1].set_ylabel('Latency')
                    ax[1].grid(True)
                    
                    # 吞吐量图表
                    ax[2].plot(sequences, throughputs, 'r-o', label='Throughput')
                    ax[2].set_title(f'Edge Node {edge_id} - Throughput')
                    ax[2].set_xlabel('Sequence')
                    ax[2].set_ylabel('Throughput')
                    ax[2].grid(True)
                    
                    # 调整布局
                    fig.tight_layout()
                    
                    # 将图表转换为base64编码的图像
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    chart_images[f'edge_node_{edge_id}'] = img_str
                    
                    # 关闭图表以释放内存
                    plt.close(fig)
            
            # 生成任务分发历史图表
            with self.task_history_lock:
                if self.task_history:
                    # 按序列号组织任务数量
                    sequences = []
                    task_counts = {}
                    
                    for record in self.task_history:
                        seq = record["sequence"]
                        edge_id = record["edge_id"]
                        
                        if seq not in sequences:
                            sequences.append(seq)
                        
                        if edge_id not in task_counts:
                            task_counts[edge_id] = {}
                        
                        if seq not in task_counts[edge_id]:
                            task_counts[edge_id][seq] = 0
                        
                        task_counts[edge_id][seq] += record["task_count"]
                    
                    # 创建图表
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 为每个边缘节点绘制任务数量
                    for edge_id in task_counts:
                        counts = [task_counts[edge_id].get(seq, 0) for seq in sequences]
                        ax.plot(sequences, counts, 'o-', label=f'Edge Node {edge_id}')
                    
                    ax.set_title('Task Distribution History')
                    ax.set_xlabel('Sequence')
                    ax.set_ylabel('Task Count')
                    ax.legend()
                    ax.grid(True)
                    
                    # 将图表转换为base64编码的图像
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    chart_images['task_history'] = img_str
                    
                    # 关闭图表以释放内存
                    plt.close(fig)
            
        except Exception as e:
            self.log_message(f"Error generating chart images: {e}")
            chart_images['error'] = str(e)
        
        return chart_images

    def log_message(self, message):
        """记录消息到日志和队列"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        logging.info(message)
        self.output_queue.put(formatted_message)

    def get_html_template(self):
        """获取HTML模板"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Center Node CSV Monitor</title>
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
            height: 300px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 5px;
        }
        .data-table {
            font-size: 0.9rem;
        }
        .edge-node-card {
            transition: all 0.3s;
        }
        .edge-node-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
        .task-history-item {
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .refresh-timestamp {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 10px;
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
        .tab-content {
            padding: 20px;
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-top: 0;
            border-radius: 0 0 5px 5px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                        <h4 class="mb-0">Center Node CSV Monitor</h4>
                        <div>
                            <button id="refresh-btn" class="btn btn-sm btn-light me-2">Refresh Data</button>
                            <div class="form-check form-switch d-inline-block me-2">
                                <input class="form-check-input" type="checkbox" id="auto-refresh" checked>
                                <label class="form-check-label text-white" for="auto-refresh">Auto Refresh</label>
                            </div>
                            <button id="export-btn" class="btn btn-sm btn-success">Export Data</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="edge-nodes-tab" data-bs-toggle="tab" data-bs-target="#edge-nodes" type="button" role="tab" aria-controls="edge-nodes" aria-selected="true">Edge Nodes</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="task-history-tab" data-bs-toggle="tab" data-bs-target="#task-history" type="button" role="tab" aria-controls="task-history" aria-selected="false">Task History</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="charts-tab" data-bs-toggle="tab" data-bs-target="#charts" type="button" role="tab" aria-controls="charts" aria-selected="false">Charts</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="raw-data-tab" data-bs-toggle="tab" data-bs-target="#raw-data" type="button" role="tab" aria-controls="raw-data" aria-selected="false">Raw Data</button>
                    </li>
                </ul>
                <div class="tab-content" id="myTabContent">
                    <!-- Edge Nodes Tab -->
                    <div class="tab-pane fade show active" id="edge-nodes" role="tabpanel" aria-labelledby="edge-nodes-tab">
                        <div id="edge-nodes-container" class="row">
                            <!-- Edge nodes will be displayed here -->
                            <div class="col-12 text-center">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <p>Loading edge nodes data...</p>
                            </div>
                        </div>
                    </div>

                    <!-- Task History Tab -->
                    <div class="tab-pane fade" id="task-history" role="tabpanel" aria-labelledby="task-history-tab">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">Task Distribution History</h5>
                            </div>
                            <div class="card-body p-0">
                                <div id="task-history-container" class="list-group list-group-flush">
                                    <!-- Task history will be displayed here -->
                                    <div class="list-group-item text-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p>Loading task history data...</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Charts Tab -->
                    <div class="tab-pane fade" id="charts" role="tabpanel" aria-labelledby="charts-tab">
                        <div class="card">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0">Performance Charts</h5>
                            </div>
                            <div class="card-body">
                                <div id="charts-container">
                                    <!-- Charts will be displayed here -->
                                    <div class="text-center">
                                        <div class="spinner-border text-success" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p>Loading charts data...</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Raw Data Tab -->
                    <div class="tab-pane fade" id="raw-data" role="tabpanel" aria-labelledby="raw-data-tab">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0">Raw CSV Data</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped table-hover data-table">
                                        <thead id="raw-data-header">
                                            <!-- Table header will be inserted here -->
                                        </thead>
                                        <tbody id="raw-data-body">
                                            <!-- Table data will be inserted here -->
                                            <tr>
                                                <td colspan="4" class="text-center">
                                                    <div class="spinner-border text-secondary" role="status">
                                                        <span class="visually-hidden">Loading...</span>
                                                    </div>
                                                    <p>Loading raw data...</p>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-secondary text-white d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">System Log</h5>
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

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let autoRefresh = true;
        let autoScroll = true;
        let refreshInterval = null;
        
        // DOM elements
        const edgeNodesContainer = document.getElementById('edge-nodes-container');
        const taskHistoryContainer = document.getElementById('task-history-container');
        const chartsContainer = document.getElementById('charts-container');
        const rawDataHeader = document.getElementById('raw-data-header');
        const rawDataBody = document.getElementById('raw-data-body');
        const consoleOutput = document.getElementById('console');
        const refreshBtn = document.getElementById('refresh-btn');
        const exportBtn = document.getElementById('export-btn');
        const autoRefreshCheckbox = document.getElementById('auto-refresh');
        const autoScrollCheckbox = document.getElementById('auto-scroll');
        const clearConsoleBtn = document.getElementById('clear-console');
        
        // Initialize app
        function initializeApp() {
            // Set up event listeners
            refreshBtn.addEventListener('click', refreshAllData);
            exportBtn.addEventListener('click', exportData);
            autoRefreshCheckbox.addEventListener('change', function() {
                autoRefresh = this.checked;
                if (autoRefresh) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            });
            autoScrollCheckbox.addEventListener('change', function() {
                autoScroll = this.checked;
            });
            clearConsoleBtn.addEventListener('click', function() {
                consoleOutput.innerHTML = '';
            });
            
            // Set up Server-Sent Events for console output
            setupConsoleEventSource();
            
            // Start auto refresh
            startAutoRefresh();
            
            // Initial data load
            refreshAllData();
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
            refreshInterval = setInterval(refreshAllData, 5000);
        }
        
        // Stop auto refresh
        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
        }
        
        // Refresh all data
        function refreshAllData() {
            fetchEdgeNodes();
            fetchTaskHistory();
            fetchCharts();
            fetchRawData();
        }
        
        // Fetch edge nodes data
        function fetchEdgeNodes() {
            fetch('/edge_nodes')
                .then(response => response.json())
                .then(data => {
                    updateEdgeNodesDisplay(data);
                })
                .catch(error => {
                    console.error('Error fetching edge nodes data:', error);
                });
        }
        
        // Fetch task history data
        function fetchTaskHistory() {
            fetch('/task_history')
                .then(response => response.json())
                .then(data => {
                    updateTaskHistoryDisplay(data);
                })
                .catch(error => {
                    console.error('Error fetching task history data:', error);
                });
        }
        
        // Fetch charts data
        function fetchCharts() {
            fetch('/charts')
                .then(response => response.json())
                .then(data => {
                    updateChartsDisplay(data);
                })
                .catch(error => {
                    console.error('Error fetching charts data:', error);
                });
        }
        
        // Fetch raw data
        function fetchRawData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    updateRawDataDisplay(data);
                })
                .catch(error => {
                    console.error('Error fetching raw data:', error);
                });
        }
        
        // Export data
        function exportData() {
            window.location.href = '/export_data';
        }
        
        // Update edge nodes display
        function updateEdgeNodesDisplay(data) {
            edgeNodesContainer.innerHTML = '';
            
            if (Object.keys(data).length === 0) {
                edgeNodesContainer.innerHTML = '<div class="col-12 text-center"><p>No edge nodes data available</p></div>';
                return;
            }
            
            for (const [nodeId, nodeData] of Object.entries(data)) {
                const observation = nodeData.current_observation || {};
                const accuracy = observation.accuracy || 0;
                const latency = observation.latency || 0;
                const throughput = observation.avg_throughput || 0;
                
                const nodeCard = document.createElement('div');
                nodeCard.className = 'col-md-4 mb-3';
                nodeCard.innerHTML = `
                    <div class="card edge-node-card h-100">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0">Edge Node ${nodeId}</h5>
                        </div>
                        <div class="card-body">
                            <div class="row text-center">
                                <div class="col-4">
                                    <div class="metric-value text-success">${accuracy.toFixed(4)}</div>
                                    <div class="metric-label">Accuracy</div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-value text-primary">${latency.toFixed(1)}</div>
                                    <div class="metric-label">Latency</div>
                                </div>
                                <div class="col-4">
                                    <div class="metric-value text-info">${throughput.toFixed(2)}</div>
                                    <div class="metric-label">Throughput</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                edgeNodesContainer.appendChild(nodeCard);
            }
        }
        
        // Update task history display
        function updateTaskHistoryDisplay(data) {
            taskHistoryContainer.innerHTML = '';
            
            if (data.length === 0) {
                taskHistoryContainer.innerHTML = '<div class="list-group-item text-center"><p>No task distribution history</p></div>';
                return;
            }
            
            // Display only the last 10 task distributions
            const recentTasks = data.slice(-10).reverse();
            
            for (const task of recentTasks) {
                const taskItem = document.createElement('div');
                taskItem.className = 'list-group-item task-history-item';
                
                const taskTime = task.task_inference_time ? task.task_inference_time.toFixed(2) + ' ms' : 'N/A';
                
                taskItem.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Tasks to Edge Node ${task.edge_id}</h5>
                        <small>Sequence: ${task.sequence}</small>
                    </div>
                    <p class="mb-1">
                        <span class="badge bg-primary">${task.task_count} tasks</span>
                        <span class="badge bg-secondary">Inference Time: ${taskTime}</span>
                    </p>
                    <div class="mt-2 small">
                        <strong>Task IDs:</strong>
                        <ul class="mb-0">
                            ${task.tasks.map(t => `<li>ID: ${t.id}, Type: ${t.type}</li>`).join('')}
                        </ul>
                    </div>
                `;
                
                taskHistoryContainer.appendChild(taskItem);
            }
        }
        
        // Update charts display
        function updateChartsDisplay(data) {
            chartsContainer.innerHTML = '';
            
            if (data.error) {
                chartsContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                return;
            }
            
            // Display task history chart if available
            if (data.task_history) {
                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart-container mb-4';
                chartDiv.innerHTML = `
                    <h4>Task Distribution History</h4>
                    <img src="data:image/png;base64,${data.task_history}" class="chart-image" alt="Task Distribution History">
                `;
                chartsContainer.appendChild(chartDiv);
            }
            
            // Display edge node charts
            for (const [key, value] of Object.entries(data)) {
                if (key.startsWith('edge_node_')) {
                    const nodeId = key.replace('edge_node_', '');
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart-container mb-4';
                    chartDiv.innerHTML = `
                        <h4>Edge Node ${nodeId} Performance</h4>
                        <img src="data:image/png;base64,${value}" class="chart-image" alt="Edge Node ${nodeId} Performance">
                    `;
                    chartsContainer.appendChild(chartDiv);
                }
            }
            
            if (chartsContainer.innerHTML === '') {
                chartsContainer.innerHTML = '<div class="alert alert-info">No chart data available</div>';
            }
        }
        
        // Update raw data display
        function updateRawDataDisplay(data) {
            if (!data.data || data.data.length === 0) {
                rawDataHeader.innerHTML = '<tr><th>No data available</th></tr>';
                rawDataBody.innerHTML = '<tr><td>No data available</td></tr>';
                return;
            }
            
            // Get column names from first row
            const columns = Object.keys(data.data[0]);
            
            // Create header
            rawDataHeader.innerHTML = `
                <tr>
                    ${columns.map(col => `<th>${col}</th>`).join('')}
                </tr>
            `;
            
            // Create rows
            rawDataBody.innerHTML = '';
            for (const row of data.data) {
                const tr = document.createElement('tr');
                
                for (const col of columns) {
                    const td = document.createElement('td');
                    
                    // Format cell content based on column type
                    if (typeof row[col] === 'object') {
                        td.textContent = JSON.stringify(row[col]);
                    } else {
                        td.textContent = row[col];
                    }
                    
                    tr.appendChild(td);
                }
                
                rawDataBody.appendChild(tr);
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
            } else if (text.toLowerCase().includes('loaded data')) {
                line.style.color = '#55ff55';
            } else if (text.toLowerCase().includes('processing')) {
                line.style.color = '#55ccff';
            }
            
            consoleOutput.appendChild(line);
            
            // Auto-scroll if enabled
            if (autoScroll) {
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }
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
            print(f"Starting center CSV monitor on port {self.port}...")
            print(f"Open http://localhost:{self.port} in your browser")
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        except Exception as e:
            logging.error(f"Error starting center CSV monitor: {e}")
        finally:
            # 确保观察者停止
            if hasattr(self, 'observer'):
                self.observer.stop()
                self.observer.join()


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('metrics/core', exist_ok=True)
    
    # 创建并运行中心节点CSV监控系统
    monitor = CenterCSVMonitor(csv_file_path='metrics/core/core_metrics.csv', port=12348)
    monitor.run()