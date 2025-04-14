import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import ast
import numpy as np
import queue
import socket
import struct
import json

class MetricsVisualizer:
    def __init__(self, csv_file_path='metrics/client/client_metrics.csv'):
        """
        Initialize metrics visualizer
        
        Args:
            csv_file_path: Path to the CSV file
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.last_modified_time = 0
        
        # Create a queue for command output
        self.output_queue = queue.Queue()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Client Metrics Visualization")
        self.root.geometry("1600x1000")  # Larger window to fit all components
        self.root.configure(bg='white')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for controls and stats
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create control panel in top frame
        self.create_control_panel(top_frame)
        
        # Create data display area
        self.create_data_display(main_frame)
        
        # Create bottom frame that will be split into left and right
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame for charts
        chart_frame = ttk.Frame(bottom_frame)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right frame for console output
        console_frame = ttk.LabelFrame(bottom_frame, text="Command Output")
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
        console_frame.config(width=500)
        
        # Create console output text widget
        self.create_console_output(console_frame)
        
        # Create chart area
        self.create_chart_area(chart_frame)
        
        # Set up file monitoring
        self.setup_file_monitoring()
        
        # Set up socket for command output
        self.setup_command_output_receiver()
        
        # Initial data load
        self.load_data()
        self.update_chart()
        self.update_data_tree()
        
        # Start periodic updates of the console
        self.update_console_output()

    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Title label
        title_label = ttk.Label(control_frame, text="Sequence:", font=("Arial", 12, "bold"))
        title_label.pack(side=tk.LEFT, padx=5)
        
        self.sequence_label = ttk.Label(control_frame, text="N/A", font=("Arial", 12))
        self.sequence_label.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_btn = ttk.Button(control_frame, text="Refresh Data", command=self.load_data_and_update)
        refresh_btn.pack(side=tk.LEFT, padx=20)
        
        # Auto refresh option
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = ttk.Checkbutton(
            control_frame, 
            text="Auto Refresh", 
            variable=self.auto_refresh_var
        )
        auto_refresh_check.pack(side=tk.LEFT, padx=5)
        
        # Display statistics
        self.stats_label = ttk.Label(control_frame, text="")
        self.stats_label.pack(side=tk.RIGHT, padx=5)

    def create_data_display(self, parent):
        """Create data display area for showing data_need_to_record"""
        # Create a frame for data display
        data_frame = ttk.LabelFrame(parent, text="Data Records")
        data_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Create table to display data
        self.data_tree = ttk.Treeview(data_frame)
        
        # Define columns
        self.data_tree["columns"] = (
            "sequence", "time", "accuracy", "throughput", 
            "batch_num", "task_num", "total_accuracy", 
            "total_time", "total_throughput"
        )
        
        # Format columns
        self.data_tree.column("#0", width=0, stretch=tk.NO)  # Hide first column
        self.data_tree.column("sequence", anchor=tk.CENTER, width=80)
        self.data_tree.column("time", anchor=tk.CENTER, width=100)
        self.data_tree.column("accuracy", anchor=tk.CENTER, width=100)
        self.data_tree.column("throughput", anchor=tk.CENTER, width=100)
        self.data_tree.column("batch_num", anchor=tk.CENTER, width=80)
        self.data_tree.column("task_num", anchor=tk.CENTER, width=80)
        self.data_tree.column("total_accuracy", anchor=tk.CENTER, width=100)
        self.data_tree.column("total_time", anchor=tk.CENTER, width=100)
        self.data_tree.column("total_throughput", anchor=tk.CENTER, width=100)
        
        # Define column headings
        self.data_tree.heading("sequence", text="Sequence")
        self.data_tree.heading("time", text="Time (ms)")
        self.data_tree.heading("accuracy", text="Accuracy")
        self.data_tree.heading("throughput", text="Throughput")
        self.data_tree.heading("batch_num", text="Batches")
        self.data_tree.heading("task_num", text="Tasks")
        self.data_tree.heading("total_accuracy", text="Total Acc")
        self.data_tree.heading("total_time", text="Total Time")
        self.data_tree.heading("total_throughput", text="Total Thpt")
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(data_frame, orient="vertical", command=self.data_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.data_tree.configure(yscrollcommand=tree_scroll.set)
        self.data_tree.pack(fill=tk.X, expand=True, padx=5, pady=5)

    def update_data_tree(self):
        """Update data tree with the latest records"""
        if self.data is None or self.data.empty:
            return
        
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add new data (display up to 20 most recent records)
        display_data = self.data.tail(20)
        
        for _, row in display_data.iterrows():
            self.data_tree.insert(
                "", tk.END,
                values=(
                    row.get('sequence', ''),
                    f"{row.get('single_batch_time_consumption', 0):.2f}",
                    f"{row.get('average_batch_accuracy_score_per_batch', 0):.4f}",
                    f"{row.get('avg_throughput_score_per_batch', 0):.2f}",
                    row.get('client_sum_batch_num', 0),
                    row.get('client_sum_task_num', 0),
                    f"{row.get('client_sum_batch_accuravy_score', 0):.2f}",
                    f"{row.get('client_sum_batch_time_consumption', 0):.2f}",
                    f"{row.get('client_sum_batch_throughput_score', 0):.2f}"
                )
            )

    def create_console_output(self, parent):
        """Create enhanced console output area"""
        # Create terminal output frame
        terminal_frame = ttk.Frame(parent)
        terminal_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget with scrollbar using fixed-width font for better terminal display
        self.console_output = scrolledtext.ScrolledText(
            terminal_frame, 
            wrap=tk.WORD, 
            font=("Courier", 10),
            bg="#000000",  # Black background
            fg="#FFFFFF",  # White text
            insertbackground="#FFFFFF"  # White cursor
        )
        self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console_output.config(state=tk.DISABLED)  # Make it read-only
        
        # Create control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Clear button
        clear_btn = ttk.Button(control_frame, text="Clear Console", command=self.clear_console)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto scroll option
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = ttk.Checkbutton(
            control_frame, 
            text="Auto Scroll", 
            variable=self.auto_scroll_var
        )
        auto_scroll_check.pack(side=tk.LEFT, padx=5)

    def clear_console(self):
        """Clear the console output"""
        self.console_output.config(state=tk.NORMAL)
        self.console_output.delete(1.0, tk.END)
        self.console_output.config(state=tk.DISABLED)

    def append_to_console(self, text):
        """Append text to the console output with color formatting"""
        self.console_output.config(state=tk.NORMAL)
        
        # Apply different colors based on content
        if "error" in text.lower() or "failed" in text.lower():
            # Error messages in red
            self.console_output.tag_config("error", foreground="red")
            self.console_output.insert(tk.END, text + "\n", "error")
        elif "warning" in text.lower():
            # Warning messages in yellow
            self.console_output.tag_config("warning", foreground="yellow")
            self.console_output.insert(tk.END, text + "\n", "warning")
        elif "success" in text.lower() or "completed" in text.lower():
            # Success messages in green
            self.console_output.tag_config("success", foreground="green")
            self.console_output.insert(tk.END, text + "\n", "success")
        elif "loading" in text.lower() or "tokenizing" in text.lower():
            # Loading messages in cyan
            self.console_output.tag_config("info", foreground="cyan")
            self.console_output.insert(tk.END, text + "\n", "info")
        elif "llama_print_timings" in text:
            # Timing information in magenta
            self.console_output.tag_config("timing", foreground="magenta")
            self.console_output.insert(tk.END, text + "\n", "timing")
        else:
            # Other text in white
            self.console_output.insert(tk.END, text + "\n")
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.console_output.see(tk.END)
        
        self.console_output.config(state=tk.DISABLED)

    def create_chart_area(self, parent):
        """Create chart area with all metrics in a single view"""
        # Create figure with multiple subplots
        self.fig, self.axes = plt.subplots(3, 3, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_file_monitoring(self):
        """Set up file monitoring"""
        try:
            # Create file system event handler
            class FileChangeHandler(FileSystemEventHandler):
                def __init__(self, callback):
                    self.callback = callback
                    
                def on_modified(self, event):
                    if not event.is_directory and event.src_path.endswith(self.callback.csv_file_path):
                        if self.callback.auto_refresh_var.get():
                            self.callback.load_data_and_update()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            
            # Set up observer
            self.event_handler = FileChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, path=os.path.dirname(self.csv_file_path), recursive=False)
            self.observer.start()
        except Exception as e:
            print(f"Error setting up file monitoring: {e}")
            # Continue without file monitoring if it fails

    def setup_command_output_receiver(self):
        """Set up a socket server to receive command output"""
        def socket_server():
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind(('localhost', 12346))
                server_socket.listen(5)
                print("Command output receiver started, listening on port 12346")
                
                while True:
                    try:
                        # Accept connections
                        client_socket, addr = server_socket.accept()
                        print(f"Connection accepted from {addr}")
                        
                        # Start a thread to handle this client connection
                        client_thread = threading.Thread(
                            target=self.handle_client_connection,
                            args=(client_socket,),
                            daemon=True
                        )
                        client_thread.start()
                    except Exception as e:
                        print(f"Error accepting connection: {e}")
                        time.sleep(1)  # Avoid high CPU usage
            except Exception as e:
                print(f"Socket server error: {e}")
        
        # Start socket server thread
        server_thread = threading.Thread(target=socket_server, daemon=True)
        server_thread.start()

    def handle_client_connection(self, client_socket):
        """Handle client connection, receiving command output"""
        try:
            while True:
                # First receive 4-byte length prefix
                length_prefix = client_socket.recv(4)
                if not length_prefix:
                    break  # Connection closed
                    
                # Parse message length
                message_length = struct.unpack('>I', length_prefix)[0]
                
                # Receive complete message
                received = 0
                message_data = b''
                
                while received < message_length:
                    chunk = client_socket.recv(min(4096, message_length - received))
                    if not chunk:
                        break
                    message_data += chunk
                    received += len(chunk)
                
                if received == message_length:
                    # Decode and add to output queue
                    try:
                        decoded_message = message_data.decode('utf-8')
                        self.output_queue.put(decoded_message)
                    except UnicodeDecodeError:
                        print(f"Error decoding message: {message_data}")
        except Exception as e:
            print(f"Error handling client connection: {e}")
        finally:
            client_socket.close()

    def update_console_output(self):
        """Update console with any new output from the command"""
        try:
            # Process all available items in the queue
            while not self.output_queue.empty():
                output = self.output_queue.get_nowait()
                self.append_to_console(output)
        except queue.Empty:
            pass
        
        # Schedule the next update
        self.root.after(100, self.update_console_output)

    def load_data(self):
        """Load CSV data"""
        try:
            if os.path.exists(self.csv_file_path):
                self.data = pd.read_csv(self.csv_file_path)
                
                # Process client_task_num_batch column, which is a list in string form
                if 'client_task_num_batch' in self.data.columns:
                    self.data['client_task_num_batch'] = self.data['client_task_num_batch'].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                
                # Update statistics and sequence label
                self.update_stats()
                if not self.data.empty:
                    latest_sequence = self.data['sequence'].iloc[-1]
                    self.sequence_label.config(text=str(latest_sequence))
                
                return True
            else:
                print(f"File does not exist: {self.csv_file_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def update_stats(self):
        """Update statistics"""
        if self.data is not None and not self.data.empty:
            last_row = self.data.iloc[-1]
            
            # Calculate accuracy avoiding division by zero
            task_num = last_row.get('client_sum_task_num', 0)
            accuracy = 0 if task_num == 0 else last_row.get('client_sum_batch_accuravy_score', 0) / task_num
            
            # Calculate average time avoiding division by zero
            batch_num = last_row.get('client_sum_batch_num', 0)
            avg_time = 0 if batch_num == 0 else last_row.get('client_sum_batch_time_consumption', 0) / batch_num
            
            stats_text = (
                f"Total Batches: {batch_num} | "
                f"Total Tasks: {task_num} | "
                f"Avg Accuracy: {accuracy:.4f} | "
                f"Avg Time: {avg_time:.4f}ms"
            )
            self.stats_label.config(text=stats_text)

    def load_data_and_update(self):
        """Load data and update charts and data tree"""
        if self.load_data():
            self.update_chart()
            self.update_data_tree()

    def update_chart(self):
        """Update all charts in a single view"""
        if self.data is None or self.data.empty:
            return
        
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Set chart style
            sns.set_style("whitegrid")
            
            # Get sequence for x-axis
            x = self.data['sequence']
            
            # 1. Single batch time consumption
            self.axes[0, 0].plot(x, self.data['single_batch_time_consumption'], 'b-o')
            self.axes[0, 0].set_title('Batch Processing Time')
            self.axes[0, 0].set_xlabel('Sequence')
            self.axes[0, 0].set_ylabel('Time (ms)')
            
            # 2. Accuracy score per batch
            self.axes[0, 1].plot(x, self.data['average_batch_accuracy_score_per_batch'], 'g-o')
            self.axes[0, 1].set_title('Batch Accuracy Score')
            self.axes[0, 1].set_xlabel('Sequence')
            self.axes[0, 1].set_ylabel('Accuracy')
            self.axes[0, 1].set_ylim([0, 1.1])
            
            # 3. Throughput score per batch
            self.axes[0, 2].plot(x, self.data['avg_throughput_score_per_batch'], 'r-o')
            self.axes[0, 2].set_title('Batch Throughput Score')
            self.axes[0, 2].set_xlabel('Sequence')
            self.axes[0, 2].set_ylabel('Score')
            
            # 4. Cumulative batch count
            self.axes[1, 0].plot(x, self.data['client_sum_batch_num'], 'c-o')
            self.axes[1, 0].set_title('Cumulative Batch Count')
            self.axes[1, 0].set_xlabel('Sequence')
            self.axes[1, 0].set_ylabel('Count')
            
            # 5. Cumulative task count
            self.axes[1, 1].plot(x, self.data['client_sum_task_num'], 'm-o')
            self.axes[1, 1].set_title('Cumulative Task Count')
            self.axes[1, 1].set_xlabel('Sequence')
            self.axes[1, 1].set_ylabel('Count')
            
            # 6. Cumulative accuracy score
            self.axes[1, 2].plot(x, self.data['client_sum_batch_accuravy_score'], 'y-o')
            self.axes[1, 2].set_title('Cumulative Accuracy Score')
            self.axes[1, 2].set_xlabel('Sequence')
            self.axes[1, 2].set_ylabel('Score')
            
            # 7. Cumulative time consumption
            self.axes[2, 0].plot(x, self.data['client_sum_batch_time_consumption'], 'b-o')
            self.axes[2, 0].set_title('Cumulative Processing Time')
            self.axes[2, 0].set_xlabel('Sequence')
            self.axes[2, 0].set_ylabel('Time (ms)')
            
            # 8. Cumulative throughput score
            self.axes[2, 1].plot(x, self.data['client_sum_batch_throughput_score'], 'g-o')
            self.axes[2, 1].set_title('Cumulative Throughput Score')
            self.axes[2, 1].set_xlabel('Sequence')
            self.axes[2, 1].set_ylabel('Score')
            
            # 9. Task distribution (from the latest record)
            if len(self.data) > 0 and 'client_task_num_batch' in self.data.columns:
                last_row = self.data.iloc[-1]
                task_batches = last_row['client_task_num_batch']
                
                if isinstance(task_batches, list) and len(task_batches) > 0:
                    self.axes[2, 2].hist(task_batches, bins=min(10, len(set(task_batches))), 
                                         color='skyblue', edgecolor='black')
                    self.axes[2, 2].set_title('Task Batch Size Distribution')
                    self.axes[2, 2].set_xlabel('Tasks per Batch')
                    self.axes[2, 2].set_ylabel('Frequency')
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating charts: {e}")

    def run(self):
        """Run visualization client"""
        try:
            self.root.mainloop()
        finally:
            # Ensure observer stops when program exits
            if hasattr(self, 'observer'):
                self.observer.stop()
                self.observer.join()


if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('metrics/client', exist_ok=True)
    
    # Create and run visualizer
    visualizer = MetricsVisualizer()
    visualizer.run()