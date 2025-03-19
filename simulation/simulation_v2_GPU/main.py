import os
import time
import pickle
import tensorflow as tf
from tqdm import tqdm
from config.default_config import *
from evaluation.evaluator import TaskAllocationEvaluator
from evaluation.visualization import ResultVisualizer
from utils.helpers import measure_execution_time

# 配置 TensorFlow 以使用 GPU
def configure_tensorflow():
    """配置 TensorFlow 以优化性能"""
    # 检查 GPU 可用性
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"发现 {len(physical_devices)} 个 GPU:")
        for device in physical_devices:
            print(f"  {device.name}")
        
        # 配置 GPU 内存增长
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已启用 GPU 内存动态增长")
            
            # 设置混合精度策略
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("已启用混合精度训练")
            
            # 启用XLA加速
            tf.config.optimizer.set_jit(True)
            print("已启用XLA编译加速")
            
            # 配置环境变量以提高性能
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '1'
            os.environ['TF_USE_CUDNN_AUTOTUNE'] = '1'
            print("已优化GPU线程配置")
            
            return True
        except RuntimeError as e:
            print(f"GPU 配置错误: {e}")
    
    print("未找到 GPU，将使用 CPU 运行")
    return False

def test_gpu():
    """测试 GPU 是否正常工作"""
    print("测试 GPU 功能...")
    
    # 检查 TensorFlow 是否能检测到 GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("未检测到 GPU，请检查驱动和 CUDA 配置")
        return False
    
    # 创建一个简单的模型并在 GPU 上训练
    try:
        # 创建一个简单的模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # 创建一些假数据
        x = tf.random.normal((1000, 5))
        y = tf.random.normal((1000, 1))
        
        # 在 GPU 上训练
        with tf.device('/GPU:0'):
            start_time = time.time()
            model.fit(x, y, epochs=5, verbose=1)
            end_time = time.time()
        
        print(f"GPU 训练完成，耗时 {end_time - start_time:.2f} 秒")
        return True
    except Exception as e:
        print(f"GPU 测试失败: {e}")
        return False

@measure_execution_time
def main():
    print("=" * 80)
    print("启动边缘计算任务分配和模型卸载评估")
    print("=" * 80)
    
    # 测试并配置 GPU
    gpu_working = test_gpu()
    using_gpu = configure_tensorflow()
    
    print(f"配置:")
    print(f"- 边缘节点: {NUM_EDGE_NODES}")
    print(f"- 模型: {NUM_MODELS}")
    print(f"- 特征维度: {FEATURE_DIM}")
    print(f"- 迭代次数: {NUM_ITERATIONS}")
    print(f"- 带宽: {BANDWIDTHS} MB/s")
    print(f"- 批次大小: {BATCH_SIZES}")
    print(f"- 使用GPU: {'是' if using_gpu else '否'}")
    print("-" * 80)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 创建评估器
    evaluator = TaskAllocationEvaluator(
        num_edge_nodes=NUM_EDGE_NODES,
        num_models=NUM_MODELS,
        feature_dim=FEATURE_DIM,
        num_iterations=NUM_ITERATIONS
    )
    
    # 运行评估
    print("开始评估...")
    start_time = time.time()
    results = evaluator.evaluate(bandwidths=BANDWIDTHS, batch_sizes=BATCH_SIZES)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"评估完成，耗时 {total_time:.2f} 秒")
    print("=" * 80)
    
    # 保存原始结果
    with open("results/evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("结果已保存到 results/evaluation_results.pkl")
    
    # 创建可视化器
    visualizer = ResultVisualizer(output_dir="results")
    
    # 生成可视化图表
    print("\n生成可视化图表:")
    print("1. 生成条形图...")
    visualizer.visualize_bar_charts(results, BANDWIDTHS, BATCH_SIZES)
    
    print("2. 生成收敛图...")
    visualizer.visualize_convergence(results, BANDWIDTHS, BATCH_SIZES)
    
    print("3. 生成带宽影响图...")
    for batch_size in tqdm(BATCH_SIZES, desc="带宽影响图"):
        visualizer.visualize_bandwidth_impact(results, BANDWIDTHS, batch_size)
    
    print("4. 生成批次大小影响图...")
    for bandwidth in tqdm(BANDWIDTHS, desc="批次大小影响图"):
        visualizer.visualize_batch_size_impact(results, bandwidth, BATCH_SIZES)
    
    # 生成模型卸载相关指标图表
    print("5. 生成模型卸载指标图表...")
    visualizer.visualize_model_offloading_metrics(results, BANDWIDTHS, BATCH_SIZES)
    
    print("\n" + "=" * 80)
    print("所有可视化完成!")
    print(f"输出文件已保存在 '{visualizer.output_dir}' 目录")
    print("=" * 80)

if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'，避免TensorFlow的多进程问题
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()