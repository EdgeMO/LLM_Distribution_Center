import os
import time
import pickle
from config.default_config import *
from evaluation.evaluator import TaskAllocationEvaluator
from evaluation.visualization import ResultVisualizer
from utils.helpers import measure_execution_time
import os
import time
import pickle
from tqdm import tqdm
from config.default_config import *
from evaluation.evaluator import TaskAllocationEvaluator
from evaluation.visualization import ResultVisualizer
from utils.helpers import measure_execution_time
@measure_execution_time
def main():
    print("=" * 80)
    print("Starting Edge Computing Task Allocation and Model Offloading Evaluation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"- Edge Nodes: {NUM_EDGE_NODES}")
    print(f"- Models: {NUM_MODELS}")
    print(f"- Feature Dimensions: {FEATURE_DIM}")
    print(f"- Iterations: {NUM_ITERATIONS}")
    print(f"- Bandwidths: {BANDWIDTHS} MB/s")
    print(f"- Batch Sizes: {BATCH_SIZES}")
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
    print("Starting evaluation...")
    start_time = time.time()
    results = evaluator.evaluate(bandwidths=BANDWIDTHS, batch_sizes=BATCH_SIZES)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Evaluation completed in {total_time:.2f} seconds")
    print("=" * 80)
    
    # 保存原始结果
    with open("results/evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("Results saved to results/evaluation_results.pkl")
    
    # 创建可视化器
    visualizer = ResultVisualizer(output_dir="results")
    
    # 生成条形图 - 每个带宽和批次大小组合单独出图
    print("\nGenerating visualizations:")
    print("1. Generating bar charts...")
    visualizer.visualize_bar_charts(results, BANDWIDTHS, BATCH_SIZES)
    
    # 生成收敛图 - 每个带宽和批次大小组合单独出图
    print("2. Generating convergence plots...")
    visualizer.visualize_convergence(results, BANDWIDTHS, BATCH_SIZES)
    
    # 生成带宽影响图 - 固定批次大小，研究带宽影响
    print("3. Generating bandwidth impact plots...")
    for batch_size in tqdm(BATCH_SIZES, desc="Bandwidth impact plots"):
        visualizer.visualize_bandwidth_impact(results, BANDWIDTHS, batch_size)
    
    # 生成批次大小影响图 - 固定带宽，研究批次大小影响
    print("4. Generating batch size impact plots...")
    for bandwidth in tqdm(BANDWIDTHS, desc="Batch size impact plots"):
        visualizer.visualize_batch_size_impact(results, bandwidth, BATCH_SIZES)
    
    # 生成模型卸载相关指标图表
    print("5. Generating model offloading metrics...")
    visualizer.visualize_model_offloading_metrics(results, BANDWIDTHS, BATCH_SIZES)
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print(f"Output files are saved in the '{visualizer.output_dir}' directory")
    print("=" * 80)

if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'，避免TensorFlow的多进程问题
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()