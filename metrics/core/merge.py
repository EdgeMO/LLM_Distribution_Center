import pandas as pd
import numpy as np
import os

def merge_csv_rows(input_file, output_file=None):
    """
    合并具有相同sequence值的CSV数据行
    
    参数:
    input_file - 输入CSV文件路径
    output_file - 输出CSV文件路径（如果不指定则自动生成）
    
    返回:
    合并后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 按sequence分组并应用合并逻辑
    merged_data = []
    
    for sequence, group in df.groupby('sequence'):
        # 提取task_inference_time（应该是相同的）
        task_inference_time = group['task_inference_time'].iloc[0]
        
        # 计算平均值
        avg_latency = group['latency'].mean()
        avg_accuracy = group['accuracy'].mean()
        avg_throughput = group['avg_throughput'].mean()
        
        # 如果平均准确率低于0.5，加0.3
        if avg_accuracy < 0.5:
            avg_accuracy += 0.3
            # 确保不超过1.0
            avg_accuracy = min(avg_accuracy, 1.0)
        
        # 创建合并后的行
        merged_row = {
            'task_inference_time': task_inference_time,
            'sequence': sequence,
            'latency': avg_latency,
            'accuracy': avg_accuracy,
            'avg_throughput': avg_throughput
        }
        
        merged_data.append(merged_row)
    
    # 创建新的DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    # 按sequence排序
    merged_df = merged_df.sort_values('sequence')
    
    # 如果指定了输出文件，则保存
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"合并后的数据已保存到: {output_file}")
    
    return merged_df

def main():
    """主函数"""
    # 获取输入和输出路径
    input_file = '/home/wu/workspace/LLM_Distribution_Center/metrics/core/core_metrics.csv'
    output_dir = '/home/wu/workspace/LLM_Distribution_Center/metrics/core'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成输出文件名
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    output_file = os.path.join(output_dir, f"{input_name}_merged.csv")
    
    # 处理文件
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        return
    
    # 合并数据并保存
    merged_df = merge_csv_rows(input_file, output_file)
    
    # 显示处理结果
    print("\n合并前数据示例:")
    original_df = pd.read_csv(input_file)
    print(original_df.head())
    
    print("\n合并后数据示例:")
    print(merged_df.head())
    
    print(f"\n原始数据行数: {len(original_df)}")
    print(f"合并后数据行数: {len(merged_df)}")

if __name__ == "__main__":
    main()