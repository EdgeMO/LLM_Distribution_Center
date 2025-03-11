import os
import sys
from pathlib import Path
from common_tool.cmd import CMD
test_file = "/home/wu/workspace/LLM_Distribution_Center/data/perpelexity.csv"
perplexity_path = "/home/wu/workspace/LLM_Distribution_Edge/build/bin/llama-perplexity"
cmd = CMD()
def list_model_files(directory_path, extensions=None):
    """
    获取指定目录下的所有模型文件路径，并按默认顺序排序
    
    参数:
    directory_path -- 目录路径
    extensions -- 文件扩展名列表，例如 ['.gguf', '.bin']，如果为 None，则获取所有文件
    
    返回:
    排序后的文件路径列表
    """
    try:
        # 确保目录存在
        if not os.path.exists(directory_path):
            print(f"错误: 目录 '{directory_path}' 不存在")
            return []
        
        # 获取所有文件
        all_files = []
        
        # 遍历目录
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            
            # 检查是否为文件
            if os.path.isfile(item_path):
                # 如果指定了扩展名，则只包含匹配的文件
                if extensions is None or any(item.lower().endswith(ext.lower()) for ext in extensions):
                    all_files.append(item_path)
        
        # 按默认顺序（字母数字）排序
        all_files.sort()
        
        return all_files
    
    except Exception as e:
        print(f"获取文件列表时出错: {e}")
        return []

def main():
    # 模型目录
    models_dir = "/mnt/d/models"
    
    # 常见的模型文件扩展名
    model_extensions = ['.gguf', '.bin', '.ggml', '.pt', '.pth', '.safetensors']
    
    # 获取所有模型文件路径
    print(f"正在获取 {models_dir} 中的模型文件...")
    
    # 方式1: 获取所有文件
    all_files = list_model_files(models_dir)
    
    # 方式2: 只获取特定扩展名的文件
    model_files = list_model_files(models_dir, model_extensions)
    # 
    # 打印结果
    target_file = "/mnt/d/models/t5-v1_1-xxl-encoder-Q4_K_S.gguf"
    found_target = False

    for i, file_path in enumerate(model_files, 1):
        # 检查是否找到目标文件或目标文件就是当前文件
        if found_target or file_path == target_file:
            if not found_target and file_path == target_file:
                print(f"找到目标文件: {file_path}")
                found_target = True
            
            try:
                print(f"处理文件 {i}/{len(model_files)}: {file_path}")
                res = cmd.run_perplexity_process_cmd(perplexity_path=perplexity_path, model_path=file_path, test_txt_file_path=test_file)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        else:
            print(f"跳过文件 {i}/{len(model_files)}: {file_path}")

if __name__ == "__main__":
    model_files = main()




