from pathlib import Path
import time
from common_tool.cmd import CMD
test_file = "/mnt/data/workspace/llama.cpp/perplexity_test.csv"
question = f"what's the final sum of 2 + 2"
perplexity_path = "/mnt/data/workspace/llama.cpp/build/bin/llama-perplexity"
cmd = CMD()
def find_model_files(base_path: str, output_file: str = "model_paths.txt"):
    """
    使用 pathlib 遍历目录找到所有的 .gguf 模型文件并写入到文件中
    
    Args:
        base_path: 基础目录路径
        output_file: 输出文件路径
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Base path {base_path} does not exist!")
        return
     
    # 遍历所有模型目录
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        # 检查 snapshots 目录
        snapshots_path = model_dir / 'snapshots'
        if not snapshots_path.exists():
            continue
            
        # 获取第一个 hash 目录
        hash_dirs = [d for d in snapshots_path.iterdir() if d.is_dir()]
        if not hash_dirs:
            continue
            
        # 在 hash 目录中查找所有 .gguf 文件
        for model_file in hash_dirs[0].glob('*.gguf'):
            # 计算文件大小
            try:
                res = cmd.run_perplexity_process_cmd(perplexity_path=perplexity_path,model_path=model_file,test_txt_file_path=test_file)
            except Exception as e:
                print(f"Error processing {model_file}: {e}")
                continue
                
    print(f"Model paths have been written to {output_file}")

if __name__ == "__main__":
    base_path = "/mnt/data/workspace/LLM_Distribution_Center/model/HF"
    find_model_files(base_path)