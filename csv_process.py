import pandas as pd

def process_csv_files(original_file, append_file, output_file):
    # 读取原始 CSV 文件
    df_original = pd.read_csv(original_file)
    
    # 过滤掉 task_type 为 2 的行
    df_filtered = df_original[df_original['task_type'] != 2]
    
    # 读取需要插入的 CSV 文件
    df_append = pd.read_csv(append_file)
    
    # 合并两个 DataFrame，将插入文件的数据添加到过滤后的数据之后
    df_result = pd.concat([df_filtered, df_append], ignore_index=True)
    
    # 将合并后的数据写入新的 CSV 文件
    df_result.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    original_file = '/mnt/data/workspace/LLM_Distribution_Center/data/example.csv'
    append_file = '/mnt/data/workspace/LLM_Distribution_Center/test.csv'
    output_file = '/mnt/data/workspace/LLM_Distribution_Center/final_example.csv'
    
    process_csv_files(original_file, append_file, output_file)