import subprocess

# 定义要执行的命令和参数
command = [
    "/mnt/data/workspace/LLM_Distribution_Edge/llama.cpp-b4174/build/bin/llama-cli",
    "-m", "/mnt/data/workspace/LLM_Distribution_Center/downloaded_models/distilgpt2.IQ3_M.gguf",
    "-p", "I believe the meaning of life is",
    "-n", "128"
]

try:
    # 运行命令并捕获输出
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)
    temp_res = result.stdout
    # 将标准输出保存到文件中
    with open("output.txt", "a+") as file:
        file.write(result.stdout)

    # # 标准错误也可以选择性地保存
    # with open("error.txt", "a+") as file:
    #     file.write(result.stderr)

    print("输出已保存到文件 output.txt")

except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    # 错误信息也保存到文件中
    with open("error.txt", "a+") as file:
        file.write(e.stderr)