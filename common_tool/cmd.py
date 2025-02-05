import subprocess

query_words = "please answer the following question based on the text provided without explanation \n\n Question:"
question = f"Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
# 定义要执行的命令和参数
command = [
    "/mnt/data/workspace/LLM_Distribution_Edge/build/bin/llama-cli",
    "-m", "/mnt/data/workspace/LLM_Distribution_Center/model/models/t5/DPOB-NMTOB-7B.i1-Q4_K_M.gguf",
    "-p", f"{query_words} {question}",
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