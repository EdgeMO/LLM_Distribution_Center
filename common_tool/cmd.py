import subprocess
import psutil
import shutil
class CMD:
    def __init__(self):
        
        pass
    def get_available_memory(self):
        
        """
        获取系统剩余内存（以GB为单位）
        """
        mem = psutil.virtual_memory()
        left_mem = mem.available / (1024 ** 3)
        return left_mem
    def get_free_disk_space(self, path = "/"):
        total, used, free = shutil.disk_usage(path)
        res = free / (2 ** 30)
        return res
        pass
    def run_task_process_cmd(self,query_prefix,query_word,llama_cli_path,model_path):
        """ for edge node to run task process command

        Args:
            query_prefix (_type_): specific prefix for query
            query_word (_type_): actual query word
            llama_cli_path (_type_): path to llama-cli
            model_path (_type_): path to model
        """
        command = [
            llama_cli_path,
            "-m", model_path,
            "-p", f"{query_prefix} {query_word}",
            "-n","128"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(result.stdout)
            temp_res = result.stdout
            return temp_res
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            with open("error.txt", "a+") as file:
                file.write(e.stderr)
            return ''

    def run_perplexity_process_cmd(self,perplexity_path,model_path,test_txt_file_path):
        """ for edge node to run perplexity process command

        Args:
            model_path (_type_): path to model
            test_txt_file_path (_type_): path to test txt file
        """
        command = [
            perplexity_path,
            "-m", model_path,
            "-f", f"{test_txt_file_path}",
        ]
        try:
            result = subprocess.run(command,capture_output=True, text=True, check=True)
            print(result.stdout)
            temp_res = result.stdout
            return temp_res
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            with open("perplexity_error.txt", "a+") as file:
                file.write(e.stderr)
            return ''
# 定义要执行的命令和参数

        
if __name__ == '__main__':
    cmd = CMD()
    #res = cmd.get_free_disk_space()
    query_words = "please answer the following question based on the text provided without explanation \n\n Question:"
    question = f"what's the final sum of 2 + 2"
    llama_cli_path = "/mnt/data/workspace/LLM_Distribution_Edge/build/bin/llama-cli"
    model_path = "/mnt/data/workspace/LLM_Distribution_Center/merge_model.gguf"
    res = cmd.run_task_process_cmd(query_words,question,llama_cli_path,model_path)
    memory_left_get = cmd.get_available_memory()
    print(res)
    pass