from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
# [{"repo_id":"","file_list":[""]}]
local_dir = snapshot_download(repo_id="bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF",cache_dir="/mnt/data/workspace/LLM_Distribution_Center/model/HF")
local_dit_2 = snapshot_download(repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",cache_dir="/mnt/data/workspace/LLM_Distribution_Center/model/HF")
local_dit_3 = snapshot_download(repo_id="Qwen/Qwen2-7B-Instruct-GGUF",cache_dir="/mnt/data/workspace/LLM_Distribution_Center/model/HF")

download_dict =[{"repo_id":"unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF","file_list":["qwen2.5-7b-instruct-q3_k_m.gguf","qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf","qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf","qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf","qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"]},{"repo_id":"bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF","file_list":["DeepSeek-R1-Distill-Qwen-7B-IQ2_M.gguf","DeepSeek-R1-Distill-Qwen-7B-IQ3_M.gguf","DeepSeek-R1-Distill-Qwen-7B-IQ3_XS.gguf","DeepSeek-R1-Distill-Qwen-7B-IQ4_NL.gguf","DeepSeek-R1-Distill-Qwen-7B-Q2_K_L.gguf"]},{"repo_id":"bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF","file_list":["Llama-3.2-3B-Instruct-uncensored-Q2_K.gguf","Llama-3.2-3B-Instruct-uncensored-IQ3_M.gguf","Llama-3.2-3B-Instruct-uncensored-Q2_K_L.gguf","Llama-3.2-3B-Instruct-uncensored-Q3_K_L.gguf","Llama-3.2-3B-Instruct-uncensored-IQ4_XS.gguf",""]},{"repo_id":"mradermacher/DPOB-NMTOB-7B-i1-GGUF","file_list":["DPOB-NMTOB-7B.i1-IQ2_M.gguf","DPOB-NMTOB-7B.i1-IQ2_S.gguf","DPOB-NMTOB-7B.i1-IQ1_M.gguf","DPOB-NMTOB-7B.i1-IQ3_M.gguf","DPOB-NMTOB-7B.i1-IQ3_S.gguf","DPOB-NMTOB-7B.i1-IQ3_XS.gguf"]},{"repo_id":"bartowski/gemma-1.1-7b-it-GGUF","file_list":["gemma-1.1-7b-it-IQ3_M.gguf","gemma-1.1-7b-it-IQ3_S.gguf","gemma-1.1-7b-it-Q2_K.gguf","gemma-1.1-7b-it-Q3_K_S.gguf"]},{"repo_id":"RichardErkhov/distilbert_-_distilgpt2-gguf","file_list":["distilgpt2.IQ3_S.gguf","distilgpt2.Q3_K.gguf","distilgpt2.Q4_K.gguf","distilgpt2.Q5_K.gguf","distilgpt2.Q6_K.gguf"]},{"repo_id":"city96/t5-v1_1-xxl-encoder-gguf","file_list":["t5-v1_1-xxl-encoder-Q3_K_L.gguf","t5-v1_1-xxl-encoder-Q3_K_M.gguf","t5-v1_1-xxl-encoder-Q3_K_S.gguf","t5-v1_1-xxl-encoder-Q4_K_M.gguf","t5-v1_1-xxl-encoder-Q4_K_S.gguf"]}]


download_dict = [{"repo_id":"MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF","file_list":["Qwen2.5-7B-Instruct.IQ1_M.gguf","Qwen2.5-7B-Instruct.IQ1_S.gguf","Qwen2.5-7B-Instruct.IQ2_XS.gguf","Qwen2.5-7B-Instruct.IQ3_XS.gguf","Qwen2.5-7B-Instruct.IQ4_XS.gguf","Qwen2.5-7B-Instruct.Q2_K.gguf","Qwen2.5-7B-Instruct.Q3_K_L.gguf","Qwen2.5-7B-Instruct.Q3_K_M.gguf","Qwen2.5-7B-Instruct.Q3_K_S.gguf"]}]




# for download_batch in download_dict:
#     repo_id = download_batch["repo_id"]
#     file_list = download_batch["file_list"]
#     for file in file_list:
#         try:
#             file_path = hf_hub_download(repo_id,filename=file,cache_dir="/mnt/data/workspace/LLM_Distribution_Center/model/HF")
#             print(file_path)
#         except Exception as e:
#             continue

