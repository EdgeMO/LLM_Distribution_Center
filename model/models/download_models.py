from huggingface_hub import snapshot_download
repo_id = "mradermacher/NMTOB-7B-i1-GGUF"  
file_path = 'NMTOB-7B.i1-IQ3_M.gguf'
token = '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'
local_dir_use_symlinks  = False
#model_dir = snapshot_download('mradermacher/NMTOB-7B-i1-GGUF', cache_dir='NMT')
#print(model_dir)
from huggingface_hub import hf_hub_download
# 模型所在仓库的ID
# 要下载的文件路径（相对于仓库根目录）
# 本地保存路径
local_path = "/mnt/data/workspace/models/NMT"
# 下载文件
hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=local_path)