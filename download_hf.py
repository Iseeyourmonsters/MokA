import os
from huggingface_hub import snapshot_download

# 强制使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制把所有隐藏缓存也放到空间大的数据盘里
os.environ["HF_HOME"] = "/data/zhangst/project/moka/hf_cache"

# 你的 Hugging Face Token (务必替换！)
# 加载 .env 文件中的环境变量
load_dotenv()

# 安全地获取 Token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("未找到 HF_TOKEN，请检查 .env 文件是否配置正确！")

# 新的高容量目标路径
TARGET_DIR = "/data/zhangst/project/moka/pre-train"

print("🚀 开始检查并下载 LLaMA-2-7b-chat-hf...")
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir=f"{TARGET_DIR}/Llama-2-7b-chat-hf",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
    resume_download=True # 开启断点续传
)
print("✅ LLaMA-2-7b-chat-hf 检查/下载完成！\n")

print("🚀 开始检查并续传 openai-clip-vit-large-patch14...")
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir=f"{TARGET_DIR}/clip-vit-large-patch14",
    local_dir_use_symlinks=False,
    resume_download=True # 开启断点续传
)
print("✅ CLIP 下载完成！")

print("🚀 开始下载 bert-base-uncased...")
snapshot_download(
    repo_id="bert-base-uncased",
    local_dir="/data/zhangst/project/moka/pre-train/bert-base-uncased",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("✅ BERT 下载完成！")