import os
from modelscope.hub.snapshot_download import snapshot_download

# 新的高容量目标路径
TARGET_DIR = "/data/zhangst/project/moka/pre-train"

# 确保目录存在
os.makedirs(TARGET_DIR, exist_ok=True)

print("🚀 1. 开始从魔搭社区下载 LLaMA-2-7b-chat-hf...")
# ModelScope 上 LLaMA-2-7b-chat 的官方同步仓库 ID
snapshot_download(
    model_id="modelscope/Llama-2-7b-chat-ms",
    local_dir=f"{TARGET_DIR}/Llama-2-7b-chat-hf",
    revision="master"
)
print("✅ LLaMA-2-7b-chat-hf 下载完成！\n")


print("🚀 2. 开始从魔搭社区下载 openai-clip-vit-large-patch14...")
# ModelScope 上 CLIP 的同步仓库 ID
snapshot_download(
    model_id="AI-ModelScope/clip-vit-large-patch14",
    local_dir=f"{TARGET_DIR}/clip-vit-large-patch14",
    revision="master"
)
print("✅ CLIP 下载完成！\n")


print("🚀 3. 开始从魔搭社区下载 bert-base-uncased...")
# ModelScope 上 BERT 的同步仓库 ID
snapshot_download(
    model_id="AI-ModelScope/bert-base-uncased",
    local_dir=f"{TARGET_DIR}/bert-base-uncased",
    revision="master"
)
print("✅ BERT 下载完成！\n")

print("🎉 所有模型均已满速下载完毕！")