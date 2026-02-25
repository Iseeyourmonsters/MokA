#!/bin/bash

# 设置目标数据集目录
DATA_DIR="/data/zhangst/data/LLaVA-Instruct-150K"
mkdir -p "$DATA_DIR"

echo "🚀 开始下载 Visual-Text Stage 2 (LLaVA-Instruct-150K) 数据集到 $DATA_DIR..."

# ---------------------------------------------------------
# 1. 下载原始图像数据 (COCO train2017)
# 这步最耗时 (约18GB)，使用 wget 的 -c 参数支持断点续传
# ---------------------------------------------------------
echo "下载 COCO 2017 训练集图片 (18GB)..."
wget -c http://images.cocodataset.org/zips/train2017.zip -O "$DATA_DIR/train2017.zip"

echo "图片下载完成，正在解压 train2017.zip... (这可能需要几分钟)"
# 检查是否安装了 unzip
if ! command -v unzip &> /dev/null
then
    echo "未找到 unzip 工具，请新开一个终端运行: sudo apt install unzip，然后手动解压。"
else
    # -q 安静模式解压，避免终端被几万行日志淹没
    unzip -q "$DATA_DIR/train2017.zip" -d "$DATA_DIR/"
    echo "✅ 图片解压完成！(图片已存在于 $DATA_DIR/train2017 文件夹中)"

    # 选做：如果你硬盘空间吃紧，取消下面这行的注释来删除压缩包
     rm "$DATA_DIR/train2017.zip"
fi

echo "🎉 Visual-Text 数据集脚本执行完毕！"