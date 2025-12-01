#!/bin/bash

# 创建目录
mkdir -p models/nllb-200-distilled-600M
cd models/nllb-200-distilled-600M

echo "开始下载 NLLB 模型文件..."

# 定义基础 URL (使用 hf-mirror 镜像)
BASE_URL="https://hf-mirror.com/facebook/nllb-200-distilled-600M/resolve/main"

# 下载文件列表
files=(
    "config.json"
    "pytorch_model.bin"
    "tokenizer.json"
    "sentencepiece.bpe.model"
    "special_tokens_map.json"
    "tokenizer_config.json"
    "generation_config.json"
)

for file in "${files[@]}"; do
    echo "正在下载: $file"
    # 使用 -c 支持断点续传
    wget -c "$BASE_URL/$file"
    if [ $? -ne 0 ]; then
        echo "❌ 下载 $file 失败！"
        exit 1
    fi
done

echo "✅ 所有文件下载完成！"
ls -lh
cd ../..

