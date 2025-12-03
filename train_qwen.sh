#!/bin/bash

# # 中英翻译微调
# echo "开始中英翻译微调..."
# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --model_type qwen \
#     --base_model Qwen/Qwen3-4B-Instruct-2507 \
#     --lang_pair zh-en \
#     --epochs 10 \
#     --output_dir ./checkpoints/qwen3_4b_zh2en_v2 \
#     --batch_size 4 \
#     --lr 1e-4 \
#     --max_length 256 2>&1 | tee train_zh2en_qwen.log

# 中日翻译微调
echo "开始中日翻译微调..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_type qwen \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --lang_pair zh-ja \
    --epochs 10 \
    --output_dir ./checkpoints/qwen3_4b_zh2ja_v2 \
    --batch_size 4 \
    --lr 1e-4 \
    --max_length 256 2>&1 | tee train_zh2ja_qwen.log

echo "全部训练完成！"
