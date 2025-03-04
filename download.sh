#!/bin/bash
# 这是使用huggingface-cli下载模型的脚本
# 设置环境变量HF_ENDPOINT
export HF_ENDPOINT="https://hf-mirror.com"

echo "HF_ENDPOINT is set to: $HF_ENDPOINT"

MODEL_NAME=$1
LOCAL_DIR=$2

echo "MODEL_NAME is: $MODEL_NAME"
echo "LOCAL_DIR is: $LOCAL_DIR"

# 使用huggingface-cli下载模型
huggingface-cli download --resume-download $MODEL_NAME --local-dir $LOCAL_DIR

# sh download.sh Qwen/Qwen1.5-1.8B-Chat Qwen/Qwen1.5-1.8B-Chat