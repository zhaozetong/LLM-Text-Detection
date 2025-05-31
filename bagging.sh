#!/bin/bash
export TQDM_DISABLE=1

echo "开始运行文本分类预测..."

# python llm_bagging.py --train --model=llama-3-3b >> train.log 2>&1
# echo "llama-3-3b 模型训练完成"
python llm_bagging.py --train --model=llama-2-13b>> train.log 2>&1
echo "llama-2-13b 模型训练完成"
python llm_bagging.py --train --model=llama-2-7b >> train.log 2>&1
echo "llama-2-7b 模型训练完成"
python llm_bagging.py --train --model=llama-3-8b  >> train.log 2>&1
echo "llama-3-8b 模型训练完成"
python llm_bagging.py --train --model=qwen-7b >> train.log 2>&1
echo "qwen-7b 模型训练完成"
python llm_bagging.py --train --model=mistral-7b >> train.log 2>&1
echo "mistral-7b 模型训练完成"

echo "所有模型训练完成"
