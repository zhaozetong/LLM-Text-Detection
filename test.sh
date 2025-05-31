#!/bin/bash

echo "开始运行文本分类预测..."
python llm_predict.py  --model=qwen-7b >> train.log 2>&1
echo "qwen-7b 模型测试完成"
python llm_predict.py  --model=llama-2-13b >> train.log 2>&1
echo "llama-2-13b 模型测试完成"
python llm_predict.py  --model=llama-2-7b>> train.log 2>&1
echo "llama-2-7b 模型测试完成"
python llm_predict.py  --model=mistral-7b>> train.log 2>&1
echo "mistral-7b 模型测试完成"
python llm_predict.py  --model=llama-3-3b>> train.log 2>&1
echo "llama-3-3b 模型测试完成"
python llm_predict.py  --model=llama-3-8b>> train.log 2>&1
echo "llama-3-8b 模型测试完成"

echo "所有模型测试完成"