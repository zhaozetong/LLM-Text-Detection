#!/bin/bash
export TQDM_DISABLE=1

## 开始评测
echo "不进行eval模式的开启"
python main.py  --model=llama-3-3b  >> train.log 2>&1
echo "llama-3-3b 模型预测完成"
python main.py  --model=llama-2-7b  >> train.log 2>&1
echo "llama-2-7b 模型预测完成"
python main.py  --model=llama-2-13b   >> train.log 2>&1
echo "llama-2-13b 模型预测完成"
python main.py  --model=llama-3-8b  >> train.log 2>&1
echo "llama-3-8b 模型预测完成"
python main.py  --model=qwen-7b  >> train.log 2>&1
echo "qwen-7b  模型预测完成"


echo "所有模型测试完成"