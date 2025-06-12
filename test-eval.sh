#!/bin/bash
export TQDM_DISABLE=1

## 开始评测
echo "进行eval模式的开启"
python main.py  --model=llama-3-3b --eval >> train.log 2>&1
echo "llama-3-3b 模型预测完成"
python main.py  --model=llama-2-7b --eval >> train.log 2>&1
echo "llama-2-7b 模型预测完成"
python main.py  --model=llama-2-13b --eval  >> train.log 2>&1
echo "llama-2-13b 模型预测完成"
python main.py  --model=llama-3-8b --eval >> train.log 2>&1
echo "llama-3-8b 模型预测完成"
python main.py  --model=qwen-7b --eval >> train.log 2>&1
echo "qwen-7b  模型预测完成"


echo "所有模型测试完成"