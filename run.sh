#!/bin/bash
export TQDM_DISABLE=1

echo "开始运行文本分类预测..."

python llm_predict.py --train --model=llama-3-3b --seed=1 >> train.log 2>&1
echo "llama-3-3b 模型训练完成"
python llm_predict.py --train --model=llama-2-13b --seed=2 >> train.log 2>&1
echo "llama-2-13b 模型训练完成"
python llm_predict.py --train --model=llama-2-7b --seed=3 >> train.log 2>&1
echo "llama-2-7b 模型训练完成"
python llm_predict.py --train --model=llama-3-8b --seed=4 >> train.log 2>&1
echo "llama-3-8b 模型训练完成"
python llm_predict.py --train --model=qwen-7b --seed=5 >> train.log 2>&1
echo "qwen-7b 模型训练完成"
python llm_predict.py --train --model=mistral-7b --seed=6 >> train.log 2>&1
echo "mistral-7b 模型训练完成"

echo "所有模型训练完成"
## 开始评测

python llm_predict.py  --model=llama-2-13b >> train.log 2>&1
echo "llama-2-13b 模型测试完成"
python llm_predict.py  --model=llama-2-7b >> train.log 2>&1
echo "llama-2-7b 模型测试完成"
python llm_predict.py  --model=llama-3-3b >> train.log 2>&1
echo "llama-3-3b 模型测试完成"
python llm_predict.py  --model=llama-3-8b >> train.log 2>&1
echo "llama-3-8b 模型测试完成"
python llm_predict.py  --model=qwen-7b >> train.log 2>&1
echo "qwen-7b 模型测试完成"
python llm_predict.py  --model=mistral-7b >> train.log 2>&1
echo "mistral-7b 模型测试完成"


echo "所有模型测试完成"