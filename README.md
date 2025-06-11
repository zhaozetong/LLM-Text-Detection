## 使用方法
1. 首先提取特征:
```
python extract_features.py --model llama-3-3b --train_file train.jsonl --test_file test.jsonl
```
2. 然后训练MLP集成分类器：
```
python train_mlp_bagging.py --model llama-3-3b --train --num_models 5 --soft_voting
```
3. 或者直接预测（如果已有训练好的模型）：
```
python train_mlp_bagging.py --model llama-3-3b --num_models 5 --soft_voting
```
这样的重构带来以下优势：