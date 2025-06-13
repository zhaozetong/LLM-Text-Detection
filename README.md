## 使用方法
1. 首先提取特征:
```
python extract_features.py --model llama-3-3b --train_file train.jsonl --test_file test.jsonl
```
2. 然后训练MLP集成分类器：
```
python main.py --model llama-3-3b --train --num_models 5 --soft_voting
```
3. 或者直接预测（如果已有训练好的模型）：
```
python main.py --model llama-3-3b --num_models 5 --soft_voting
```

## 非常需要注意
对于这个任务而言，使用Dropout会使得结果预测标签出现偏差。具体来说，由于这里的测试数据的分布整体跟训练数据中的1标签数据分别比较接近，不进行正则化的时候预测结果中会有很多（0:1 lablel≈3:10）1标签的结果。如果这里在预测的时候使用了`model.eval()`，模型就会使用所有的特征进行预测，结果就会效果很差（如果没有数据增强和正则化等处理）；而不使用`model.eval()`的时候，预测结果的偏差就没有这么大，01标签分布相对均衡，这一点在特征处理不够的时候表现很明显。这一点是很奇怪的，很大可能是上是跟测试数据的分布偏差相关的。后续的随机森林的预测结果比例接近`1:5`也证实了这一点。当数据增强和模型的正则化较为完整的时候，eval()是否设置就对结果的影响不大了，但是这里的**eval()**对于普通模型的影响之大是意料之外的。
