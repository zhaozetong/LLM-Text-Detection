import numpy as np
from collections import Counter

model_list = ['llama-2-7b','llama-2-13b','llama-3-8b','qwen-7b','llama-3-3b', \
              'mistral-7b','bert',]
res = []

# 读取所有模型预测结果
predictions = []
for model in model_list:
    file_name = f"llm_{model}.txt"

    with open(file_name, 'r') as f:
        model_preds = [int(line.strip()) for line in f.readlines()]
        if len(model_preds) != 2800:
            print(f"警告: {file_name} 包含 {len(model_preds)} 条预测，预期应为2800条")
        predictions.append(model_preds)
        print(f"已加载 {file_name} 的预测结果")


# 确保至少有一个模型的预测结果被成功读取
if not predictions:
    print("错误: 没有成功读取任何模型的预测结果")
    exit(1)

# 转换为numpy数组以方便处理
predictions = np.array(predictions)
print(f"成功加载 {len(predictions)} 个模型的预测结果")

# 进行bagging (多数投票)
bagging_results = []
# 记录投票比例为3:4或4:3的样本索引
close_vote_indices = []

for i in range(predictions.shape[1]):
    # 获取所有模型对第i个样本的预测
    votes = predictions[:, i]
    # 计算0和1的投票数
    vote_counts = Counter(votes)
    # 选择票数最多的类别作为最终预测
    majority_vote = vote_counts.most_common(1)[0][0]
    bagging_results.append(majority_vote)
    
    # 检查是否为3:4或4:3的情况
    if (vote_counts.get(0, 0) == 3 and vote_counts.get(1, 0) == 4) or \
       (vote_counts.get(0, 0) == 4 and vote_counts.get(1, 0) == 3):
        close_vote_indices.append(i)

# 将bagging结果保存到文件
with open('bagging.txt', 'w') as f:
    for result in bagging_results:
        f.write(f"{result}\n")

# 保存接近平衡(3:4或4:3)的样本索引
with open('close_votes_indices.txt', 'w') as f:
    for idx in close_vote_indices:
        f.write(f"{idx}\n")

print(f"Bagging完成，结果已保存到 bagging.txt")
print(f"共 {len(bagging_results)} 条预测，其中标签0: {bagging_results.count(0)}个，标签1: {bagging_results.count(1)}个")
print(f"发现 {len(close_vote_indices)} 个投票比例为3:4或4:3的样本，索引已保存到 close_votes_indices.txt")
