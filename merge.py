import numpy as np
from collections import Counter

model_list = ['llama-2-7b','llama-2-13b','llama-3-8b','qwen-7b','llama-3-3b']
res = []

# 读取所有模型预测结果
predictions = []
for model in model_list:

    file_name = f'{model}_mlp.txt'

    with open(file_name, 'r') as f:
        model_preds = [int(line.strip()) for line in f.readlines()]
        if len(model_preds) != 2800:
            print(f"警告: {file_name} 包含 {len(model_preds)} 条预测，预期应为2800条")
        predictions.append(model_preds)
        print(f"已加载 {file_name} 的预测结果")


# 转换为numpy数组以方便处理
predictions = np.array(predictions)
print(f"成功加载 {len(predictions)} 个模型的预测结果")

# 进行bagging (多数投票)
bagging_results = []
# 记录比例为15:16或16:15的样本索引
close_vote_indices = []

for i in range(predictions.shape[1]):
    # 获取所有模型对第i个样本的预测
    votes = predictions[:, i]
    # 计算0和1的投票数
    vote_counts = Counter(votes)
    # 选择票数最多的类别作为最终预测
    majority_vote = vote_counts.most_common(1)[0][0]
    bagging_results.append(majority_vote)
    
    # 检查是否为15:16或16:15的比例
    zeros_count = vote_counts.get(0, 0)
    ones_count = vote_counts.get(1, 0)
    total_votes = zeros_count + ones_count
    
    # if (zeros_count == 15 and ones_count == 16) or (zeros_count == 16 and ones_count == 15):
    #     close_vote_indices.append(i)
    #     print(f"索引 {i} 的投票比例为 {zeros_count}:{ones_count}")
    p = zeros_count / (ones_count + 0.1)  # 防止除以0
    if 0.3<p<0.7: # 找出不太确定的标签
        close_vote_indices.append(i)
        
# 将bagging结果保存到文件
with open('final.txt', 'w') as f:
    for result in bagging_results:
        f.write(f"{result}\n")

# 将15:16或16:15比例的索引保存到15index.txt
with open('uncertain_index.txt', 'w') as f:
    for idx in close_vote_indices:
        f.write(f"{idx}\n")

print(f"Bagging完成，结果已保存到 final.txt")
print(f"共 {len(bagging_results)} 条预测，其中标签0: {bagging_results.count(0)}个，标签1: {bagging_results.count(1)}个")
print(f"发现 {len(close_vote_indices)} 个投票比例不中的样本，索引已保存到 uncertain_index.txt")
