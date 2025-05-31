import numpy as np
from collections import Counter

model_list = ['llama-2-7b','llama-2-13b','llama-3-8b','qwen-7b','llama-3-3b', \
              'mistral-7b','no_seed']
res = []

# 读取所有模型预测结果
all_model_predictions = {}  # 存储每个模型的所有子模型预测
final_model_predictions = []  # 存储每个模型的最终预测

# 首先加载所有数据
for model in model_list:
    if model == 'no_seed':
        # 对于no_seed，直接加载单个文件
        file_name = "./5res/no_seed.txt"
        with open(file_name, 'r') as f:
            model_preds = [int(line.strip()) for line in f.readlines()]
            if len(model_preds) != 2800:
                print(f"警告: {file_name} 包含 {len(model_preds)} 条预测，预期应为2800条")
            final_model_predictions.append(model_preds)
            print(f"已加载 {file_name} 的预测结果")
    else:
        # 对于其他模型，加载5个子模型的预测
        model_subset_preds = []
        for i in range(1, 6):
            file_name = f"./5res/llm_{model}_subset_{i}.txt"
            try:
                with open(file_name, 'r') as f:
                    subset_preds = [int(line.strip()) for line in f.readlines()]
                    if len(subset_preds) != 2800:
                        print(f"警告: {file_name} 包含 {len(subset_preds)} 条预测，预期应为2800条")
                    model_subset_preds.append(subset_preds)
                    print(f"已加载 {file_name} 的预测结果")
            except Exception as e:
                print(f"读取文件 {file_name} 错误: {str(e)}")
                
        # 如果成功加载了子模型预测，进行内部bagging
        if model_subset_preds:
            all_model_predictions[model] = np.array(model_subset_preds)
            
            # 为每个模型进行内部bagging
            model_bagging_results = []
            for i in range(len(model_subset_preds[0])):  # 对每个样本
                # 获取所有子模型对第i个样本的预测
                subset_votes = [preds[i] for preds in model_subset_preds]
                # 计算0和1的票数
                vote_counts = Counter(subset_votes)
                # 选择票数最多的类别作为最终预测
                majority_vote = vote_counts.most_common(1)[0][0]
                model_bagging_results.append(majority_vote)
            
            # 保存每个模型的内部bagging结果
            final_model_predictions.append(model_bagging_results)
            
            # 可选：保存每个模型的bagging结果
            with open(f'./5res/{model}_bagged.txt', 'w') as f:
                for result in model_bagging_results:
                    f.write(f"{result}\n")
            print(f"模型 {model} 的5个子模型已bagging，结果已保存到 ./5res/{model}_bagged.txt")

# 将所有模型的最终预测转换为numpy数组
final_predictions = np.array(final_model_predictions)
print(f"所有模型bagging后，共有 {len(final_predictions)} 个模型的预测结果")

# 进行最终的bagging (多数投票)
bagging_results = []
# 记录比例为15:16或16:15的样本索引
close_vote_indices = []

for i in range(final_predictions.shape[1]):
    # 获取所有模型对第i个样本的预测
    votes = final_predictions[:, i]
    # 计算0和1的投票数
    vote_counts = Counter(votes)
    # 选择票数最多的类别作为最终预测
    majority_vote = vote_counts.most_common(1)[0][0]
    bagging_results.append(majority_vote)
    
    # 检查是否为投票比例接近的情况
    zeros_count = vote_counts.get(0, 0)
    ones_count = vote_counts.get(1, 0)
    total_votes = zeros_count + ones_count
    
    if (zeros_count == 15 and ones_count == 16) or (zeros_count == 16 and ones_count == 15):
        close_vote_indices.append(i)
        print(f"索引 {i} 的投票比例为 {zeros_count}:{ones_count}")

# 将bagging结果保存到文件
with open('merge.txt', 'w') as f:
    for result in bagging_results:
        f.write(f"{result}\n")

# 将15:16或16:15比例的索引保存到15index.txt
with open('15index.txt', 'w') as f:
    for idx in close_vote_indices:
        f.write(f"{idx}\n")

print(f"Bagging完成，结果已保存到 merge.txt")
print(f"共 {len(bagging_results)} 条预测，其中标签0: {bagging_results.count(0)}个，标签1: {bagging_results.count(1)}个")
print(f"发现 {len(close_vote_indices)} 个投票比例为15:16或16:15的样本，索引已保存到 15index.txt")
