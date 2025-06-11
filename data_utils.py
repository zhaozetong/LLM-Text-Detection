import json
import random
import numpy as np
import torch

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_non_overlapping_subsets(data_indices, num_subsets, subset_ratio=0.3, labels=None):
    """
    创建重叠度较小的数据子集，支持标签平衡
    
    Args:
        data_indices: 数据索引列表
        num_subsets: 子集数量
        subset_ratio: 每个子集占总数据的比例
        labels: 标签张量，用于平衡正负样本
    
    Returns:
        List[List[int]]: 每个子集的索引列表
    """
    total_size = len(data_indices)
    subset_size = int(total_size * subset_ratio)
    
    subsets = []
    
    if num_subsets == 1:
        # 如果只有一个模型，返回完整数据集
        return [data_indices]
    
    # 如果提供了标签，进行平衡采样
    if labels is not None:
        # 按标签分组索引
        label_0_indices = [idx for idx in data_indices if labels[idx] == 0]
        label_1_indices = [idx for idx in data_indices if labels[idx] == 1]
        
        # 计算每个子集中每个标签的样本数（保持1:1比例）
        samples_per_label = subset_size // 2
        
        for i in range(num_subsets):
            subset_indices = []
            
            # 从标签0中采样
            if len(label_0_indices) >= samples_per_label:
                sampled_0 = random.sample(label_0_indices, samples_per_label)
            else:
                # 如果标签0样本不足，进行有放回采样
                sampled_0 = random.choices(label_0_indices, k=samples_per_label)
            
            # 从标签1中采样
            if len(label_1_indices) >= samples_per_label:
                sampled_1 = random.sample(label_1_indices, samples_per_label)
            else:
                # 如果标签1样本不足，进行有放回采样
                sampled_1 = random.choices(label_1_indices, k=samples_per_label)
            
            subset_indices = sampled_0 + sampled_1
            random.shuffle(subset_indices)  # 打乱顺序
            
            subsets.append(subset_indices)
    
    else:
        # 原有的随机采样逻辑
        for i in range(num_subsets):
            subset_indices = random.sample(data_indices, subset_size)
            subsets.append(subset_indices)
    
    return subsets

def load_precomputed_features(feature_path):
    """加载预计算的特征"""
    feature_data = torch.load(feature_path, map_location='cpu')
    features = feature_data['features']
    labels = feature_data.get('labels', None)  # 安全获取labels
    hidden_dim = feature_data['hidden_dim']
    return features, labels, hidden_dim
