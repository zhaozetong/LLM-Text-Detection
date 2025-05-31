import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from tqdm import tqdm

# 设定随机种子，确保可复现性
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 创建数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 从张量中移除批次维度
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        if 'label' in item:
            label = torch.tensor(item['label'], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'id': idx
            }

# 加载数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 创建模型
def create_model(num_labels=2, pretrained_path='bert-base-uncased'):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_path,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    return model

# 保存模型参数（不保存整个模型）
def save_model_params(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 提取模型超参数和权重
    model_params = {
        'model_config': model.config.to_dict(),
        'model_state': model.state_dict()
    }
    
    torch.save(model_params, save_path)
    print(f"模型参数已保存到 {save_path}")

# 加载模型参数
def load_model_params(model, load_path):
    model_params = torch.load(load_path)
    model.load_state_dict(model_params['model_state'])
    return model
