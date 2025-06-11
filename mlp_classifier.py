import torch
import torch.nn as nn

class MLP_classifier(torch.nn.Module):
    """
    简单的MLP，添加了BatchNorm和Dropout正则化防止梯度爆炸
    
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(MLP_classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim//10)
        self.bn1 = torch.nn.BatchNorm1d(input_dim//10)
        self.fc2 = torch.nn.Linear(input_dim//10, input_dim//100)
        self.bn2 = torch.nn.BatchNorm1d(input_dim//100)
        self.fc3 = torch.nn.Linear(input_dim//100, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)  # 类型转换
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        """返回概率分布用于软投票"""
        logits = self.forward(x)
        return self.softmax(logits)
