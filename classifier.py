import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MLP_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):  # 增加dropout
        super(MLP_classifier, self).__init__()
        # 添加更多层和正则化
        self.fc1 = torch.nn.Linear(input_dim, input_dim//8)  # 减小第一层
        self.bn1 = torch.nn.BatchNorm1d(input_dim//8)
        self.fc2 = torch.nn.Linear(input_dim//8, input_dim//64)  # 更小的中间层
        self.bn2 = torch.nn.BatchNorm1d(input_dim//64)
        self.fc3 = torch.nn.Linear(input_dim//64, input_dim//128)  # 新增层
        self.bn3 = torch.nn.BatchNorm1d(input_dim//128)
        self.fc4 = torch.nn.Linear(input_dim//128, output_dim)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return self.softmax(logits)
    

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, 512)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, 512))
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=num_heads, 
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x).unsqueeze(1)  # [batch, 1, 512]
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        return self.classifier(x)

class ResNetClassifier(nn.Module):
    """ResNet风格的分类器"""
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # ResNet块
        self.block1 = self._make_block(input_dim, input_dim//2, dropout)
        self.block2 = self._make_block(input_dim//2, input_dim//4, dropout)
        self.block3 = self._make_block(input_dim//4, input_dim//8, dropout)
        
        self.classifier = nn.Linear(input_dim//8, num_classes)
        
    def _make_block(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        
        # Block 1 with residual
        identity = F.adaptive_avg_pool1d(x.unsqueeze(1), self.block1[0].out_features).squeeze(1)
        x = self.block1(x) + identity
        x = F.relu(x)
        
        # Block 2 with residual
        identity = F.adaptive_avg_pool1d(x.unsqueeze(1), self.block2[0].out_features).squeeze(1)
        x = self.block2(x) + identity
        x = F.relu(x)
        
        # Block 3 with residual
        identity = F.adaptive_avg_pool1d(x.unsqueeze(1), self.block3[0].out_features).squeeze(1)
        x = self.block3(x) + identity
        x = F.relu(x)
        
        return self.classifier(x)

class RandomForestWrapper:
    """随机森林分类器的PyTorch兼容封装"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.device = 'cpu'  # 随机森林只支持CPU
        
    def to(self, device):
        # 随机森林不需要设备转换，保持兼容性
        return self
        
    def train(self):
        # 兼容PyTorch接口
        pass
        
    def eval(self):
        # 兼容PyTorch接口
        pass
        
    def fit(self, X, y):
        """训练随机森林模型"""
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        return self.model.fit(X, y)
        
    def predict(self, X):
        """预测类别"""
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        predictions = self.model.predict(X)
        return torch.tensor(predictions)
        
    def predict_proba(self, X):
        """预测概率"""
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        probabilities = self.model.predict_proba(X)
        return torch.tensor(probabilities)
        
    def __call__(self, X):
        """前向传播兼容接口"""
        probabilities = self.predict_proba(X)
        # 返回logits格式 (log概率)
        return torch.log(probabilities + 1e-8)