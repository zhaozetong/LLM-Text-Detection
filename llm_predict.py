import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import numpy as np
import torch
import random
import os
import sys
from torch.utils.data import DataLoader

from tqdm import tqdm

os.environ["HF_DATASETS_OFFLINE"] = "1" # 使用本地模型


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


def load_model_tok(device):
    safetensors_exist = any(f.endswith(".safetensors") for f in os.listdir(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                use_safetensors=safetensors_exist,
                                                torch_dtype=torch.float16).to(device)# 加载bin 文件
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        )

    return model, tok


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


class LLMFeatureClassifier(torch.nn.Module):
    """
    使用LLM的最后一层隐藏状态作为特征提取器，并训练一个MLP分类器
    LLM权重会被冻结，只有MLP部分参与训练
    """
    def __init__(self, llm_model, tokenizer, device, output_dim=2):
        super(LLMFeatureClassifier, self).__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.device = device
        
        # 冻结LLM参数
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # 获取LLM最后一层隐藏层维度
        # 不同模型可能需要调整
        if hasattr(self.llm.config, 'hidden_size'):
            self.hidden_dim = self.llm.config.hidden_size
        elif hasattr(self.llm.config, 'n_embd'):
            self.hidden_dim = self.llm.config.n_embd
        elif hasattr(self.llm.config, 'dim'):
            self.hidden_dim = self.llm.config.dim
        else:
            raise ValueError(f"无法直接获取隐藏层维度，使用默认值: {self.hidden_dim}")
        
        # MLP分类器
        self.classifier = MLP_classifier(self.hidden_dim, output_dim).to(device)
        
        # 存储隐藏状态的变量
        self.hidden_states = None
        
        # 特征记录相关
        self.recorded_features = []
        self.recorded_labels = []
        self.recording_mode = False
        
        # 注册钩子获取最后一层隐藏状态
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子函数以捕获最后一层隐藏状态"""
        def hook_fn(module, input, output):
            # 保存最后一层transformer块的隐藏状态
            if isinstance(output, tuple):
                self.hidden_states = output[0]  # 一些模型会返回元组
            else:
                self.hidden_states = output
        
        # 针对不同模型结构注册钩子
        # Llama 系列模型
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
            # Llama 3.2, Llama 3, Llama 2等模型
            last_layer = self.llm.model.layers[-1]
            last_layer.register_forward_hook(hook_fn)
        # Mistral 模型
        elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'blocks'):
            last_layer = self.llm.model.blocks[-1]
            last_layer.register_forward_hook(hook_fn)
        # Qwen 模型
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'h'):
            last_layer = self.llm.transformer.h[-1]
            last_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError("不支持的模型结构，无法注册钩子")
    
    def _get_pooled_output(self, hidden_states, attention_mask=None):
        """获取池化后的特征表示（取最后token的隐藏状态）"""
        # 如果提供了attention mask，则使用最后一个非填充token的隐藏状态
        if attention_mask is not None:
            # 找到每个序列最后一个token的位置
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            pooled_output = hidden_states[torch.arange(batch_size), last_token_indices]
        else:
            # 否则直接使用序列最后一个位置
            pooled_output = hidden_states[:, -1]
        
        return pooled_output
    
    def start_recording(self):
        """开始记录特征"""
        self.recording_mode = True
        self.recorded_features = []
        self.recorded_labels = []
    
    def stop_recording(self):
        """停止记录特征"""
        self.recording_mode = False
    
    def get_recorded_features(self):
        """获取记录的特征和标签"""
        if self.recorded_features:
            features = torch.cat(self.recorded_features, dim=0)
            labels = torch.cat(self.recorded_labels, dim=0) if self.recorded_labels else None
            return features, labels
        return None, None
    
    def save_recorded_features(self, save_path, model_name):
        """保存记录的特征"""
        features, labels = self.get_recorded_features()
        if features is not None:
            save_data = {
                'features': features,
                'hidden_dim': self.hidden_dim
            }
            if labels is not None:
                save_data['labels'] = labels
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(save_data, save_path)
            print(f"特征已保存到: {save_path}")
            print(f"特征形状: {features.shape}")
            return True
        return False

    def forward(self, inputs, labels=None):
        """
        前向传递
        inputs: 已经用tokenizer处理过的输入
        labels: 标签（可选，用于记录）
        """
        # 确保模型处于评估模式，不更新参数统计
        self.llm.eval()
        
        # 通过LLM获取最后一层的隐藏状态
        with torch.no_grad():
            _ = self.llm(**inputs)
        
        # 提取最后token的隐藏状态作为特征
        features = self._get_pooled_output(self.hidden_states, inputs.get('attention_mask'))
        
        # 确保特征张量在正确的设备上
        features = features.to(self.device)
        
        # 记录特征（如果处于记录模式）
        if self.recording_mode:
            self.recorded_features.append(features.detach().cpu())
            if labels is not None:
                self.recorded_labels.append(labels.detach().cpu())
        
        # 将提取的特征输入MLP分类器
        logits = self.classifier(features)
        
        return logits, features


def load_model_tok(device):
    safetensors_exist = any(f.endswith(".safetensors") for f in os.listdir(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                use_safetensors=safetensors_exist,
                                                torch_dtype=torch.float16).to(device)# 加载bin 文件
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        )

    return model, tok


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


def train_classifier(model, train_data, val_data, device, batch_size=8, learning_rate=1e-4, num_epochs=1, model_name=""):
    """训练LLMFeatureClassifier模型"""
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    
    best_val_acc = 0.0
    best_model_state = None
    
    # 开始记录训练特征
    model.start_recording()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示训练进度并实时更新loss
        pbar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}/{num_epochs}",disable=not sys.stdout.isatty())
        
        for batch in pbar:
            batch_texts = []
            batch_labels = []
            
            for item in batch:
                batch_texts.append(item["text"])
                batch_labels.append(item["label"])
                
            # 构建输入提示
            prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                      for text in batch_texts]
            
            # 编码输入并确保所有张量都在正确的设备上
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # 将所有张量移到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播（传入labels用于记录）
            logits, _ = model(inputs, labels)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)  # 计算当前平均损失
            
            # 更新进度条，显示实时损失和当前批次准确率
            batch_acc = (torch.argmax(logits, dim=-1) == labels).float().mean().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'batch_acc': f"{batch_acc:.4f}"
            })
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            
            # 释放不需要的张量，节省GPU内存
            del inputs, labels, logits, loss, predictions
            torch.cuda.empty_cache()
        
        epoch_loss = total_loss / len(train_dataloader)
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, 损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}")
        # 验证
        val_acc = evaluate_classifier(model, val_data, device, batch_size) # 这里先不验证,为了提速
        # val_acc = 0.9
        print(f"验证准确率: {val_acc:.4f}")
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.classifier.state_dict().copy()

    # 停止记录并保存训练特征
    model.stop_recording()
    feature_dir = '/root/autodl-tmp/checkpoint/features'
    train_feature_path = os.path.join(feature_dir, f"{model_name}_train_features.pt")
    if len(val_data)==0:
        # 如果没有验证集，直接保存训练特征
        model.save_recorded_features(train_feature_path, model_name)

    # 恢复最佳模型
    if best_model_state is not None:
        model.classifier.load_state_dict(best_model_state)
        print(f"使用最佳验证准确率 {best_val_acc:.4f} 的模型")
    
    return model

def split_train_val_data(data, val_ratio=0.2):
    """
    将数据集随机打乱并划分为训练集和验证集
    
    参数:
        data: 数据列表
        val_ratio: 验证集占比，默认0.2
        seed: 随机种子，保证可重复性
        
    返回:
        train_data, val_data: 划分后的训练集和验证集
    """
    
    # 深拷贝数据，避免修改原始数据
    data_copy = data.copy()
    
    # 随机打乱数据
    random.shuffle(data_copy)
    
    # 计算验证集大小
    val_size = int(len(data_copy) * val_ratio)
    
    # 划分数据
    train_data = data_copy[val_size:]
    val_data = data_copy[:val_size]
    
    return train_data, val_data

def evaluate_classifier(model, data, device, batch_size=8):
    """评估LLMFeatureClassifier模型"""
    if not data:
        print("没有数据可供评估")
        return 0.8
    model.classifier.eval()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=lambda x: x)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="评估中",disable=not sys.stdout.isatty()):
        batch_texts = []
        batch_labels = []
        
        for item in batch:
            batch_texts.append(item["text"])
            batch_labels.append(item["label"])
        
        # 构建输入提示
        prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                  for text in batch_texts]
        
        # 编码输入并移到GPU
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        # 预测
        with torch.no_grad():
            logits, _ = model(inputs)
            predictions = torch.argmax(logits, dim=-1)
        
        # 统计正确预测数量
        correct += (predictions == labels).sum().item()
        total += len(labels)
        
        # 收集所有预测和标签
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 释放GPU内存
        del inputs, labels, logits, predictions
        torch.cuda.empty_cache()
    
    accuracy = correct / total
    
    # 打印混淆矩阵和详细指标
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_predictions))
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=["人类撰写", "AI生成"]))
    
    return accuracy

if __name__ == "__main__":

    model_list = ['llama-2-7b','llama-2-13b','llama-3-8b','qwen-7b','llama-3-3b', 'mistral-7b']
    MODEL_PATHS = {
        'llama-2-7b': "/root/.cache/huggingface/hub/Llama-2-7b-chat-hf",
        'llama-2-13b': "/root/autodl-tmp/Llama-2-13b-chat-hf",
        'llama-3-8b': "/root/autodl-tmp/Llama-3.1-8B-Instruct",
        'qwen-7b': '/root/autodl-tmp/Qwen2.5-7B-Instruct/',
        'llama-3-3b': '/root/autodl-tmp/Llama-3.2-3B-Instruct/',
        'mistral-7b': '/root/autodl-tmp/Mistral-7B-Instruct-v0.3/',
    }
    MODEL = f'llama-3-3b'

    
    import argparse
    parser = argparse.ArgumentParser(description="使用大语言模型预测文本分类")
    parser.add_argument("--model", type=str, default=f"{MODEL}", choices=["llama-2-7b", "llama-2-13b", "llama-3-8b", "qwen-7b", "llama-3-3b", "mistral-7b"], help="选择模型")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, default="test.jsonl", help="测试数据文件路径")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl", help="预测结果输出路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--train",action="store_true", help="训练模型")
    parser.add_argument("--seed",type=int,default=42, help="随机数种子")
    parser.add_argument("--use_pre_features", action="store_true", help="使用预计算的特征进行训练和预测")
    parser.add_argument("--pre_features", action="store_true", help="预计算并保存特征")
    
    args = parser.parse_args()
    set_seed(args.seed)
    MODEL=args.model
    MODEL_PATH = MODEL_PATHS[MODEL]
    # args.use_precomputed_features = False # 默认使用预训练特征
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 特征文件路径
    feature_dir = '/root/autodl-tmp/checkpoint/features'
    train_feature_path = os.path.join(feature_dir, f"{args.model}_train_features.pt")
    test_feature_path = os.path.join(feature_dir, f"{args.model}_test_features.pt")
    model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_best_params.pt'

    # 模式1: 预计算特征模式
    if args.pre_features:
        print("预计算特征模式：提取并保存训练和测试特征...")
        
        # 加载模型和分词器
        print(f"加载模型: {MODEL_PATH}")
        model, tokenizer = load_model_tok(device=device)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 加载数据
        train_data = load_jsonl(args.train_file)
        test_data = load_jsonl(args.test_file)
        print(f"训练数据量: {len(train_data)}, 测试数据量: {len(test_data)}")
        
        # 创建LLM特征分类器
        llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)
        llm_classifier.classifier.to(device)
        
        # 提取训练特征
        print("提取训练特征...")
        llm_classifier.start_recording()
        
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=lambda x: x)
        for batch in tqdm(train_dataloader, desc="提取训练特征"):
            batch_texts = []
            batch_labels = []
            
            for item in batch:
                batch_texts.append(item["text"])
                batch_labels.append(item["label"])
            
            prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                      for text in batch_texts]
            
            inputs = llm_classifier.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            with torch.no_grad():
                _, _ = llm_classifier(inputs, labels)
            
            del inputs, labels
            torch.cuda.empty_cache()
        
        llm_classifier.stop_recording()
        llm_classifier.save_recorded_features(train_feature_path, args.model)
        
        # 提取测试特征
        print("提取测试特征...")
        llm_classifier.start_recording()
        
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda x: x)
        for batch in tqdm(test_dataloader, desc="提取测试特征"):
            batch_texts = []
            
            for item in batch:
                batch_texts.append(item["text"])
            
            prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                      for text in batch_texts]
            
            inputs = llm_classifier.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _, _ = llm_classifier(inputs)
            
            del inputs
            torch.cuda.empty_cache()
        
        llm_classifier.stop_recording()
        llm_classifier.save_recorded_features(test_feature_path, args.model)
        
        print("特征预计算完成！")
        
    # 模式2: 使用预计算特征进行训练和预测
    elif args.use_pre_features:
        print("使用预计算特征模式...")
        
        if args.train:
            print("使用预计算特征进行训练...")
            
            # 检查特征文件是否存在
            if not os.path.exists(train_feature_path):
                raise FileNotFoundError(f"训练特征文件不存在: {train_feature_path}")
            
            # 加载预计算的训练特征
            print(f"加载预计算的训练特征: {train_feature_path}")
            train_feature_data = torch.load(train_feature_path)
            train_features = train_feature_data['features']
            train_labels = train_feature_data['labels']
            hidden_dim = train_feature_data['hidden_dim']
            
            # 创建简单的MLP分类器
            from llm_predict import MLP_classifier
            classifier = MLP_classifier(hidden_dim, 2).to(device)
            
            # 使用预提取特征训练
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(train_features, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            print("使用预计算特征开始训练...")
            classifier.train()
            for epoch in range(1):
                total_loss = 0.0
                correct = 0
                total = 0
                
                pbar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}")
                for features, labels in pbar:
                    features, labels = features.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    logits = classifier(features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += len(labels)
                    
                    # 更新进度条
                    batch_acc = (predictions == labels).float().mean().item()
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'batch_acc': f"{batch_acc:.4f}"
                    })
                
                epoch_acc = correct / total
                print(f"Epoch {epoch+1}, 损失: {total_loss/len(train_dataloader):.4f}, 训练准确率: {epoch_acc:.4f}")
            
            # 保存分类器参数
            torch.save(classifier.state_dict(), model_path)
            print(f"模型参数已保存到 {model_path}")
            
        else:
            print("使用预计算特征进行预测...")
            
            # 检查特征文件是否存在
            if not os.path.exists(test_feature_path):
                raise FileNotFoundError(f"测试特征文件不存在: {test_feature_path}")
            
            # 加载预计算的测试特征
            print(f"加载预计算的测试特征: {test_feature_path}")
            test_feature_data = torch.load(test_feature_path)
            test_features = test_feature_data['features']
            hidden_dim = test_feature_data['hidden_dim']
            
            # 创建分类器并加载训练好的参数
            from llm_predict import MLP_classifier
            classifier = MLP_classifier(hidden_dim, 2).to(device)
            classifier.load_state_dict(torch.load(model_path, map_location=device))
            classifier.eval()
            
            print("使用预计算特征开始预测...")
            predictions = []
            
            test_dataloader = DataLoader(test_features, batch_size=args.batch_size, shuffle=False)
            
            with torch.no_grad():
                for batch_features in tqdm(test_dataloader, desc="预测中"):
                    batch_features = batch_features.to(device)
                    logits = classifier(batch_features)
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    predictions.extend(batch_predictions)
            
            # 保存预测结果
            with open(f'llm_{MODEL}.txt', 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            print(f"预测结果已保存到 llm_{MODEL}.txt，共 {len(predictions)} 条预测")
    
    # 模式3: 原始实时特征提取模式
    else:
        print("实时特征提取模式...")
        
        # 加载模型和分词器
        print(f"加载模型: {MODEL_PATH}")
        model, tokenizer = load_model_tok(device=device)
        tokenizer.pad_token = tokenizer.eos_token
        train_data,val_data = split_train_val_data(load_jsonl(args.train_file), val_ratio=0.0)
        print(f"训练数据量: {len(train_data)}, 验证数据量: {len(val_data)}")
        
        # 创建LLM特征分类器
        llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)
        llm_classifier.classifier.to(device)

        if args.train:
            print("开始训练模式...")
            llm_classifier = train_classifier(llm_classifier, train_data, val_data, device, 
                                            batch_size=args.batch_size, num_epochs=1, model_name=args.model)
            torch.save(llm_classifier.classifier.state_dict(), model_path)
            print(f"模型参数已保存到 {model_path}")

        else:
            print("开始测试模式...")
            llm_classifier.classifier.load_state_dict(torch.load(model_path, map_location=device))
            llm_classifier.eval()
            test_data = load_jsonl(args.test_file)
            print(f"加载测试数据: {len(test_data)} 条")
            
            # 开始记录测试特征
            llm_classifier.start_recording()
            
            predictions = []
            test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda x: x)
            
            for batch in tqdm(test_dataloader, desc="预测中"):
                batch_texts = []
                for item in batch:
                    batch_texts.append(item["text"])
                
                prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                          for text in batch_texts]
                
                inputs = llm_classifier.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits, _ = llm_classifier(inputs)
                    print("分类器训练模式:", llm_classifier.classifier.training)
                    print('feature',_[:5,0:3]) # 测试一下
                    print('logits',logits[:10])
                    print('bias',llm_classifier.classifier.fc1.bias[:5])

                    print("logits",llm_classifier.classifier(_)[:10])
                    logits = llm_classifier.classifier(_) # 手动计算
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                exit()
                predictions.extend(batch_predictions)
            
            # 停止记录并保存测试特征
            llm_classifier.stop_recording()
            llm_classifier.save_recorded_features(test_feature_path, args.model)
            
            with open(f'llm_{MODEL}.txt', 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            print(f"预测结果已保存到 llm_{MODEL}.txt，共 {len(predictions)} 条预测")
