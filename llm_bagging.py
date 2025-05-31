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


class MLP_classifier2(torch.nn.Module):
    """
    2层的mlp,拟合效果更好
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(MLP_classifier2, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 200)
        self.bn1 = torch.nn.BatchNorm1d(200)
        self.fc3 = torch.nn.Linear(200, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.to(torch.float32)  # 类型转换
        x = self.fc1(x)
        x = self.bn1(x)
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
            # 默认Llama模型
            self.hidden_dim = 4096
            print(f"无法直接获取隐藏层维度，使用默认值: {self.hidden_dim}")
        
        # MLP分类器
        # self.classifier = MLP_classifier(self.hidden_dim, output_dim).to(device)
        #
        #
        #
        #
        #
        #
        
        self.classifier = MLP_classifier2(self.hidden_dim, output_dim).to(device)
        # 存储隐藏状态的变量
        self.hidden_states = None
        
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
    
    def forward(self, inputs):
        """
        前向传递
        inputs: 已经用tokenizer处理过的输入
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
        
        # 将提取的特征输入MLP分类器
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, text):
        """预测单个文本的分类"""
        # 准备输入
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # 获取预测结果
        with torch.no_grad():
            logits = self.forward(inputs)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.item()
    
    def get_last_hidden_state(self):
        """返回最后捕获的隐藏状态"""
        return self.hidden_states


def train_classifier(model, train_data, val_data, device, batch_size=8, learning_rate=1e-4, num_epochs=1):
    """训练LLMFeatureClassifier模型"""
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    
    best_model_state = None
    
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
            prompts = [PROMPT.format(text)
                      for text in batch_texts]
            
            # 编码输入并确保所有张量都在正确的设备上
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # 将所有张量移到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播
            logits = model(inputs)
            
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
        
        # 保存最后一轮模型状态（不再需要验证）
        best_model_state = model.classifier.state_dict().copy()

    # 恢复最佳模型（此时是最后一轮的状态）
    if best_model_state is not None:
        model.classifier.load_state_dict(best_model_state)
        print(f"训练完成，使用最后一轮模型")
    
    return model

def evaluate_classifier(model, data, device, batch_size=8):
    """评估LLMFeatureClassifier模型"""
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
        prompts = [PROMPT.format(text)
                  for text in batch_texts]
        
        # 编码输入并移到GPU
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        # 预测
        with torch.no_grad():
            logits = model(inputs)
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
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--train",action="store_true", help="训练模型")
    parser.add_argument("--seed",type=int,default=42, help="随机数种子")
    
    args = parser.parse_args()
    set_seed(args.seed)
    MODEL=args.model
    MODEL_PATH = MODEL_PATHS[MODEL]
    PROMPT = f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{{}}\"" 
    PROMPT = '{}' # 不用提示词
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 加载模型和分词器
    print(f"加载模型: {MODEL_PATH}")
    # model, tokenizer = load_model_tok(device=device)
    # tokenizer.pad_token = tokenizer.eos_token
    # # 创建LLM特征分类器分训练集和验证集
    # llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)
    # # 确保模型的所有部分都在正确设备上
    # llm_classifier.classifier.to(device)

    if args.train:
        # 创建5个子数据集并分别训练模型
        print("创建5个子数据集并训练5个不同模型...")
        
        # 创建保存结果的目录
        os.makedirs('./5res', exist_ok=True)
        
        # 随机打乱所有训练数据
        all_train_data = load_jsonl(args.train_file)
        random.shuffle(all_train_data)
        
        # 划分为5个子集
        subset_size = len(all_train_data) // 5
        subsets = []
        for i in range(5):
            if i < 4:
                subsets.append(all_train_data[i*subset_size:(i+1)*subset_size])
            else:
                # 最后一个子集可能包含剩余的所有数据
                subsets.append(all_train_data[i*subset_size:])
        
        print(f"已将数据划分为5个子集，每个子集大小约为: {subset_size}")
        
        # 对每个子集单独训练模型
        for subset_idx, subset_data in enumerate(subsets):
            print(f"\n开始训练模型 {subset_idx+1}/5...")
            
            # 直接使用全部子集数据训练，不再划分验证集
            print(f"子集 {subset_idx+1} 训练数据量: {len(subset_data)}")
            
            # 重新创建模型（确保每个模型独立训练）
            subset_model, subset_tokenizer = load_model_tok(device=device)
            subset_tokenizer.pad_token = subset_tokenizer.eos_token
            subset_classifier = LLMFeatureClassifier(subset_model, subset_tokenizer, device, output_dim=2)
            subset_classifier.classifier.to(device)
            
            # 训练模型（传入空验证集，因为我们不用验证）
            subset_classifier = train_classifier(subset_classifier, subset_data, [], 
                                              device, batch_size=args.batch_size, num_epochs=1)
            
            # 保存模型
            subset_model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_subset_{subset_idx+1}.pt'
            torch.save(subset_classifier.classifier.state_dict(), subset_model_path)
            print(f"子集 {subset_idx+1} 模型参数已保存到 {subset_model_path}")
            
            # 加载测试数据进行预测
            test_data = load_jsonl(args.test_file)
            print(f"加载测试数据: {len(test_data)} 条")
            
            # 进行预测
            print(f"开始子集 {subset_idx+1} 模型预测...")
            predictions = []
            
            # 批量处理测试数据以加快预测速度
            test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda x: x)
            
            for batch in tqdm(test_dataloader, desc=f"子集 {subset_idx+1} 预测中", disable=not sys.stdout.isatty()):
                batch_texts = []
                batch_ids = []
                
                for item in batch:
                    batch_texts.append(item["text"])
                    batch_ids.append(item.get("id", ""))
                
                # 构建输入提示
                prompts = [ PROMPT.format(text)
                           for text in batch_texts]
                
                # 编码输入
                inputs = subset_classifier.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 预测
                with torch.no_grad():
                    logits = subset_classifier(inputs)
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # 收集预测结果
                predictions.extend(batch_predictions)
                
                # 释放GPU内存
                del inputs, logits, batch_predictions
                torch.cuda.empty_cache()
            
            # 将预测结果保存到5res/目录下
            with open(f'./5res/llm_{args.model}_subset_{subset_idx+1}.txt', 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            print(f"子集 {subset_idx+1} 预测结果已保存到 ./5res/llm_{args.model}_subset_{subset_idx+1}.txt，共 {len(predictions)} 条预测")
            
            # 释放模型内存
            del subset_classifier, subset_model, subset_tokenizer
            torch.cuda.empty_cache()
            sys.stdout.flush() # 确保输出被刷新
        
        print("所有5个子集模型训练和预测完成！")

    else:
        # 加载并测试5个模型
        print("加载并测试5个子集模型...")
        
        # 创建保存结果的目录
        os.makedirs('./5res', exist_ok=True)
        
        # 加载测试数据
        test_data = load_jsonl(args.test_file)
        print(f"加载测试数据: {len(test_data)} 条")
        
        # 对每个子集模型进行测试
        for subset_idx in range(5):
            subset_model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_subset_{subset_idx+1}.pt'
            
            if not os.path.exists(subset_model_path):
                print(f"警告: 子集 {subset_idx+1} 模型文件不存在: {subset_model_path}")
                continue
                
            print(f"\n加载子集 {subset_idx+1} 模型参数: {subset_model_path}")
            
            # 加载模型和分词器
            subset_model, subset_tokenizer = load_model_tok(device=device)
            subset_tokenizer.pad_token = subset_tokenizer.eos_token
            subset_classifier = LLMFeatureClassifier(subset_model, subset_tokenizer, device, output_dim=2)
            subset_classifier.classifier.to(device)
            
            # 加载保存的模型参数
            try:
                subset_classifier.classifier.load_state_dict(torch.load(subset_model_path, map_location=device))
                print(f"成功加载子集 {subset_idx+1} 模型参数")
            except Exception as e:
                print(f"加载子集 {subset_idx+1} 模型参数失败: {str(e)}")
                continue
            
            # 进行预测
            print(f"开始子集 {subset_idx+1} 模型预测...")
            predictions = []
            
            # 批量处理测试数据以加快预测速度
            test_dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda x: x)
            
            for batch in tqdm(test_dataloader, desc=f"子集 {subset_idx+1} 预测中", disable=not sys.stdout.isatty()):
                batch_texts = []
                batch_ids = []
                
                for item in batch:
                    batch_texts.append(item["text"])
                    batch_ids.append(item.get("id", ""))
                
                # 构建输入提示
                prompts = [PROMPT.format(text)
                           for text in batch_texts]
                
                # 编码输入
                inputs = subset_classifier.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 预测
                with torch.no_grad():
                    logits = subset_classifier(inputs)
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # 收集预测结果
                predictions.extend(batch_predictions)
                
                # 释放GPU内存
                del inputs, logits, batch_predictions
                torch.cuda.empty_cache()
            
            # 将预测结果保存到5res/目录下
            with open(f'./5res/llm_{args.model}_subset_{subset_idx+1}.txt', 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            print(f"子集 {subset_idx+1} 预测结果已保存到 ./5res/llm_{args.model}_subset_{subset_idx+1}.txt，共 {len(predictions)} 条预测")
            
            # 释放模型内存
            del subset_classifier, subset_model, subset_tokenizer
            torch.cuda.empty_cache()
            
        print("所有5个子集模型预测完成！")
