import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import numpy as np
import torch
import random
import os
import sys
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

os.environ["HF_DATASETS_OFFLINE"] = "1" # 使用本地模型

class MLP_classifier(torch.nn.Module):
    """简单的MLP，添加了BatchNorm和Dropout正则化防止梯度爆炸"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(MLP_classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim//10)
        self.bn1 = torch.nn.BatchNorm1d(input_dim//10)
        self.fc2 = torch.nn.Linear(input_dim//10, input_dim//100)
        self.bn2 = torch.nn.BatchNorm1d(input_dim//100)
        self.fc3 = torch.nn.Linear(input_dim//100, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)  # 添加softmax用于软投票

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
        return x

    def predict_proba(self, x):
        """返回概率分布用于软投票"""
        logits = self.forward(x)
        return self.softmax(logits)


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
    """使用LLM的最后一层隐藏状态作为特征提取器，并训练一个MLP分类器"""
    def __init__(self, llm_model, tokenizer, device, output_dim=2):
        super(LLMFeatureClassifier, self).__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.device = device
        
        # 冻结LLM参数
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # 获取LLM最后一层隐藏层维度
        if hasattr(self.llm.config, 'hidden_size'):
            self.hidden_dim = self.llm.config.hidden_size
        elif hasattr(self.llm.config, 'n_embd'):
            self.hidden_dim = self.llm.config.n_embd
        elif hasattr(self.llm.config, 'dim'):
            self.hidden_dim = self.llm.config.dim
        else:
            self.hidden_dim = 4096
            print(f"无法直接获取隐藏层维度，使用默认值: {self.hidden_dim}")
        
        # MLP分类器
        self.classifier = MLP_classifier(self.hidden_dim, output_dim).to(device)
        self.hidden_states = None
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子函数以捕获最后一层隐藏状态"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hidden_states = output[0]
            else:
                self.hidden_states = output
        
        # 针对不同模型结构注册钩子
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
            last_layer = self.llm.model.layers[-1]
            last_layer.register_forward_hook(hook_fn)
        elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'blocks'):
            last_layer = self.llm.model.blocks[-1]
            last_layer.register_forward_hook(hook_fn)
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'h'):
            last_layer = self.llm.transformer.h[-1]
            last_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError("不支持的模型结构，无法注册钩子")
    
    def _get_pooled_output(self, hidden_states, attention_mask=None):
        """改进：使用平均池化代替最后token池化"""
        if attention_mask is not None:
            # 扩展attention_mask维度用于广播
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # 将padding位置的hidden_states清零
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            # 计算每个序列的实际长度
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            # 计算平均值
            pooled_output = sum_embeddings / sum_mask
        else:
            # 如果没有attention mask，则对所有token进行平均
            pooled_output = torch.mean(hidden_states, dim=1)
        
        return pooled_output
    
    def forward(self, inputs):
        """前向传递"""
        self.llm.eval()
        
        with torch.no_grad():
            _ = self.llm(**inputs)
        
        features = self._get_pooled_output(self.hidden_states, inputs.get('attention_mask'))
        features = features.to(self.device)
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, inputs):
        """返回概率分布用于软投票"""
        logits = self.forward(inputs)
        return self.classifier.softmax(logits)

def train_and_validate_classifier(model, train_data, val_data, device, batch_size=8, learning_rate=1e-4, num_epochs=1):
    """训练并验证LLMFeatureClassifier模型"""
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}/{num_epochs}", disable=not sys.stdout.isatty())
        
        for batch in pbar:
            batch_texts = [item["text"] for item in batch]
            batch_labels = [item["label"] for item in batch]
            
            prompts = [PROMPT.format(text) for text in batch_texts]
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
            
            del inputs, labels, logits, loss, predictions
            torch.cuda.empty_cache()
        
        train_acc = correct / total
        
        # 验证阶段
        if val_data:
            val_acc = evaluate_classifier(model, val_data, device, batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}")
    
    return model

def evaluate_classifier(model, data, device, batch_size=8):
    """评估LLMFeatureClassifier模型"""
    model.classifier.eval()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=lambda x: x)
    
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="验证中", disable=not sys.stdout.isatty()):
        batch_texts = [item["text"] for item in batch]
        batch_labels = [item["label"] for item in batch]
        
        prompts = [PROMPT.format(text) for text in batch_texts]
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += len(labels)
        
        del inputs, labels, logits, predictions
        torch.cuda.empty_cache()
    
    return correct / total

def bagging_predict(models, test_data, device, batch_size=8, use_soft_voting=True):
    """集成多个模型进行预测"""
    all_predictions = []
    all_probabilities = []
    
    for model_idx, model in enumerate(models):
        print(f"使用模型 {model_idx+1}/{len(models)} 进行预测...")
        model.classifier.eval()
        
        model_predictions = []
        model_probabilities = []
        
        dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=lambda x: x)
        
        for batch in tqdm(dataloader, desc=f"模型{model_idx+1}预测中", disable=not sys.stdout.isatty()):
            batch_texts = [item["text"] for item in batch]
            
            prompts = [PROMPT.format(text) for text in batch_texts]
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if use_soft_voting:
                    probs = model.predict_proba(inputs)
                    model_probabilities.extend(probs.cpu().numpy())
                
                logits = model(inputs)
                predictions = torch.argmax(logits, dim=-1)
                model_predictions.extend(predictions.cpu().numpy())
            
            del inputs, logits, predictions
            torch.cuda.empty_cache()
        
        all_predictions.append(model_predictions)
        if use_soft_voting:
            all_probabilities.append(model_probabilities)
    
    # 集成预测结果
    if use_soft_voting and all_probabilities:
        # 软投票：平均概率
        avg_probs = np.mean(all_probabilities, axis=0)
        final_predictions = np.argmax(avg_probs, axis=1)
        print("使用软投票进行集成")
    else:
        # 硬投票：多数决
        all_predictions = np.array(all_predictions)
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            vote_counts = Counter(votes)
            majority_vote = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)
        print("使用硬投票进行集成")
    
    return final_predictions

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
    
    import argparse
    parser = argparse.ArgumentParser(description="使用大语言模型预测文本分类")
    parser.add_argument("--model", type=str, default="llama-3-3b", choices=model_list, help="选择模型")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, default="test.jsonl", help="测试数据文件路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--num_models", type=int, default=5, help="集成模型数量")
    parser.add_argument("--soft_voting", action="store_true", help="使用软投票")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    MODEL = args.model
    MODEL_PATH = MODEL_PATHS[MODEL]
    PROMPT = '{}'  # 不用提示词
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"加载模型: {MODEL_PATH}")

    if args.train:
        print(f"创建{args.num_models}个自助采样数据集并训练模型...")
        os.makedirs('./5res', exist_ok=True)
        
        # 加载和准备数据
        all_train_data = load_jsonl(args.train_file)
        random.shuffle(all_train_data)
        
        # 划分验证集
        val_size = int(len(all_train_data) * args.val_ratio)
        val_data = all_train_data[:val_size]
        train_pool = all_train_data[val_size:]
        
        print(f"训练数据池大小: {len(train_pool)}, 验证集大小: {len(val_data)}")
        
        # 加载基础模型（只加载一次）
        base_model, base_tokenizer = load_model_tok(device=device)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
        trained_models = []
        
        # 训练多个模型
        for i in range(args.num_models):
            print(f"\n开始训练模型 {i+1}/{args.num_models}...")
            
            # 创建自助采样数据集
            if args.num_models==1:
                bootstrap_data = random.choices(train_pool, k=len(train_pool))
            else:
                bootstrap_data = random.choices(train_pool, k=int(len(train_pool)/args.num_models))
            print(f"第 {i+1} 个自助采样数据集大小: {len(bootstrap_data)}")
            
            # 创建分类器实例
            llm_classifier = LLMFeatureClassifier(base_model, base_tokenizer, device, output_dim=2)
            llm_classifier.classifier.to(device)
            
            # 训练模型
            llm_classifier = train_and_validate_classifier(
                llm_classifier, bootstrap_data, val_data,
                device, batch_size=args.batch_size, num_epochs=1
            )
            
            # 保存模型
            model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_bootstrap_{i+1}.pt'
            torch.save(llm_classifier.classifier.state_dict(), model_path)
            print(f"模型 {i+1} 参数已保存到 {model_path}")
            
            trained_models.append(llm_classifier)
        
        # 使用集成模型进行预测
        test_data = load_jsonl(args.test_file)
        print(f"\n开始集成预测，测试数据量: {len(test_data)}")
        
        final_predictions = bagging_predict(
            trained_models, test_data, device, 
            batch_size=args.batch_size, use_soft_voting=args.soft_voting
        )
        
        # 保存最终预测结果
        output_file = f'bagging_{args.model}_final.txt'
        with open(output_file, 'w') as f:
            for pred in final_predictions:
                f.write(f"{pred}\n")
        
        print(f"集成预测完成，结果已保存到 {output_file}")
        # print(f"预测分布 - 标签0: {final_predictions.count(0)}个，标签1: {final_predictions.count(1)}个")
        print(
            f"预测分布 - 标签0: {np.count_nonzero(final_predictions == 0)}个，"
            f"标签1: {np.count_nonzero(final_predictions == 1)}个"
        )
    else:
        # 加载已训练的模型进行预测
        print(f"加载{args.num_models}个已训练模型进行预测...")
        
        base_model, base_tokenizer = load_model_tok(device=device)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
        loaded_models = []
        
        for i in range(args.num_models):
            model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_bootstrap_{i+1}.pt'
            
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}")
                continue
            
            llm_classifier = LLMFeatureClassifier(base_model, base_tokenizer, device, output_dim=2)
            llm_classifier.classifier.load_state_dict(torch.load(model_path, map_location=device))
            llm_classifier.classifier.to(device)
            loaded_models.append(llm_classifier)
            print(f"成功加载模型 {i+1}")
        
        if loaded_models:
            test_data = load_jsonl(args.test_file)
            final_predictions = bagging_predict(
                loaded_models, test_data, device,
                batch_size=args.batch_size, use_soft_voting=args.soft_voting
            )
            
            output_file = f'bagging_{args.model}_final.txt'
            with open(output_file, 'w') as f:
                for pred in final_predictions:
                    f.write(f"{pred}\n")
            
            print(f"集成预测完成，结果已保存到 {output_file}")
