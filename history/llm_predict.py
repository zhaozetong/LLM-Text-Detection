import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import random
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

os.environ["HF_DATASETS_OFFLINE"] = "1"


class MLP_classifier(torch.nn.Module):
    """
    简单的MLP，添加了BatchNorm和Dropout正则化防止梯度爆炸

    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(MLP_classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 10)
        self.bn1 = torch.nn.BatchNorm1d(input_dim // 10)
        self.fc2 = torch.nn.Linear(input_dim // 10, input_dim // 100)
        self.bn2 = torch.nn.BatchNorm1d(input_dim // 100)
        self.fc3 = torch.nn.Linear(input_dim // 100, output_dim)
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
                                                torch_dtype=torch.float16).to(device)  # 加载bin 文件
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


def split_train_val_data(data, val_ratio=0.2):
    """将数据集随机打乱并划分为训练集和验证集"""
    data_copy = data.copy()
    random.shuffle(data_copy)
    val_size = int(len(data_copy) * val_ratio)
    train_data = data_copy[val_size:]
    val_data = data_copy[:val_size]
    return train_data, val_data


def extract_features(model, data, device, batch_size=8, is_test=False):
    """提取特征的通用函数"""
    model.start_recording()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=lambda x: x)

    for batch in tqdm(dataloader, desc="提取特征"):
        batch_texts = []
        batch_labels = []

        for item in batch:
            batch_texts.append(item["text"])
            if not is_test and "label" in item:
                batch_labels.append(item["label"])

        prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\""
                  for text in batch_texts]

        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if batch_labels:
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            with torch.no_grad():
                _, _ = model(inputs, labels)
        else:
            with torch.no_grad():
                _, _ = model(inputs)

        del inputs
        if batch_labels:
            del labels
        torch.cuda.empty_cache()

    model.stop_recording()
    return model.get_recorded_features()


def train_with_real_time_features(model, train_data, val_data, device, batch_size=8, learning_rate=1e-4, num_epochs=1):
    """使用实时特征提取进行训练"""
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}/{num_epochs}", disable=not sys.stdout.isatty())

        for batch in pbar:
            batch_texts = [item["text"] for item in batch]
            batch_labels = [item["label"] for item in batch]

            prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\""
                      for text in batch_texts]

            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            logits, _ = model(inputs, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

            batch_acc = (predictions == labels).float().mean().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch_acc': f"{batch_acc:.4f}"
            })

            del inputs, labels, logits, loss, predictions
            torch.cuda.empty_cache()

        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, 损失: {total_loss / len(train_dataloader):.4f}, 训练准确率: {epoch_acc:.4f}")

        val_acc = evaluate_with_real_time_features(model, val_data, device, batch_size) if val_data else 0.8
        print(f"验证准确率: {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.classifier.state_dict().copy()

    if best_model_state is not None:
        model.classifier.load_state_dict(best_model_state)
        print(f"使用最佳验证准确率 {best_val_acc:.4f} 的模型")

    return model


def train_with_precomputed_features(train_feature_path, model_path, device, batch_size=8, learning_rate=1e-4, num_epochs=1):
    """使用预计算特征进行训练"""
    if not os.path.exists(train_feature_path):
        raise FileNotFoundError(f"训练特征文件不存在: {train_feature_path}")

    print(f"加载预计算的训练特征: {train_feature_path}")
    train_feature_data = torch.load(train_feature_path)
    train_features = train_feature_data['features']
    train_labels = train_feature_data['labels']
    hidden_dim = train_feature_data['hidden_dim']

    classifier = MLP_classifier(hidden_dim, 2).to(device)
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    print("使用预计算特征开始训练...")
    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}")
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

            batch_acc = (predictions == labels).float().mean().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch_acc': f"{batch_acc:.4f}"
            })

        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}, 损失: {total_loss / len(train_dataloader):.4f}, 训练准确率: {epoch_acc:.4f}")

    torch.save(classifier.state_dict(), model_path)
    print(f"模型参数已保存到 {model_path}")


def evaluate_with_real_time_features(model, data, device, batch_size=8):
    """使用实时特征提取进行评估"""
    if not data:
        return 0.8

    model.classifier.eval()
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=lambda x: x)

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader, desc="评估中", disable=not sys.stdout.isatty()):
        batch_texts = [item["text"] for item in batch]
        batch_labels = [item["label"] for item in batch]

        prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\""
                  for text in batch_texts]

        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            logits, _ = model(inputs)
            predictions = torch.argmax(logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += len(labels)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        del inputs, labels, logits, predictions
        torch.cuda.empty_cache()

    accuracy = correct / total
    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_predictions))
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=["人类撰写", "AI生成"]))

    return accuracy


def predict_with_precomputed_features(test_feature_path, model_path, device, batch_size=8, output_file=None):
    """使用预计算特征进行预测"""
    if not os.path.exists(test_feature_path):
        raise FileNotFoundError(f"测试特征文件不存在: {test_feature_path}")

    print(f"加载预计算的测试特征: {test_feature_path}")
    test_feature_data = torch.load(test_feature_path)
    test_features = test_feature_data['features']
    hidden_dim = test_feature_data['hidden_dim']

    classifier = MLP_classifier(hidden_dim, 2).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    print("使用预计算特征开始预测...")
    predictions = []
    test_dataloader = DataLoader(test_features, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_features in tqdm(test_dataloader, desc="预测中"):
            batch_features = batch_features.to(device)
            logits = classifier(batch_features)
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)

    if output_file:
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        print(f"预测结果已保存到 {output_file}")

    return predictions


def predict_with_real_time_features(model, test_data, device, batch_size=8, output_file=None):
    """使用实时特征提取进行预测"""
    model.eval()
    predictions = []
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=lambda x: x)

    for batch in tqdm(test_dataloader, desc="预测中"):
        batch_texts = [item["text"] for item in batch]

        prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\""
                  for text in batch_texts]

        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits, _ = model(inputs)
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        predictions.extend(batch_predictions)
        del inputs
        torch.cuda.empty_cache()

    if output_file:
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        print(f"预测结果已保存到 {output_file}")

    return predictions


if __name__ == "__main__":
    model_list = ['llama-2-7b', 'llama-2-13b', 'llama-3-8b', 'qwen-7b', 'llama-3-3b', 'mistral-7b']
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
    parser.add_argument("--use_pre_features", action="store_true", help="使用预计算的特征进行训练和预测")
    parser.add_argument("--pre_features", action="store_true", help="预计算并保存特征")

    args = parser.parse_args()
    set_seed(args.seed)

    MODEL_PATH = MODEL_PATHS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置文件路径
    feature_dir = '/root/autodl-tmp/checkpoint/features'
    train_feature_path = os.path.join(feature_dir, f"{args.model}_train_features.pt")
    test_feature_path = os.path.join(feature_dir, f"{args.model}_test_features.pt")
    model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{args.model}_best_params.pt'
    output_file = f'llm_{args.model}.txt'

    # 模式1: 预计算特征
    if args.pre_features:
        print("=" * 50)
        print("预计算特征模式")
        print("=" * 50)

        print(f"加载模型: {MODEL_PATH}")
        model, tokenizer = load_model_tok(device=device)
        tokenizer.pad_token = tokenizer.eos_token

        train_data = load_jsonl(args.train_file)
        test_data = load_jsonl(args.test_file)
        print(f"训练数据量: {len(train_data)}, 测试数据量: {len(test_data)}")

        llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)

        # 提取并保存训练特征
        print("提取训练特征...")
        extract_features(llm_classifier, train_data, device, args.batch_size)
        llm_classifier.save_recorded_features(train_feature_path, args.model)

        # 提取并保存测试特征
        print("提取测试特征...")
        extract_features(llm_classifier, test_data, device, args.batch_size, is_test=True)
        llm_classifier.save_recorded_features(test_feature_path, args.model)

        print("特征预计算完成！")

    # 模式2: 使用预计算特征
    elif args.use_pre_features:
        print("=" * 50)
        print("使用预计算特征模式")
        print("=" * 50)

        if args.train:
            print("使用预计算特征进行训练...")
            train_with_precomputed_features(train_feature_path, model_path, device, args.batch_size)

        print("使用预计算特征进行预测...")
        predict_with_precomputed_features(test_feature_path, model_path, device, args.batch_size, output_file)

    # 模式3: 实时特征提取
    else:
        print("=" * 50)
        print("实时特征提取模式")
        print("=" * 50)

        print(f"加载模型: {MODEL_PATH}")
        model, tokenizer = load_model_tok(device=device)
        tokenizer.pad_token = tokenizer.eos_token

        if args.train:
            print("开始训练...")
            train_data = load_jsonl(args.train_file)
            train_data, val_data = split_train_val_data(train_data, val_ratio=0.2)
            print(f"训练数据量: {len(train_data)}, 验证数据量: {len(val_data)}")

            llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)
            llm_classifier = train_with_real_time_features(llm_classifier, train_data, val_data, device, args.batch_size)

            torch.save(llm_classifier.classifier.state_dict(), model_path)
            print(f"模型参数已保存到 {model_path}")

        print("开始预测...")
        test_data = load_jsonl(args.test_file)
        print(f"测试数据量: {len(test_data)}")

        llm_classifier = LLMFeatureClassifier(model, tokenizer, device, output_dim=2)
        llm_classifier.classifier.load_state_dict(torch.load(model_path, map_location=device))

        predict_with_real_time_features(llm_classifier, test_data, device, args.batch_size, output_file)
