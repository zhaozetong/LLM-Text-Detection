import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from model import set_seed, TextClassificationDataset, load_jsonl, create_model, save_model_params

def train_model(train_file='train.jsonl', test_file='test.jsonl', output_dir='model_output', 
               pretrained_path='bert-base-uncased', batch_size=16, epochs=3, learning_rate=2e-5):
    # 确定设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(42)
    
    # 加载数据
    print("加载数据...")
    train_data = load_jsonl(train_file)
    test_data = load_jsonl(test_file)
    
    # 将训练数据划分为训练集和验证集
    train_subset, val_data = train_test_split(train_data, test_size=0.8, random_state=42, shuffle=True)
    print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_data)}")
    
    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    
    # 创建数据集和数据加载器
    print("准备数据集...")
    train_dataset = TextClassificationDataset(train_subset, tokenizer)
    test_dataset = TextClassificationDataset(test_data, tokenizer)
    val_dataset = TextClassificationDataset(val_data, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 加载预训练的BERT模型
    print("加载BERT模型...")
    model = create_model(num_labels=2, pretrained_path=pretrained_path)
    model.to(device)
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
    # 设置训练参数
    total_steps = len(train_dataloader) * epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 创建模型保存目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\n开始第 {epoch + 1}/{epochs} 轮训练")
        
        # 训练模式
        model.train()
        total_loss = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 清零梯度
            model.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 计算平均损失
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 验证模式
        model.eval()
        val_preds = []
        val_labels = []
        
        # 使用预先划分好的验证集
        for batch in tqdm(val_dataloader, desc="验证中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds)
        
        print(f"验证集结果 - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
        
        # 如果当前模型表现更好，则保存模型参数
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = f"{output_dir}/bert_classifier_best_params.pt"
            save_model_params(model, model_save_path)
    
    print(f"训练完成！最佳F1分数: {best_f1:.4f}")
    
    return f"{output_dir}/bert_classifier_best_params.pt"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练文本分类模型')
    parser.add_argument('--train', type=str, default='train.jsonl', help='训练数据文件路径')
    parser.add_argument('--test', type=str, default='test.jsonl', help='测试数据文件路径')
    parser.add_argument('--output_dir', type=str, default='model_output', help='模型输出目录')
    parser.add_argument('--pretrained_path', type=str, default='bert-base-uncased', help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    
    args = parser.parse_args()
    
    train_model(
        train_file=args.train,
        test_file=args.test,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
