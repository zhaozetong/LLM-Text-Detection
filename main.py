import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import random
from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from classifier import MLP_classifier, TransformerClassifier, ResNetClassifier, RandomForestWrapper
from data_utils import set_seed, create_non_overlapping_subsets, load_precomputed_features

def train_mlp(model, train_features, train_labels, val_features=None, val_labels=None, 
              batch_size=64, learning_rate=1e-3, num_epochs=10, device='cpu'):
    """训练MLP分类器"""
    
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)
        
        train_acc = correct / total
        scheduler.step()
        
        # 验证阶段
        if val_features is not None and val_labels is not None:
            val_acc = evaluate_mlp(model, val_features, val_labels, batch_size, device)
            print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}")
    
    return model

def evaluate_mlp(model, features, labels, batch_size=64, device='cpu'):
    """评估MLP分类器"""
    model.eval()
    
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_features)
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)
    
    return correct / total

def predict_with_ensemble(models, model_types, test_features, batch_size=64, device='cpu', use_soft_voting=True):
    """使用集成模型进行预测 - 支持多种模型类型"""
    all_predictions = []
    all_probabilities = []
    
    test_dataset = TensorDataset(test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for model_idx, (model, model_type) in enumerate(zip(models, model_types)):
        print(f"使用模型 {model_idx+1}/{len(models)} ({model_type}) 进行预测...")
        model.eval()
        if model_type != 'random_forest':
            model.to(device)
        
        model_predictions = []
        model_probabilities = []
        
        if model_type == 'random_forest':
            # 随机森林直接预测整个数据集
            predictions = model.predict(test_features)
            model_predictions = predictions.numpy().tolist()
            
            if use_soft_voting:
                probs = model.predict_proba(test_features)
                model_probabilities = probs.numpy().tolist()
        else:
            # 深度学习模型批量预测
            with torch.no_grad():
                for (batch_features,) in tqdm(test_dataloader, desc=f"模型{model_idx+1}预测中"):
                    batch_features = batch_features.to(device)
                    
                    logits = model(batch_features)
                    predictions = torch.argmax(logits, dim=-1)
                    model_predictions.extend(predictions.cpu().numpy())
                    
                    if use_soft_voting:
                        probs = model.predict_proba(batch_features)
                        model_probabilities.extend(probs.cpu().numpy())
        
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

def add_feature_noise(features, noise_factor=0.01):
    """添加特征噪声进行数据增强"""
    noise = torch.randn_like(features) * noise_factor
    return features + noise

def feature_normalization(features):
    """特征标准化"""
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-8)

def create_model(model_type, input_dim, output_dim=2, **kwargs):
    """创建指定类型的模型"""
    if model_type == 'mlp':
        return MLP_classifier(input_dim, output_dim, **kwargs)
    elif model_type == 'transformer':
        return TransformerClassifier(input_dim, output_dim, **kwargs)
    elif model_type == 'resnet':
        return ResNetClassifier(input_dim, output_dim, **kwargs)
    elif model_type == 'random_forest':
        return RandomForestWrapper(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_model(model, model_type, train_features, train_labels, val_features=None, val_labels=None, 
              batch_size=64, learning_rate=1e-3, num_epochs=10, device='cpu'):
    """训练分类器 - 支持多种模型类型"""
    
    if model_type == 'random_forest':
        # 随机森林训练
        print("训练随机森林模型...")
        model.fit(train_features, train_labels)
        
        # 计算训练准确率
        train_pred = model.predict(train_features)
        train_acc = (train_pred == train_labels).float().mean().item()
        print(f"随机森林训练准确率: {train_acc:.4f}")
        
        # 验证准确率
        if val_features is not None and val_labels is not None:
            val_pred = model.predict(val_features)
            val_acc = (val_pred == val_labels).float().mean().item()
            print(f"随机森林验证准确率: {val_acc:.4f}")
        
        return model
    
    else:
        # 深度学习模型训练
        train_dataset = TensorDataset(train_features, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        
        model.to(device)
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
            
            train_acc = correct / total
            scheduler.step()
            
            # 验证阶段
            if val_features is not None and val_labels is not None:
                val_acc = evaluate_model(model, model_type, val_features, val_labels, batch_size, device)
                print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, 训练准确率: {train_acc:.4f}")
        
        return model

def evaluate_model(model, model_type, features, labels, batch_size=64, device='cpu'):
    """评估分类器 - 支持多种模型类型"""
    
    if model_type == 'random_forest':
        predictions = model.predict(features)
        accuracy = (predictions == labels).float().mean().item()
        return accuracy
    
    else:
        model.eval()
        
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_features)
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        return correct / total

def main():
    parser = argparse.ArgumentParser(description="基于预计算特征训练集成分类器")
    parser.add_argument("--model", type=str, default='llama-3-3b', help="特征模型名称")
    parser.add_argument("--classifier", type=str, default='mlp', 
                       choices=['mlp', 'transformer', 'resnet', 'random_forest'], 
                       help="分类器类型")
    parser.add_argument("--feature_dir", type=str, default="/root/autodl-tmp/checkpoint/features", help="特征文件目录")
    parser.add_argument("--checkpoint_dir", type=str, default="/root/autodl-tmp/checkpoint", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--num_models", type=int, default=5, help="集成模型数量")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--soft_voting", action="store_true", help="使用软投票")
    parser.add_argument("--train", action="store_true", help="训练模式")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.classifier == 'random_forest':
        device = 'cpu'  # 随机森林只能用CPU
    print(f"使用设备: {device}")
    
    # 加载预计算特征
    train_feature_path = os.path.join(args.feature_dir, f"{args.model}_train_features.pt")
    test_feature_path = os.path.join(args.feature_dir, f"{args.model}_test_features.pt")
    
    print("加载预计算特征...")
    train_features, train_labels, hidden_dim = load_precomputed_features(train_feature_path)
    test_features, test_labels, _ = load_precomputed_features(test_feature_path)
    
    print(f"训练特征形状: {train_features.shape}, 测试特征形状: {test_features.shape}")
    print(f"隐藏层维度: {hidden_dim}")
    
    # 特征标准化 (随机森林可能不需要，但保持一致性)
    if args.classifier != 'random_forest':
        train_features = feature_normalization(train_features)
        test_features = feature_normalization(test_features)
    
    if args.train:
        print(f"开始训练 {args.num_models} 个{args.classifier}分类器...")
        
        # 划分验证集
        val_size = int(len(train_features) * args.val_ratio)
        indices = list(range(len(train_features)))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_pool_indices = indices[val_size:]
        
        val_features = train_features[val_indices]
        val_labels = train_labels[val_indices]
        
        print(f"训练池大小: {len(train_pool_indices)}, 验证集大小: {len(val_indices)}")
        
        # 创建重叠度较小的数据子集
        subset_indices_list = create_non_overlapping_subsets(
            train_pool_indices, args.num_models, subset_ratio=0.3, labels=train_labels
        )
        
        trained_models = []
        model_types = []
        
        # 训练多个模型
        for i, subset_indices in enumerate(subset_indices_list):
            print(f"\n训练模型 {i+1}/{args.num_models}...")
            
            # 获取子集数据
            subset_features = train_features[subset_indices]
            subset_labels = train_labels[subset_indices]
            
            # 创建模型
            if args.classifier == 'random_forest':
                model = create_model(args.classifier, hidden_dim, 
                                   n_estimators=100, max_depth=10, random_state=args.seed+i)
            else:
                model = create_model(args.classifier, hidden_dim, output_dim=2)
            
            # 数据增强 (仅对深度学习模型)
            if args.classifier != 'random_forest':
                augmented_features = add_feature_noise(subset_features, noise_factor=0.3)
                augmented_labels = subset_labels.clone()
            else:
                augmented_features = subset_features
                augmented_labels = subset_labels
            
            model = train_model(
                model, args.classifier, augmented_features, augmented_labels,
                val_features, val_labels,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                device=device
            )
            
            # 保存模型
            model_path = os.path.join(args.checkpoint_dir, f"{args.classifier}_{args.model}_bootstrap_{i+1}.pt")
            if args.classifier == 'random_forest':
                import pickle
                model_path = model_path.replace('.pt', '.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model.model, f)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"模型 {i+1} 已保存到 {model_path}")
            
            trained_models.append(model)
            model_types.append(args.classifier)
        
        # 集成预测
        print(f"\n开始集成预测...")
        final_predictions = predict_with_ensemble(
            trained_models, model_types, test_features, 
            batch_size=args.batch_size, device=device, 
            use_soft_voting=args.soft_voting
        )
        
        n0=0
        n1=0
        # 保存预测结果
        output_file = f'{args.model}_{args.classifier}.txt'
        with open(output_file, 'w') as f:
            for pred in final_predictions:
                f.write(f"{pred}\n")
                if pred == 0:
                    n0 += 1
                else:
                    n1 += 1
        
        print(f"集成预测完成，结果已保存到 {output_file}")
        print(f"预测分布 - 标签0: {n0}个，标签1: {n1}个")
    
    else:
        # 加载已训练模型进行预测
        print(f"加载 {args.num_models} 个已训练{args.classifier}模型进行预测...")
        
        loaded_models = []
        model_types = []
        for i in range(args.num_models):
            if args.classifier == 'random_forest':
                model_path = os.path.join(args.checkpoint_dir, f"{args.classifier}_{args.model}_bootstrap_{i+1}.pkl")
                if not os.path.exists(model_path):
                    print(f"警告: 模型文件不存在: {model_path}")
                    continue
                
                import pickle
                with open(model_path, 'rb') as f:
                    rf_model = pickle.load(f)
                wrapper = RandomForestWrapper()
                wrapper.model = rf_model
                loaded_models.append(wrapper)
            else:
                model_path = os.path.join(args.checkpoint_dir, f"{args.classifier}_{args.model}_bootstrap_{i+1}.pt")
                if not os.path.exists(model_path):
                    print(f"警告: 模型文件不存在: {model_path}")
                    continue
                
                model = create_model(args.classifier, hidden_dim, output_dim=2)
                model.to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                loaded_models.append(model)
            
            model_types.append(args.classifier)
            print(f"成功加载模型 {i+1}")
        
        if loaded_models:
            final_predictions = predict_with_ensemble(
                loaded_models, model_types, test_features,
                batch_size=args.batch_size, device=device,
                use_soft_voting=args.soft_voting
            )
            n0=0
            n1=0
            output_file = f'{args.model}_{args.classifier}.txt'
            with open(output_file, 'w') as f:
                for pred in final_predictions:
                    f.write(f"{pred}\n")
                    if pred == 0:
                        n0 += 1
                    else:
                        n1 += 1
            
            print(f"集成预测完成，结果已保存到 {output_file}")
            print(f"预测分布 - 标签0: {n0}个，标签1: {n1}个")

if __name__ == "__main__":
    main()
