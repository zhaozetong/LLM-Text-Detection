from llm_predict import *
from mlp_classifier import MLP_classifier as mlp_c
import torch

MODEL = 'llama-3-3b'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文件路径
model_path = f'/root/autodl-tmp/checkpoint/llm_classifier_{MODEL}_best_params.pt'
test_feature_path = f'/root/autodl-tmp/checkpoint/features/{MODEL}_test_features.pt'

print("=== 1. 验证模型参数加载 ===")
# 加载保存的参数
saved_state = torch.load(model_path, map_location=device)
# 创建MLP分类器并加载参数
mlp = mlp_c(3072, 2)  # 使用你导入的mlp_c
mlp.to(device)
mlp.load_state_dict(saved_state)  # 加载保存的参数
mlp.eval()  # 设置为评估模式

print("\n✅ 参数加载成功！")


print("\n=== 2. 加载预提取特征并进行预测 ===")

# 检查特征文件是否存在
if not os.path.exists(test_feature_path):
    raise ValueError(f"❌ 特征文件不存在: {test_feature_path}")

else:
    # 加载预计算的测试特征
    print(f"加载预计算的测试特征: {test_feature_path}")
    test_feature_data = torch.load(test_feature_path, map_location=device)
    test_features = test_feature_data['features']
    hidden_dim = test_feature_data['hidden_dim']
    
    print(f"特征形状: {test_features.shape}")
    print(f"隐藏层维度: {hidden_dim}")
    
    # 验证维度匹配
    if hidden_dim != 3072:
        print(f"⚠️  警告：隐藏层维度不匹配！特征: {hidden_dim}, 模型: 3072")
    
    # 进行预测
    print("\n开始预测...")
    predictions = []
    probabilities = []
    
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_features, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch_features in tqdm(test_dataloader, desc="预测中"):
            batch_features = batch_features.to(device)
            
            logits = mlp(batch_features)

            # print('feature',batch_features[:,:3])
            # print('logits',logits[:])
            # print('bias',mlp.fc1.bias[:5])
            # exit()


            
            # 获取预测标签
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
            
            # 获取预测概率
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.extend(batch_probs)
    
    print(f"\n✅ 预测完成！共 {len(predictions)} 条预测")
    
    # 统计预测结果
    n0 = sum(1 for p in predictions if p == 0)
    n1 = sum(1 for p in predictions if p == 1)
    print(f"预测分布 - 人类撰写(0): {n0}个, AI生成(1): {n1}个")
    
    # 保存预测结果
    output_file = f'test_predictions_{MODEL}.txt'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"预测结果已保存到: {output_file}")
    
    # 显示一些预测样例（包含概率）
    print("\n预测样例（前10个）：")
    for i in range(min(10, len(predictions))):
        prob_human = probabilities[i][0]
        prob_ai = probabilities[i][1]
        print(f"样本{i+1}: 预测={predictions[i]} (人类:{prob_human:.3f}, AI:{prob_ai:.3f})")

print("\n=== 测试完成 ===")