import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from model import TextClassificationDataset, load_jsonl, create_model, load_model_params

def predict(input_file, output_file, model_params_path, pretrained_path='bert-base-uncased', batch_size=16):
    # 确定设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载数据: {input_file}")
    test_data = load_jsonl(input_file)
    
    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    
    # 创建数据集和数据加载器
    test_dataset = TextClassificationDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载预训练的BERT模型和参数
    print(f"加载模型参数: {model_params_path}")
    model = create_model(num_labels=2, pretrained_path=pretrained_path)
    model = load_model_params(model, model_params_path)
    model.to(device)
    model.eval()
    
    # 在测试集上进行预测
    test_preds = []
    
    for batch in tqdm(test_dataloader, desc="生成预测"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
    
    # 保存预测结果
    predictions = []
    for i, pred in enumerate(test_preds):
        predictions.append({
            'id': i,
            'label': int(pred)
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"预测结果已保存到 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='预测文本分类')
    parser.add_argument('--input', type=str, default='test.jsonl', help='输入文件路径')
    parser.add_argument('--output', type=str, default='predictions.jsonl', help='输出文件路径')
    parser.add_argument('--model_params', type=str, default='model_output/bert_classifier_best_params.pt', help='模型参数文件路径')
    parser.add_argument('--pretrained_path', type=str, default='bert-base-uncased', help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    predict(
        args.input,
        args.output,
        args.model_params,
        args.pretrained_path,
        args.batch_size
    )
