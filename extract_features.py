import json
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def load_model_tok(model_path, device):
    """加载模型和分词器"""
    safetensors_exist = any(f.endswith(".safetensors") for f in os.listdir(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_safetensors=safetensors_exist,
        torch_dtype=torch.float16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class FeatureExtractor:
    """LLM特征提取器"""
    def __init__(self, model, tokenizer, device, layer_indices=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 如果没有指定层索引，默认使用最后3层
        if layer_indices is None:
            layer_indices = [-3, -2, -1]
        self.layer_indices = layer_indices
        
        self.hidden_states_list = []  # 存储指定层的隐藏状态
        self._register_hooks()
        
        # 获取隐藏层维度
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'n_embd'):
            self.hidden_dim = self.model.config.n_embd
        elif hasattr(self.model.config, 'dim'):
            self.hidden_dim = self.model.config.dim
        else:
            self.hidden_dim = 4096
            print(f"无法直接获取隐藏层维度，使用默认值: {self.hidden_dim}")
    
    def _get_total_layers(self):
        """获取模型总层数"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            return len(self.model.model.blocks)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("不支持的模型结构，无法获取层数")
    
    def _normalize_layer_indices(self, layer_indices, total_layers):
        """将负数索引转换为正数索引，并验证索引有效性"""
        normalized_indices = []
        for idx in layer_indices:
            if idx < 0:
                normalized_idx = total_layers + idx
            else:
                normalized_idx = idx
            
            if normalized_idx < 0 or normalized_idx >= total_layers:
                raise ValueError(f"层索引 {idx} 超出范围 [0, {total_layers-1}] 或 [{-total_layers}, -1]")
            
            normalized_indices.append(normalized_idx)
        
        return normalized_indices
    
    def _register_hooks(self):
        """注册前向钩子函数以捕获指定层的隐藏状态"""
        total_layers = self._get_total_layers()
        normalized_indices = self._normalize_layer_indices(self.layer_indices, total_layers)
        
        # 初始化隐藏状态列表
        self.hidden_states_list = [None] * len(normalized_indices)
        
        def create_hook_fn(storage_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                self.hidden_states_list[storage_idx] = hidden_state
            return hook_fn
        
        # 针对不同模型结构注册钩子
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            for storage_idx, layer_idx in enumerate(normalized_indices):
                layers[layer_idx].register_forward_hook(create_hook_fn(storage_idx))
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            blocks = self.model.model.blocks
            for storage_idx, layer_idx in enumerate(normalized_indices):
                blocks[layer_idx].register_forward_hook(create_hook_fn(storage_idx))
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            for storage_idx, layer_idx in enumerate(normalized_indices):
                layers[layer_idx].register_forward_hook(create_hook_fn(storage_idx))
        else:
            raise ValueError("不支持的模型结构，无法注册钩子")
        
        print(f"已注册钩子到层: {normalized_indices} (原始输入: {self.layer_indices})")

    def _get_pooled_output(self, attention_mask=None):
        """获取池化后的特征表示，对多层隐藏状态求平均"""
        # 收集所有有效的隐藏状态
        valid_hidden_states = [hs for hs in self.hidden_states_list if hs is not None]
        
        if not valid_hidden_states:
            raise ValueError("没有捕获到有效的隐藏状态")
        
        # 对多层隐藏状态求平均
        averaged_hidden_states = torch.stack(valid_hidden_states, dim=0).mean(dim=0)
        
        if attention_mask is not None:
            # 找到每个序列最后一个非填充token的位置
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = averaged_hidden_states.shape[0]
            pooled_output = averaged_hidden_states[torch.arange(batch_size), last_token_indices]
        else:
            # 否则直接使用序列最后一个位置
            pooled_output = averaged_hidden_states[:, -1]
        
        return pooled_output
    
    def extract_features(self, texts, batch_size=8, max_length=512):
        """批量提取特征"""
        self.model.eval()
        features = []
        
        # 使用完全相同的任务描述提示
        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        
        for batch_texts in tqdm(dataloader, desc="提取特征中"):
            # 使用相同的prompt格式
            prompts = [f"Determine if the following text was written by a human (0) or generated by an AI language model (1).\n\nText: \"{text}\"" 
                      for text in batch_texts]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 清空之前的隐藏状态
                self.hidden_states_list = [None] * len(self.layer_indices)
                _ = self.model(**inputs)
            
            # 使用修改后的特征提取方法
            batch_features = self._get_pooled_output(inputs.get('attention_mask'))
            features.append(batch_features.cpu())
            
            del inputs, batch_features
            torch.cuda.empty_cache()
        
        return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser(description="预计算LLM特征")
    # 手动可以设置model
    parser.add_argument("--model", type=str, default='llama-3-3b', help="模型名称")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="训练数据文件")
    parser.add_argument("--test_file", type=str, default="test.jsonl", help="测试数据文件")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/checkpoint/features", help="特征保存目录")
    args = parser.parse_args()
    layer_indices = [10,11]#中间层
    # 解析层索引

    model_paths = {
        'llama-2-7b': "/root/.cache/huggingface/hub/Llama-2-7b-chat-hf",
        'llama-2-13b': "/root/autodl-tmp/Llama-2-13b-chat-hf",
        'llama-3-8b': "/root/autodl-tmp/Llama-3.1-8B-Instruct",
        'qwen-7b': '/root/autodl-tmp/Qwen2.5-7B-Instruct/',
        'llama-3-3b': '/root/autodl-tmp/Llama-3.2-3B-Instruct/',
        'mistral-7b': '/root/autodl-tmp/Mistral-7B-Instruct-v0.3/',
    }
    
    model_path = model_paths[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"加载模型: {model_path}")
    model, tokenizer = load_model_tok(model_path, device)
    
    extractor = FeatureExtractor(model, tokenizer, device, layer_indices=layer_indices)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练数据
    print("处理训练数据...")
    train_data = load_jsonl(args.train_file)
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    
    train_features = extractor.extract_features(train_texts, args.batch_size)
    
    # 保存训练特征
    train_save_path = os.path.join(args.output_dir, f"{args.model}_train_features.pt")
    torch.save({
        'features': train_features,
        'labels': torch.tensor(train_labels),
        'hidden_dim': extractor.hidden_dim
    }, train_save_path)
    print(f"训练特征已保存到: {train_save_path}")
    
    # 处理测试数据
    print("处理测试数据...")
    test_data = load_jsonl(args.test_file)
    test_texts = [item["text"] for item in test_data]
    
    test_features = extractor.extract_features(test_texts, args.batch_size)
    
    # 保存测试特征
    test_save_path = os.path.join(args.output_dir, f"{args.model}_test_features.pt")
    torch.save({
        'features': test_features,
        'hidden_dim': extractor.hidden_dim
    }, test_save_path)
    print(f"测试特征已保存到: {test_save_path}")
    
    print(f"特征提取完成! 使用层索引{layer_indices}. 训练特征形状: {train_features.shape}, 测试特征形状: {test_features.shape}")

if __name__ == "__main__":
    main()
