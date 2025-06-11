import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

class EnhancedFeatureExtractor:
    def __init__(self, model, tokenizer, device, num_layers=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = num_layers
        self.hidden_states_list = []
        self._register_hooks()
        
        # 初始化TF-IDF提取器
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
    def extract_enhanced_features(self, texts, batch_size=8, max_length=512):
        """提取增强特征：LLM特征 + 统计特征 + TF-IDF特征"""
        
        # 1. 提取LLM特征（多种池化策略）
        llm_features = self._extract_llm_features(texts, batch_size, max_length)
        
        # 2. 提取统计特征
        stat_features = self._extract_statistical_features(texts)
        
        # 3. 提取TF-IDF特征
        tfidf_features = self._extract_tfidf_features(texts)
        
        # 4. 特征组合
        combined_features = torch.cat([
            llm_features,
            torch.tensor(stat_features, dtype=torch.float32),
            torch.tensor(tfidf_features, dtype=torch.float32)
        ], dim=1)
        
        return combined_features
    
    def _extract_llm_features(self, texts, batch_size, max_length):
        """提取多种LLM特征"""
        self.model.eval()
        features_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 使用多种prompt策略
            prompts = self._create_diverse_prompts(batch_texts)
            
            batch_features = []
            for prompt_batch in prompts:
                inputs = self.tokenizer(
                    prompt_batch, return_tensors="pt", 
                    padding=True, truncation=True, max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    self.hidden_states_list = [None] * self.num_layers
                    outputs = self.model(**inputs)
                
                # 多种池化策略
                features = self._multi_pooling(inputs.get('attention_mask'))
                batch_features.append(features)
            
            # 平均多个prompt的特征
            avg_features = torch.stack(batch_features).mean(dim=0)
            features_list.append(avg_features.cpu())
        
        return torch.cat(features_list, dim=0)
    
    def _create_diverse_prompts(self, texts):
        """创建多样化的prompt"""
        prompt_templates = [
            "Analyze if this text is human-written (0) or AI-generated (1): \"{text}\"",
            "Is the following text written by a human (0) or an AI (1)? Text: \"{text}\"",
            "Classify the authorship - Human (0) or AI (1): \"{text}\"",
        ]
        
        prompts = []
        for template in prompt_templates:
            prompt_batch = [template.format(text=text) for text in texts]
            prompts.append(prompt_batch)
        
        return prompts
    
    def _multi_pooling(self, attention_mask):
        """多种池化策略组合"""
        valid_hidden_states = [hs for hs in self.hidden_states_list if hs is not None]
        averaged_hidden_states = torch.stack(valid_hidden_states, dim=0).mean(dim=0)
        
        # 1. Last token pooling
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_size = averaged_hidden_states.shape[0]
        last_pooled = averaged_hidden_states[torch.arange(batch_size), last_token_indices]
        
        # 2. Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(averaged_hidden_states.size())
        mean_pooled = (averaged_hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)
        
        # 3. Max pooling
        max_pooled = torch.max(averaged_hidden_states * mask_expanded + 
                              (1.0 - mask_expanded) * -1e9, dim=1)[0]
        
        # 4. Attention pooling (简化版)
        attention_weights = torch.softmax(averaged_hidden_states.sum(-1), dim=1)
        attention_pooled = (averaged_hidden_states * attention_weights.unsqueeze(-1)).sum(1)
        
        # 组合所有池化结果
        combined = torch.cat([last_pooled, mean_pooled, max_pooled, attention_pooled], dim=1)
        return combined
    
    def _extract_statistical_features(self, texts):
        """提取统计特征"""
        features = []
        for text in texts:
            stat_feat = [
                len(text),  # 文本长度
                len(text.split()),  # 词数
                len(set(text.split())),  # 唯一词数
                text.count('.'),  # 句号数
                text.count(','),  # 逗号数
                text.count('!'),  # 感叹号数
                text.count('?'),  # 问号数
                np.mean([len(word) for word in text.split()]),  # 平均词长
                len([w for w in text.split() if len(w) > 6]),  # 长词数
                text.count(' ') / len(text) if len(text) > 0 else 0,  # 空格密度
            ]
            features.append(stat_feat)
        
        return np.array(features)
    
    def _extract_tfidf_features(self, texts):
        """提取TF-IDF特征"""
        try:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            return tfidf_matrix.toarray()
        except:
            # 如果TF-IDF失败，返回零特征
            return np.zeros((len(texts), 1000))