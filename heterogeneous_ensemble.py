from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

class HeterogeneousEnsemble:
    def __init__(self, feature_dim):
        self.models = {
            'transformer': TransformerClassifier(feature_dim),
            'resnet': ResNetClassifier(feature_dim),
            'mlp': MLP_classifier(feature_dim, 2),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gbm': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        # 元学习器
        self.meta_learner = LogisticRegression()
        
    def train_base_models(self, X_train, y_train, X_val, y_val, device='cpu'):
        """训练基模型"""
        base_predictions = {}
        
        # 训练深度学习模型
        for name in ['transformer', 'resnet', 'mlp']:
            print(f"训练 {name}...")
            model = self.models[name]
            model = self._train_deep_model(model, X_train, y_train, X_val, y_val, device)
            self.models[name] = model
            
            # 获取验证集预测
            model.eval()
            with torch.no_grad():
                val_probs = F.softmax(model(X_val.to(device)), dim=1)
                base_predictions[name] = val_probs[:, 1].cpu().numpy()
        
        # 训练传统机器学习模型
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        X_val_np = X_val.numpy()
        
        for name in ['rf', 'gbm', 'xgb', 'lgb', 'svm', 'lr']:
            print(f"训练 {name}...")
            self.models[name].fit(X_train_np, y_train_np)
            val_probs = self.models[name].predict_proba(X_val_np)[:, 1]
            base_predictions[name] = val_probs
        
        # 训练元学习器
        meta_features = np.column_stack(list(base_predictions.values()))
        self.meta_learner.fit(meta_features, y_val.numpy())
        
    def predict(self, X_test, device='cpu'):
        """集成预测"""
        base_predictions = {}
        
        # 深度学习模型预测
        for name in ['transformer', 'resnet', 'mlp']:
            model = self.models[name]
            model.eval()
            with torch.no_grad():
                test_probs = F.softmax(model(X_test.to(device)), dim=1)
                base_predictions[name] = test_probs[:, 1].cpu().numpy()
        
        # 传统模型预测
        X_test_np = X_test.numpy()
        for name in ['rf', 'gbm', 'xgb', 'lgb', 'svm', 'lr']:
            test_probs = self.models[name].predict_proba(X_test_np)[:, 1]
            base_predictions[name] = test_probs
        
        # 元学习器预测
        meta_features = np.column_stack(list(base_predictions.values()))
        final_predictions = self.meta_learner.predict(meta_features)
        
        return final_predictions