import argparse
from bert_classifer.train import train_model
from predict import predict

def main():
    parser = argparse.ArgumentParser(description='BERT文本分类系统')
    
    # 添加全局选项以支持 --train 和 --predict 直接调用模式
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', action='store_true', help='使用模型进行预测')
    
    # 通用训练参数
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='训练数据文件路径')
    parser.add_argument('--test_file', type=str, default='test.jsonl', help='测试数据文件路径')
    parser.add_argument('--output_dir', type=str, default='model_output', help='模型输出目录')
    parser.add_argument('--pretrained_path', type=str, default='/root/autodl-tmp/bert-base-uncased/', help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')   
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    
    # 通用预测参数
    parser.add_argument('--input', type=str, default='test.jsonl', help='预测输入文件路径')
    parser.add_argument('--output', type=str, default='predictions.jsonl', help='预测输出文件路径')
    parser.add_argument('--model_params', type=str, default='model_output/bert_classifier_best_params.pt', help='模型参数文件路径')
    

    args = parser.parse_args()
    
    # 判断执行模式：先判断子命令，再判断参数标记
    if args.train:
        # 训练模型
        train_file = args.train_file if hasattr(args, 'train_file') else args.train 
        test_file = args.test_file if hasattr(args, 'test_file') else args.test
        
        best_model_path = train_model(
            train_file=train_file,
            test_file=test_file,
            output_dir=args.output_dir,
            pretrained_path=args.pretrained_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr
        )
        
        print(f"训练完成，最佳模型参数路径: {best_model_path}")
        
    elif args.predict:
        # 模型预测
        predict(
            input_file=args.input,
            output_file=args.output,
            model_params_path=args.model_params,
            pretrained_path=args.pretrained_path,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
