#!/usr/bin/env python3
"""
统一训练入口
支持 MT5 和 Qwen 模型微调
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import sys
from models.mt5 import MT5Trainer
try:
    from models.qwen import QwenTrainer
    QWEN_AVAILABLE = True
except ImportError as e:
    print(f"注意: Qwen 模块不可用 ({e})，仅支持 MT5")
    QWEN_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description='法律翻译模型微调 (MT5 / Qwen)')
    
    # 模型选择
    parser.add_argument('--model_type', default='mt5', choices=['mt5', 'qwen', 'nllb'],
                       help='选择模型类型: mt5 (Seq2Seq), qwen (LLM) 或 nllb (Facebook)')
    parser.add_argument('--base_model', default=None,
                       help='指定基础模型路径或名称 (例如 Qwen/Qwen3-4B-Instruct-2507)')
    
    # 数据参数
    parser.add_argument('--lang_pair', default='zh-ja', choices=['zh-ja', 'zh-en'],
                       help='语言对')
    parser.add_argument('--train_json', default=None, help='训练集JSON路径')
    parser.add_argument('--test_json', default=None, help='测试集JSON路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--output_dir', default='./checkpoints', help='输出目录')
    parser.add_argument('--enable_tensorboard', action='store_true', help='启用 TensorBoard 可视化')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 默认数据路径
    lang_suffix = 'en' if 'en' in args.lang_pair else 'ja'
    
    if not args.train_json:
        args.train_json = f'datasets/my_train_{lang_suffix}.json'
        
    if not args.test_json:
        args.test_json = f'datasets/my_test_{lang_suffix}.json'
    
    print("="*50)
    print(f"🚀 开始训练: {args.model_type.upper()}")
    print(f"📂 数据集: {args.train_json}")
    print("="*50)
    
    if args.model_type == 'qwen':
        if not QWEN_AVAILABLE:
            print("❌ Qwen 环境未就绪，请安装 peft, bitsandbytes")
            sys.exit(1)
            
        model_name = args.base_model or "Qwen/Qwen3-4B-Instruct-2507"
        trainer = QwenTrainer(model_name=model_name, max_length=args.max_length)

        # 加载数据
        train_df, test_df = trainer.load_data_from_json(args.train_json, args.test_json, args.lang_pair)
        datasets = {'train': train_df, 'test': test_df, 'lang_pair': args.lang_pair}
        
        # 训练
        trainer.train(
            datasets, 
            output_dir=f"{args.output_dir}/qwen",
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr if args.lr != 5e-5 else 2e-4  # LLM LoRA 通常学习率大一点
        )

    elif args.model_type == 'nllb':
        from models.nllb import NLLBTrainer
        # NLLB 流程
        model_name = args.base_model or "facebook/nllb-200-distilled-600M"
        trainer = NLLBTrainer(model_name=model_name, max_length=args.max_length, enable_tensorboard=args.enable_tensorboard)
        
        train_df, test_df = trainer.load_data_from_json(args.train_json, args.test_json, args.lang_pair)
        datasets = trainer.create_datasets(train_df, test_df, lang_pair=args.lang_pair)
        
        trainer.train(
            datasets,
            output_dir=f"{args.output_dir}/nllb",
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )
        
    else:
        # MT5 流程
        model_name = args.base_model or "K024/mt5-zh-ja-en-trimmed"
        trainer = MT5Trainer(model_name=model_name, max_length=args.max_length, enable_tensorboard=args.enable_tensorboard)
        
        train_df, test_df = trainer.load_data_from_json(args.train_json, args.test_json, args.lang_pair)
        datasets = trainer.create_datasets(train_df, test_df, lang_pair=args.lang_pair)
        
        trainer.train(
            datasets,
            output_dir=f"{args.output_dir}/mt5",
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )

if __name__ == "__main__":
    main()
