#!/usr/bin/env python3
import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.mt5 import MT5Trainer, TranslationDataset
from models.nllb import NLLBTrainer
try:
    from models.qwen import QwenTrainer
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data.get('entries', [])
    # 统一格式
    return pd.DataFrame([{
        'chinese': item['source'],
        'japanese': item['target']
    } for item in entries])

def evaluate_model(args):
    print(f"Loading {args.model_type} model from {args.model_path}...")
    
    if args.model_type == 'mt5':
        # 加载 MT5
        trainer = MT5Trainer(model_name=args.model_path, enable_visualization=False)
        # Hack: 因为 trainer 默认会加载 tokenizer，我们直接用它即可
        
        # 加载数据
        test_df = load_json_data(args.test_json)
        
        # 确定语言方向
        source_lang = 'zh' if 'zh' in args.lang_pair.split('-')[0] else args.lang_pair.split('-')[0]
        target_lang = args.lang_pair.split('-')[1]
        
        # 创建 Dataset 和 DataLoader
        test_ds = TranslationDataset(test_df, trainer.tokenizer, max_length=args.max_length, 
                                   source_lang=source_lang, target_lang=target_lang)
        
        # 调用 trainer 的 compute_bleu
        # compute_bleu 内部已经处理了 DataLoader 和 batch 生成
        score = trainer.compute_bleu(test_ds)
        print(f"Final Test BLEU: {score:.2f}")
        
    elif args.model_type == 'nllb':
        # 加载 NLLB
        trainer = NLLBTrainer(model_name=args.model_path, enable_visualization=False)
        test_df = load_json_data(args.test_json)
        
        from models.nllb import NLLBDataset
        # 需要获取 lang_codes
        source_lang = 'zh' if 'zh' in args.lang_pair.split('-')[0] else args.lang_pair.split('-')[0]
        target_lang = args.lang_pair.split('-')[1]
        
        test_ds = NLLBDataset(test_df, trainer.tokenizer, max_length=args.max_length, 
                            source_lang=source_lang, target_lang=target_lang, lang_codes=trainer.lang_codes)
        
        score = trainer.compute_bleu(test_ds)
        print(f"Final Test BLEU: {score:.2f}")

    elif args.model_type == 'qwen':
        if not QWEN_AVAILABLE:
            print("Qwen not available")
            return
            
        # Qwen 评估逻辑略有不同，因为它没有标准的 compute_bleu 方法在 Trainer 里
        # 这里我们需要复用 QwenPredictor 或者手动写一下
        pass

def main():
    parser = argparse.ArgumentParser(description='Evaluate Translation Model')
    parser.add_argument('--model_type', required=True, choices=['mt5', 'nllb', 'qwen'])
    parser.add_argument('--model_path', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test_json', required=True, help='Path to test dataset json')
    parser.add_argument('--lang_pair', default='zh-en', help='Language pair, e.g. zh-en, zh-ja')
    parser.add_argument('--max_length', type=int, default=256)
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()

