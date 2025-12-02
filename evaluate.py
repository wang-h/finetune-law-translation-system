#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
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
        # Qwen 评估：加载基座 + LoRA，计算 BLEU
        base_model = args.base_model or "Qwen/Qwen3-4B-Instruct-2507"
        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA adapter: {args.model_path}")
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # 加载基座模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载 LoRA adapter
        model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
        
        # 加载测试数据
        test_df = load_json_data(args.test_json)
        
        # 确定目标语言
        target_lang = args.lang_pair.split('-')[1]
        target_lang_name = {"en": "English", "ja": "Japanese", "zh": "Chinese"}.get(target_lang, "English")
        
        predictions = []
        references = []
        
        print(f"Evaluating on {len(test_df)} samples...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            source = row['chinese']
            reference = row['japanese']  # 这里其实是 target，字段名是历史遗留
            
            # 构建 prompt
            instruction = f"Please translate the following text into {target_lang_name}."
            messages = [
                {"role": "system", "content": "You are a professional legal translator."},
                {"role": "user", "content": f"{instruction}\n\n{source}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=args.max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 只取生成的部分
            generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
            
            # 打印前几个样本
            if idx < 3:
                print(f"\n[Sample {idx+1}]")
                print(f"Source: {source[:100]}...")
                print(f"Reference: {reference[:100]}...")
                print(f"Prediction: {prediction[:100]}...")
        
        # 计算 BLEU
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        print(f"\n{'='*50}")
        print(f"Final Test BLEU: {bleu.score:.2f}")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Translation Model')
    parser.add_argument('--model_type', required=True, choices=['mt5', 'nllb', 'qwen'])
    parser.add_argument('--model_path', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test_json', required=True, help='Path to test dataset json')
    parser.add_argument('--lang_pair', default='zh-en', help='Language pair, e.g. zh-en, zh-ja')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--base_model', default="Qwen/Qwen3-4B-Instruct-2507", help='Base model for Qwen (required for LoRA)')
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()

