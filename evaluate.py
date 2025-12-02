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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # 生成时左填充
        
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
        
        # 预处理所有 prompts
        all_prompts = []
        all_references = []
        instruction = f"Please translate the following text into {target_lang_name}."
        
        for idx, row in test_df.iterrows():
            source = row['chinese']
            reference = row['japanese']
            
            messages = [
                {"role": "system", "content": "You are a professional legal translator."},
                {"role": "user", "content": f"{instruction}\n\n{source}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(text)
            all_references.append(reference)
        
        # 批量处理
        batch_size = args.batch_size
        num_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        print(f"Evaluating on {len(test_df)} samples with batch_size={batch_size}...")
        
        for batch_idx in tqdm(range(num_batches), desc="Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_prompts = all_prompts[start_idx:end_idx]
            
            # 批量编码
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 解码每个生成结果
            for i, (input_ids, gen_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
                # 只取生成的部分（去掉输入）
                input_len = (input_ids != tokenizer.pad_token_id).sum().item()
                new_tokens = gen_ids[input_len:]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(prediction)
                
                # 打印前几个样本
                global_idx = start_idx + i
                if global_idx < 3:
                    print(f"\n[Sample {global_idx+1}]")
                    print(f"Source: {test_df.iloc[global_idx]['chinese'][:100]}...")
                    print(f"Reference: {all_references[global_idx][:100]}...")
                    print(f"Prediction: {prediction[:100]}...")
        
        references = all_references
        
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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--base_model', default="Qwen/Qwen3-4B-Instruct-2507", help='Base model for Qwen (required for LoRA)')
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()

