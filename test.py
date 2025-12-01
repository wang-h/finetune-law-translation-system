#!/usr/bin/env python3
"""
统一测试入口
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
from models.mt5 import MT5Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class QwenPredictor:
    def __init__(self, base_model_path, adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen from {base_model_path} with adapter {adapter_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 加载基座
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16
        )
        # 加载 LoRA
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        
    def translate(self, text, direction="zh2ja"):
        # 简化的 Prompt 构建
        target_lang = "Japanese" if "ja" in direction else "English"
        if "zh" not in direction[:2]: target_lang = "Chinese"
        
        instruction = f"Please translate the following text into {target_lang}."
        messages = [
            {"role": "system", "content": "You are a professional legal translator."},
            {"role": "user", "content": f"{instruction}\n\n{text}"}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='mt5', choices=['mt5', 'qwen'])
    parser.add_argument('--model_path', required=True, help='模型路径 (MT5是目录, Qwen是LoRA目录)')
    parser.add_argument('--base_model', default="Qwen/Qwen3-4B-Instruct-2507", help='Qwen基座模型')
    parser.add_argument('--text', help='测试文本')
    parser.add_argument('--direction', default='zh2ja')
    
    args = parser.parse_args()
    
    translator = None
    if args.model_type == 'mt5':
        # MT5 加载逻辑
        # 这里 trick 一下：MT5Trainer 默认初始化会加载默认模型，我们需要它加载我们微调的
        # 但 MT5Trainer 目前设计是 __init__ 里加载。
        # 我们可以实例化后覆盖，或者 hack 一下
        trainer = MT5Trainer(model_name=args.model_path, enable_visualization=False)
        translator = trainer
    else:
        translator = QwenPredictor(args.base_model, args.model_path)
        
    if args.text:
        print(f"原文: {args.text}")
        res = translator.translate(args.text, args.direction)
        print(f"译文: {res}")
    else:
        # 交互模式
        while True:
            t = input("Text (q to quit): ")
            if t == 'q': break
            res = translator.translate(t, args.direction)
            print(f"Target: {res}")

if __name__ == "__main__":
    main()
