
import os
import torch
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import pandas as pd
from sklearn.model_selection import train_test_split
import sacrebleu
from tqdm import tqdm


class BLEUCallback(TrainerCallback):
    """每个 epoch 结束后计算 BLEU 分数"""
    
    def __init__(self, val_df, tokenizer, lang_pair, max_length=256, sample_size=100):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.lang_pair = lang_pair
        self.max_length = max_length
        self.sample_size = sample_size  # 采样数量，避免评估太慢
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
            
        print(f"\n📊 Epoch {int(state.epoch)} 结束，计算 BLEU...")
        
        # 采样验证集
        sample_df = self.val_df.sample(n=min(self.sample_size, len(self.val_df)), random_state=42)
        
        target_lang = self.lang_pair.split('-')[1] if '-' in self.lang_pair else 'en'
        target_lang_name = {"en": "English", "ja": "Japanese", "zh": "Chinese"}.get(target_lang, "English")
        instruction = f"Please translate the following text into {target_lang_name}."
        
        predictions = []
        references = []
        
        model.eval()
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Eval BLEU"):
            source = row['source']
            reference = row['target']
            
            messages = [
                {"role": "system", "content": "You are a professional legal translator."},
                {"role": "user", "content": f"{instruction}\n\n{source}"}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 只取生成的部分
            new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
            prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
        
        # 计算 BLEU - 根据目标语言选择合适的 tokenizer
        # 日语用 'ja-mecab'，中文用 'zh'，英语用默认 '13a'
        if target_lang == 'ja':
            bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='ja-mecab')
            tokenizer_name = 'ja-mecab'
        elif target_lang == 'zh':
            bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='zh')
            tokenizer_name = 'zh'
        else:
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            tokenizer_name = '13a'
        print(f"✅ Epoch {int(state.epoch)} Validation BLEU: {bleu.score:.2f} (tokenize={tokenizer_name})")
        
        model.train()

class QwenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, lang_pair='zh-ja'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_pair = lang_pair
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        source = item['source']
        target = item['target']
        
        # 构建 Prompt
        target_lang_name = "Japanese" if "ja" in self.lang_pair else "English"
        if self.lang_pair == "ja-zh" or self.lang_pair == "en-zh":
             target_lang_name = "Chinese"
             
        instruction = f"Please translate the following text into {target_lang_name}."
        
        # 分别构建 prompt 部分和 response 部分
        messages_prompt = [
            {"role": "system", "content": "You are a professional legal translator."},
            {"role": "user", "content": f"{instruction}\n\n{source}"}
        ]
        
        # 获取 prompt 部分（不含 assistant 回复）
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True  # 添加 assistant 开始标记
        )
        
        # 完整文本（含 assistant 回复）
        messages_full = messages_prompt + [{"role": "assistant", "content": target}]
        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码 prompt 以获取长度
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        
        # 编码完整文本
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        
        # 关键：只对 assistant 回复部分计算 loss，prompt 部分设为 -100
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100  # 忽略 prompt 部分的 loss
        
        # padding 部分也设为 -100
        pad_token_id = self.tokenizer.pad_token_id
        for i in range(len(labels)):
            if input_ids[i] == pad_token_id:
                labels[i] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class QwenTrainer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Instruct-2507", max_length=512, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.max_length = max_length
        print(f"Qwen 初始化: {model_name} on {self.device}")
        
        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data_from_json(self, train_json, test_json, lang_pair="zh-ja"):
        # 复用简单的加载逻辑
        def load(path):
            with open(path, 'r', encoding='utf-8') as f:
                return pd.DataFrame(json.load(f)['entries'])
        
        train_df = load(train_json)
        test_df = load(test_json)
        return train_df, test_df

    def train(self, datasets, output_dir="./qwen_finetuned", batch_size=4, num_epochs=3, learning_rate=2e-4, **kwargs):
        
        print("加载 Qwen 模型 (bf16 精度)...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 配置 LoRA - Qwen3 的 target_modules（增强版）
        print("配置 LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 增加秩，提升表达能力
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj"      # MLP 层（翻译任务重要）
            ],
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 准备数据
        # 这里的 datasets 是从 load_data_from_json 返回的 DataFrame
        train_df = datasets['train']
        lang_pair = datasets.get('lang_pair', 'zh-ja')  # 获取语言对
        
        # 简单划分验证集
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        train_ds = QwenDataset(train_df, self.tokenizer, self.max_length, lang_pair=lang_pair)
        val_ds = QwenDataset(val_df, self.tokenizer, self.max_length, lang_pair=lang_pair)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            optim="adamw_torch"
        )
        
        # 创建 BLEU 评估回调
        bleu_callback = BLEUCallback(
            val_df=val_df, 
            tokenizer=self.tokenizer, 
            lang_pair=lang_pair,
            max_length=self.max_length,
            sample_size=100  # 每 epoch 采样 100 条计算 BLEU
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            callbacks=[bleu_callback]
        )
        
        print("开始训练 Qwen...")
        trainer.train()
        
        print(f"保存 LoRA 权重到 {output_dir}/final")
        model.save_pretrained(f"{output_dir}/final")
        self.tokenizer.save_pretrained(f"{output_dir}/final")

    def translate(self, text, direction="zh2ja", max_length=512):
        # 推理逻辑
        # 如果是加载后的 LoRA 模型，需要先加载基座再加载 LoRA
        # 这里假设 self.model 已经加载好了（如果是刚刚训练完）
        # 如果是重新运行脚本，需要专门的加载逻辑
        pass # 在 main.py 或 test.py 中处理加载

