
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
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # Qwen3/2.5 推荐使用 ChatML 格式或 apply_chat_template
        # 这里我们手动构建 system/user/assistant 消息
        
        target_lang_name = "Japanese" if "ja" in self.lang_pair else "English"
        if self.lang_pair == "ja-zh" or self.lang_pair == "en-zh":
             target_lang_name = "Chinese"
             
        instruction = f"Please translate the following text into {target_lang_name}."
        
        messages = [
            {"role": "system", "content": "You are a professional legal translator."},
            {"role": "user", "content": f"{instruction}\n\n{source}"},
            {"role": "assistant", "content": target}
        ]
        
        # 使用 tokenizer 的模板应用功能
        # 记得 Qwen 的 chat template 会自动处理特殊 token
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        # 制作 Labels
        # 我们只希望计算 Assistant 回复部分的 Loss
        # 这是一个简化的处理：直接将 input_ids 作为 labels
        # (更严谨的做法是把 user 部分的 labels 设为 -100，这里为了代码简洁先略过，
        #  或者使用 DataCollatorForCompletionOnlyLM，但这里我们手写 Trainer)
        labels = input_ids.clone()
        
        # 简单的 Masking: 找到 assistant start token 的位置
        # 但 Qwen 的 template 比较复杂，简单起见我们全量训练（指令微调通常也接受这样）
        # 或者利用 tokenizer.apply_chat_template 的生成结果来做掩码
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class QwenTrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", max_length=512, **kwargs):
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
        
        print("加载 Qwen 模型 (4-bit 量化以节省显存)...")
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        except ImportError:
            print("未安装 bitsandbytes，尝试全量加载（显存可能不足）")
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 配置 LoRA
        print("配置 LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] # Qwen 常用 LoRA 模块
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 准备数据
        # 这里的 datasets 是从 load_data_from_json 返回的 DataFrame
        train_df = datasets['train']
        # 简单划分验证集
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        train_ds = QwenDataset(train_df, self.tokenizer, self.max_length)
        val_ds = QwenDataset(val_df, self.tokenizer, self.max_length)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            optim="paged_adamw_32bit"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
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

