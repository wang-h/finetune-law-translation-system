
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Adafactor
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import sacrebleu
import numpy as np
try:
    from visualization import TrainingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, source_lang='zh', target_lang='ja'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 根据语言方向设置输入和目标
        if self.source_lang == 'zh' and self.target_lang == 'ja':
            source_text = f"zh2ja: {row['chinese']}"
            target_text = row['japanese']
        elif self.source_lang == 'ja' and self.target_lang == 'zh':
            source_text = f"ja2zh: {row['japanese']}"
            target_text = row['chinese']
        elif self.source_lang == 'zh' and self.target_lang == 'en':
            source_text = f"zh2en: {row['chinese']}"
            target_text = row['japanese']
        elif self.source_lang == 'en' and self.target_lang == 'zh':
            source_text = f"en2zh: {row['japanese']}"
            target_text = row['chinese']
        else:
            # 默认 fallback
            source_text = f"{self.source_lang}2{self.target_lang}: {row.get('chinese', '') or row.get('source', '')}"
            target_text = row.get('japanese', '') or row.get('target', '')
        
        # 编码输入
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class MT5Trainer:
    def __init__(self, model_name="K024/mt5-zh-ja-en-trimmed", max_length=512, 
                 enable_visualization=True, enable_tensorboard=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"MT5 使用设备: {self.device}")
        
        # 加载模型和分词器
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        except Exception as e:
            print(f"加载模型失败，尝试默认模型: {e}")
            default_model = "K024/mt5-zh-ja-en-trimmed"
            self.tokenizer = T5Tokenizer.from_pretrained(default_model, legacy=True)
            self.model = MT5ForConditionalGeneration.from_pretrained(default_model)
            
        self.model.to(self.device)
        
        self.max_length = max_length
        self.model_name = model_name
        
        # 初始化可视化器
        self.visualizer = None
        if VISUALIZATION_AVAILABLE and enable_visualization:
            self.visualizer = TrainingVisualizer(
                output_dir="./training_logs",
                enable_realtime=enable_visualization,
                enable_tensorboard=enable_tensorboard
            )
    
    def load_data_from_json(self, train_json, test_json, lang_pair="zh-ja"):
        """从JSON加载数据"""
        def load_json_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            entries = data.get('entries', [])
            result = []
            for entry in entries:
                source = entry.get('source', '').strip()
                target = entry.get('target', '').strip()
                if source and target:
                    result.append({'source': source, 'target': target})
            return result
        
        train_data = load_json_file(train_json)
        test_data = load_json_file(test_json)
        
        # 统一转为 DataFrame 格式，统一列名为 chinese/japanese (为了兼容旧代码逻辑)
        # 这里 chinese 对应 source, japanese 对应 target
        train_df = pd.DataFrame([{
            'chinese': item['source'],
            'japanese': item['target']
        } for item in train_data])
        
        test_df = pd.DataFrame([{
            'chinese': item['source'],
            'japanese': item['target']
        } for item in test_data])
        
        return train_df, test_df
        
    def create_datasets(self, train_df, test_df, val_ratio=0.1, lang_pair='zh-ja'):
        """创建数据集"""
        # 简单处理：包含双向
        all_train_data = []
        all_val_data = []
        all_test_data = []
        
        if lang_pair == 'zh-ja':
            directions = [('zh', 'ja'), ('ja', 'zh')]
        elif lang_pair == 'zh-en':
            directions = [('zh', 'en'), ('en', 'zh')]
        else:
            directions = [('zh', 'ja')] # 默认

        for source_lang, target_lang in directions:
            # 训练集划分验证集
            if val_ratio > 0:
                t_df, v_df = train_test_split(train_df, test_size=val_ratio, random_state=42)
            else:
                t_df, v_df = train_df, pd.DataFrame()
            
            train_ds = TranslationDataset(t_df, self.tokenizer, self.max_length, source_lang, target_lang)
            val_ds = TranslationDataset(v_df, self.tokenizer, self.max_length, source_lang, target_lang)
            test_ds = TranslationDataset(test_df, self.tokenizer, self.max_length, source_lang, target_lang)
            
            all_train_data.extend([train_ds[i] for i in range(len(train_ds))])
            all_val_data.extend([val_ds[i] for i in range(len(val_ds))])
            all_test_data.extend([test_ds[i] for i in range(len(test_ds))])
            
        return {
            'train': all_train_data,
            'val': all_val_data,
            'test': all_test_data
        }

    def create_dataloader(self, dataset, batch_size=8, shuffle=True):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors='pt'
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def compute_bleu(self, dataset, max_samples=None):
        """计算数据集的 BLEU 分数"""
        self.model.eval()
        
        preds = []
        labels = []
        
        # 如果指定了最大样本数，随机采样
        if max_samples and max_samples < len(dataset):
            import random
            indices = random.sample(range(len(dataset)), max_samples)
            subset = [dataset[i] for i in indices]
        else:
            subset = dataset
            
        print(f"计算 BLEU (样本数: {len(subset)})...")
        
        # 创建临时 DataLoader
        dataloader = self.create_dataloader(subset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                generated_tokens = self.model.generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.max_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True
                )
                
                # 解码预测结果
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                preds.extend(decoded_preds)
                
                # 解码标签 (处理 -100)
                label_ids = batch['labels'].cpu().numpy()
                label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                labels.extend(decoded_labels)
        
        # 计算 BLEU
        # 对于中文/日文，使用 zh tokenizer (按字切分) 以获得更合理的 BLEU 分数
        # 如果是纯日文目标，推荐使用 'ja-mecab'，这里为了兼容混合方向使用 'zh'
        bleu = sacrebleu.corpus_bleu(preds, [labels], tokenize='zh')
        print(f"BLEU: {bleu.score:.2f}")
        return bleu.score

    def train(self, datasets, output_dir="./mt5_finetuned", 
              batch_size=8, learning_rate=5e-5, num_epochs=3, 
              gradient_accumulation_steps=4, **kwargs):
        
        # ... (之前的 DataLoader 代码) ...
        train_dataloader = self.create_dataloader(datasets['train'], batch_size, shuffle=True)
        val_dataloader = self.create_dataloader(datasets['val'], batch_size, shuffle=False)
        
        # 切换优化器：尝试使用 Adafactor 以节省显存
        # optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        optimizer = Adafactor(
            self.model.parameters(), 
            lr=learning_rate, 
            relative_step=False, 
            scale_parameter=False, 
            warmup_init=False
        )
        
        total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)
        
        print(f"开始训练 {self.model_name}...")
        # ... (打印信息) ...
        
        global_step = 0
        self.model.zero_grad()
        
        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            # ... (训练循环) ...
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps 
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
                
                if self.visualizer and global_step % 10 == 0 and (step + 1) % gradient_accumulation_steps == 0:
                     self.visualizer.log_step(global_step, epoch+1, loss.item() * gradient_accumulation_steps, scheduler.get_last_lr()[0])
                     self.visualizer.update_plot()
            
            # Epoch 结束：计算验证集 BLEU
            print(f"\nEpoch {epoch+1} 结束，评估验证集 BLEU...")
            val_bleu = self.compute_bleu(datasets['val'], max_samples=100) # 采样 100 条快速评估
            print(f"Validation BLEU: {val_bleu:.2f}")
            
            # Save per epoch
            self.save_model(f"{output_dir}/epoch-{epoch+1}")
            
        self.save_model(f"{output_dir}/final")
        
        # 训练结束：计算测试集完整 BLEU
        print("\n训练完成，评估测试集完整 BLEU...")
        test_bleu = self.compute_bleu(datasets['test'])
        print(f"Test Set BLEU: {test_bleu:.2f}")
        
        return {'test_bleu': test_bleu}

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"模型已保存: {output_dir}")

    def translate(self, text, direction="zh2ja", max_length=200):
        self.model.eval()
        input_text = f"{direction}: {text}"
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

