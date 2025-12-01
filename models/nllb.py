import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import numpy as np
import sacrebleu
from .mt5 import MT5Trainer, TranslationDataset  # 复用 MT5 的一些组件

class NLLBTrainer(MT5Trainer):
    """
    NLLB (No Language Left Behind) 模型训练器
    继承自 MT5Trainer，重写加载和部分逻辑
    """
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", max_length=512, 
                 enable_visualization=True, enable_tensorboard=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"NLLB 使用设备: {self.device}")
        
        # NLLB 需要特殊的 Tokenizer 加载方式
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"加载 NLLB 模型失败: {e}")
            raise e
            
        self.model.to(self.device)
        
        # 启用梯度检查点以节省显存
        self.model.gradient_checkpointing_enable()
        print("已启用梯度检查点 (Gradient Checkpointing) 以节省显存")
        
        self.max_length = max_length
        self.model_name = model_name
        
        # NLLB 的语言代码映射
        # 中文通常是 zho_Hans, 日文是 jpn_Jpan, 英文是 eng_Latn
        self.lang_codes = {
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'en': 'eng_Latn'
        }
        
        # 初始化可视化器 (复用父类逻辑)
        # super().__init__ 会再次初始化 MT5Trainer，导致重复加载模型和覆盖属性
        # 我们只需要调用 MT5Trainer 的初始化逻辑中除了模型加载以外的部分
        # 但由于 Python 的 super 机制，这里直接调用有点麻烦。
        # 最简单的办法是：不调用 super().__init__，而是手动设置剩下的属性
        
        self.visualizer = None
        if enable_visualization:
             from visualization import TrainingVisualizer
             self.visualizer = TrainingVisualizer(
                output_dir="./training_logs",
                enable_realtime=enable_visualization,
                enable_tensorboard=enable_tensorboard
            )

    def create_datasets(self, train_df, test_df, val_ratio=0.1, lang_pair='zh-ja'):
        # 简单处理：包含双向
        all_train_data = []
        all_val_data = []
        all_test_data = []
        
        from sklearn.model_selection import train_test_split
        
        if lang_pair == 'zh-ja':
            directions = [('zh', 'ja'), ('ja', 'zh')]
        elif lang_pair == 'zh-en':
            directions = [('zh', 'en'), ('en', 'zh')]
        else:
            directions = [('zh', 'ja')] # 默认

        for source_lang, target_lang in directions:
            if val_ratio > 0:
                t_df, v_df = train_test_split(train_df, test_size=val_ratio, random_state=42)
            else:
                t_df, v_df = train_df, pd.DataFrame()
            
            # 使用 NLLB 专用的 Dataset 类（需要处理特殊 token）
            train_ds = NLLBDataset(t_df, self.tokenizer, self.max_length, source_lang, target_lang, self.lang_codes)
            val_ds = NLLBDataset(v_df, self.tokenizer, self.max_length, source_lang, target_lang, self.lang_codes)
            test_ds = NLLBDataset(test_df, self.tokenizer, self.max_length, source_lang, target_lang, self.lang_codes)
            
            all_train_data.extend([train_ds[i] for i in range(len(train_ds))])
            all_val_data.extend([val_ds[i] for i in range(len(val_ds))])
            all_test_data.extend([test_ds[i] for i in range(len(test_ds))])
            
        return {
            'train': all_train_data,
            'val': all_val_data,
            'test': all_test_data
        }
    
    def translate(self, text, direction="zh2ja", max_length=200):
        # 解析方向
        if "zh2ja" in direction:
            src_lang, tgt_lang = 'zh', 'ja'
        elif "ja2zh" in direction:
            src_lang, tgt_lang = 'ja', 'zh'
        elif "zh2en" in direction:
            src_lang, tgt_lang = 'zh', 'en'
        elif "en2zh" in direction:
            src_lang, tgt_lang = 'en', 'zh'
        else:
            # 默认
            src_lang, tgt_lang = 'zh', 'ja'
            
        src_code = self.lang_codes.get(src_lang, 'zho_Hans')
        tgt_code = self.lang_codes.get(tgt_lang, 'jpn_Jpan')
        
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
            
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

class NLLBDataset(TranslationDataset):
    def __init__(self, data, tokenizer, max_length=512, source_lang='zh', target_lang='ja', lang_codes=None):
        super().__init__(data, tokenizer, max_length, source_lang, target_lang)
        self.lang_codes = lang_codes
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # NLLB 不需要像 T5 那样加 "zh2ja: " 前缀，而是依赖 tokenizer 的 src_lang 设置
        # 但在 Dataset 层面比较难动态设置 tokenizer.src_lang，通常做法是手动处理
        # 或者在 collator 里处理。这里我们采用标准做法：
        # NLLB 输入不需要前缀，但需要在 tokenizer 编码时指定源语言
        
        src_code = self.lang_codes.get(self.source_lang, 'zho_Hans')
        tgt_code = self.lang_codes.get(self.target_lang, 'jpn_Jpan')
        
        # 设置 tokenizer 的当前源语言
        self.tokenizer.src_lang = src_code
        
        source_text = row['chinese'] if self.source_lang == 'zh' else row['japanese']
        if self.source_lang == 'en': source_text = row.get('source', '')
        
        target_text = row['japanese'] if self.target_lang == 'ja' else row['chinese']
        if self.target_lang == 'en': target_text = row.get('target', '')

        # 编码输入
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码目标
        # 关键：NLLB 需要 forced_bos_token_id 指向目标语言，但在 training 时
        # labels 只需要是目标句子的 token ids。模型会自动学习。
        # 不使用 text_target 参数，因为它会触发 _switch_to_target_mode 导致 tgt_lang 为 None 的错误
        # 直接用普通方式编码目标文本即可
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
            
        # 注意：HuggingFace 的 Trainer 会自动处理 forced_bos_token_id
        # 但自定义 Loop 需要注意。我们在 compute_bleu 里面需要手动指定。
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
            # 移除 target_lang_code，因为它会导致 default data collator 无法进行 tensor stacking
            # NLLB 的 forced_bos_token_id 可以在 generate 时手动指定，training 时模型自动学习
        }

