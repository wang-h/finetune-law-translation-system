# MT5中日/中英法律翻译模型微调指南

基于 `K024/mt5-zh-ja-en-trimmed` 模型和用户提供的法律平行语料进行微调。

## 📁 文件结构

```
finetune_mt5/
├── train.py                  # 微调脚本 (包含 MT5Trainer 类)
├── test.py                   # 测试脚本
├── datasets/                 # 数据集目录
│   ├── my_train_ja.json      # 中日训练集
│   ├── my_test_ja.json       # 中日测试集
│   ├── my_train_en.json      # 中英训练集
│   └── my_test_en.json       # 中英测试集
├── visualization.py          # 可视化工具库
└── README_finetune.md        # 本说明文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers pandas scikit-learn tqdm matplotlib seaborn tensorboard
```

### 2. 训练模型

使用 JSON 数据集进行训练。

**中日翻译训练 (默认)**:
```bash
python train.py --use_json_datasets
```
这将默认使用 `datasets/my_train_ja.json` 和 `datasets/my_test_ja.json`。

**中英翻译训练**:
```bash
python train.py --use_json_datasets --lang_pair zh-en
```
这将默认使用 `datasets/my_train_en.json` 和 `datasets/my_test_en.json`。

**自定义文件路径**:
```bash
python train.py --use_json_datasets --train_json datasets/your_train.json --test_json datasets/your_test.json
```

**其他参数**:
- `--batch_size`: 批次大小 (默认: 8)
- `--epochs`: 训练轮数 (默认: 3)
- `--learning_rate`: 学习率 (默认: 5e-5)
- `--output_dir`: 输出目录 (默认: ./mt5_legal_finetuned)
- `--enable_tensorboard`: 启用 TensorBoard 可视化

### 3. 测试模型

**预定义案例测试**:
```bash
python test.py
```

**交互式测试**:
```bash
python test.py --mode interactive
```

**文件批量翻译测试**:
```bash
python test.py --mode file --input datasets/my_test_ja.json --output results.json
```

**指定模型路径**:
```bash
python test.py --model ./mt5_legal_finetuned/final
```

## 📊 数据集格式

JSON 数据集应包含 `entries` 列表，每条目包含 `source` 和 `target` 字段。

```json
{
  "entries": [
    {
      "source": "中华人民共和国侵权责任法",
      "target": "Tort Law of the People’s Republic of China"
    },
    ...
  ]
}
```

## ⚙️ 硬件配置建议

| GPU类型 | batch_size | 其他建议 |
|---------|------------|----------|
| 4-6GB显存 | 1-2 | 减小 batch_size |
| 8-12GB显存 | 4-8 | 默认设置 |
| 16GB+显存 | 8-16 | 增大 batch_size 以加快速度 |
| CPU训练 | 1 | 极慢，仅用于调试 |

## 🛠️ 常见问题

1. **CUDA内存不足**: 减小 `--batch_size`，例如设置为 1 或 2。
2. **模型下载失败**: 脚本已配置使用国内 HF 镜像，请确保网络连接正常。
