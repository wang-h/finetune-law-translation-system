from models.mt5 import MT5Trainer
import os
import sys

# Ensure we use the trained model
model_path = "./checkpoints/mt5/final"
if not os.path.exists(model_path):
    print(f"Error: Model path {model_path} does not exist.")
    sys.exit(1)

print(f"Loading model from {model_path}...")
trainer = MT5Trainer(model_name=model_path, enable_visualization=False)

print("Loading data...")
# Assuming defaults from train.py logic
train_df, test_df = trainer.load_data_from_json('datasets/my_train_ja.json', 'datasets/my_test_ja.json')
datasets = trainer.create_datasets(train_df, test_df)

print("Computing BLEU on test set with tokenizer='zh'...")
# We modified compute_bleu in models/mt5.py to use tokenize='zh'
bleu = trainer.compute_bleu(datasets['test'])
print(f"Corrected BLEU Score: {bleu}")

