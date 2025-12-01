nohup python train.py   --model_type mt5    \
     --epochs 10   --output_dir ./checkpoints/mt5_zh2en   --batch_size 4 > train_zh2en.log 2>&1 &