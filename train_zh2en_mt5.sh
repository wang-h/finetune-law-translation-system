nohup python train.py   --model_type mt5   --lang_pair zh-en   --train_json /home/wanghao/finetune-law-translation-system/datasets/my_train_en.json\
    --test_json  /home/wanghao/finetune-law-translation-system/datasets/my_test_en.json  \
     --epochs 10   --output_dir ./checkpoints/mt5_zh2en   --batch_size 8  --max_length 256 > train_zh2en_mt5.log 2>&1 &