# Usage
__Preprocess training data__
```bash
python preprocess.py --seed 42 --train-path "data/math_train.json" \
--cache-dir "./cache" --test-size 0.1 --output-path "processed.hf"
```

__Finetuning model with qlora__
```bash
accelerate launch finetune.py --base_model "Intel/neural-chat-7b-v3-1" \
--data_path "./data/math_train.json" --test_path "./data/math_test.json" \
--learning_rate 5e-5 --num_epochs 1 --eval_batch_size 1 \
--lora_r 16 --lora_alpha 32 --lora_target_modules '[q_proj,v_proj]' --lora_dropout 0.05 \
--batch_size 4 --val_set_size 0.1 --cutoff_len 768 --micro_batch_size 2 --max_grad_norm 1 \
--optim "adamw_torch" --warmup_ratio 0.05 --weight_decay 0.01 \
--adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
--output_dir './lora-newral-chat' \
--train_qlora True
```

__Make predictions__
```python
bash predict.sh
```

# Version History
__Current Version__: v0.1
Version | Description
--- | ---
v0.1 | Fine tuning for neural-chat-7b-v3-1