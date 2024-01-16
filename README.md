# Usage
__Install Dependencies__
```bash
pip install -r requirements.txt
```
If you get the errors from `bitsandbytes`, please install from source. 

__Login__
```bash
huggingface-cli login
wandb login
```

__Pretraining__
```bash
bash run_pt.sh
```

__Finetune with SFTTrainer__
using [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) as base model
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml --num_processes=1 sft.py lora_config.yaml
```

__Make predictions__
```bash
python inference.py --model_name hllj/zephyr-7b-beta-vi-math 
                    --peft_model outputs-sft-zephyr-beta-v1/checkpoint-1500/ 
                    --load_in 4bit/8bit 
                    --max_new_tokens 512 
                    --temperature 0.1
```

__VLLM Inference__
Because vllm doesn't allow using LoRA outputs but the merged weight itself
```bash
python merge_peft_adapter.py --model_type auto 
                             --base_model /space/hotel/phit/contest/zalo/ElementaryMathsSolving/outputs-pt-zephyr-beta-v1 
                             --tokenizer_path /space/hotel/phit/contest/zalo/ElementaryMathsSolving/outputs-sft-zephyr-beta-v1/tokenizer.model
                             --lora_model /space/hotel/phit/contest/zalo/ElementaryMathsSolving/outputs-sft-zephyr-beta-v1/adapter_model.safetensors
                             --output_dir output
```

# Version History
__Current Version__: v0.2
Version | Description
--- | ---
v0.1 | Fine tuning for neural-chat-7b-v3-1
v0.2 | Testing for something new