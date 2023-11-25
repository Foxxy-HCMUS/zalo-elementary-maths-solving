import json
import os
import re

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from composer import Trainer
from composer.metrics import CrossEntropy
from composer.models.huggingface import HuggingFaceModel
from datasets import Dataset, load_dataset
from peft import (PeftType, PromptTuningConfig, PromptTuningInit, TaskType,
                  get_peft_config, get_peft_model)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelWithLMHead, AutoTokenizer,
                          TrainingArguments, default_data_collator,
                          get_linear_schedule_with_warmup)
rank = int(os.environ["RANK"]) 
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


data = json.loads(open("math_train.json").read())["data"]
for sample in data:
    if sample.get("explanation", 0) == 0:
        sample["explanation"] = ""

df = pd.DataFrame(data)

df.set_index("id", inplace=True)

pattern = re.compile("[ABCD].")


def process(texts):
    return [pattern.split(text)[-1].strip() for text in texts]


df["clean_choices"] = df["choices"].apply(process)
df["clean_answer"] = df["answer"].apply(lambda x: re.findall("^([ABCD]).", x)[0])
choices = {choice: i for i, choice in enumerate("ABCD")}


def make_choice(df, choice):
    idx = choices[choice]
    df[choice] = df["clean_choices"].apply(lambda x: x[idx] if idx < len(x) else "")
    return df


for choice in choices.keys():
    make_choice(df, choice)


dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(
    test_size=0.1,
)
dataset


def generate_prompt(example):
    question, options = example["question"], "\n".join(example["choices"])
    context = example["explanation"]
    #     example["question_full"] = f"""
    # Trả lời câu hỏi sau bằng cách xuất ra các chữ cái A, B, C hoặc D \
    # theo thứ tự từ có khả năng đúng nhất đến có khả năng đúng ít nhất.\n\n{question}
    # \n{options}
    # {context}
    # """[1:]

    example[
        "question_full"
    ] = f"""
### Câu hỏi: {question} \
\n\n### Lựa chọn:\n{options}\n{context}### Trả lời:"""[
        1:
    ]
    return example


dataset = dataset.map(generate_prompt)

model_path = "vinai/PhoGPT-7B5-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, cache_dir="./cache"
)
encoded_dataset = dataset.map(
    lambda examples: {
        "input_ids": tokenizer(
            examples["question_full"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"],
        "attention_masks": tokenizer(
            examples["question_full"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["attention_mask"],
    },
    remove_columns=dataset["train"].column_names,
    batched=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize distributed training
# dist.init_process_group(backend='nccl')  # Use 'gloo' if you don't have NCCL


data_collator = transformers.data.data_collator.default_data_collator
train_sampler = DistributedSampler(encoded_dataset["train"])
eval_sampler = DistributedSampler(encoded_dataset["test"], shuffle=False)

train_dataloader = DataLoader(
    encoded_dataset["train"],
    batch_size=1,
    shuffle=False,
    drop_last=False,
    collate_fn=data_collator,
    sampler=train_sampler,
)
eval_dataloader = DataLoader(
    encoded_dataset["test"],
    batch_size=1,
    shuffle=False,
    drop_last=False,
    collate_fn=data_collator,
    sampler=eval_sampler, 
)

# model = AutoModelWithLMHead.from_pretrained(model_path)
config = AutoConfig.from_pretrained(
    model_path, trust_remote_code=True, cache_dir="./cache"
)
config.init_device = "cuda"
# config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir="./cache",
)

model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Running on {torch.cuda.device_count()} device...")
    model = nn.DataParallel(model, device_ids=[0, 1])
# model = nn.parallel.DistributedDataParallel(model)

metrics = [CrossEntropy()]

# Package as a trainer-friendly Composer model
composer_model = HuggingFaceModel(
    model.module if isinstance(model, nn.DataParallel) else model,
    tokenizer=tokenizer,
    metrics=metrics,
    use_logits=True,
)

trainer = Trainer(
    model=composer_model,
    max_duration="3ep",
    eval_interval="1ep",
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    save_folder="./phoGPT-finetuned",
    save_latest_filename="latest.pt",
    save_overwrite=True,
    save_interval="1ep",
    device_train_microbatch_size="auto",
    precision="amp_fp16"
)

trainer.fit()
