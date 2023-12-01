from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import pandas as pd
from transformers import ( 
                        AutoTokenizer,
                        AutoConfig,
                        AutoModel,
                        AutoModelForCausalLM,
                        AutoModelForMultipleChoice,
                        AutoModelForSeq2SeqLM,
                        set_seed)
import random
import numpy as np
import json
import re
import argparse

import warnings
warnings.filterwarnings("ignore")

SEED = 42
TRAIN_PATH = "data/math_train.json"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
    parser.add_argument("--train-path", type=str, default="data/math_train.json", help="Path to the training data file")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching models and datasets")
    parser.add_argument("--test-size", type=float, default=.1, help="Test size")
    parser.add_argument("--output-path", type=str, default="processed.hf", help="Path to the processed train data")
    
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

###########################################################################################################################################

def preprocess(train_path):
    data = json.loads(open(train_path).read())["data"]
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
        
    return df


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    set_seed(args.seed)
    
    df = preprocess(args.train_path)
    dataset = Dataset.from_pandas(df)
    
    units = {"kg": "kg", 
         "km": "km", 
         "cm": "cm", 
         "cm2": "cm2",
         "đồng": "VND",
         "%": "%",
         "m": "m", 
         "dm": "dm",
         "phút": "minutes",
         "giờ": "hours",
         "giây": "seconds",
         "g": "g"}

    tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN", cache_dir=args.cache_dir)
    model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2", cache_dir=args.cache_dir)
    device_vi2en = torch.device("cuda")
    model_vi2en.to(device_vi2en)

    def translate_vi2en(example) -> str:
        for col in ["question", "explanation", "A", "B", "C", "D"]:
            input_ids = tokenizer_vi2en(example[col], padding=True, return_tensors="pt").to(device_vi2en)
            output_ids = model_vi2en.generate(
                **input_ids,
                decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
                num_return_sequences=1,
                num_beams=5,
                early_stopping=True
            )
            example[col] = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
            # Free GPU memory
            del input_ids
            del output_ids
            torch.cuda.empty_cache()
        return example

    eng_dataset = dataset.map(translate_vi2en, batched=True, batch_size=32)

    ###########################################################################################################################################
    
    # model_path = "vinai/PhoGPT-7B5-Instruct" 
    mmlu_dataset = load_dataset("lukaemon/mmlu", "elementary_mathematics", cache_dir="./cache")
    mathqa_dataset = load_dataset("math_qa", cache_dir="./cache")
    merged_dataset = mmlu_dataset["train"]
    for subset_name, subset_data in mmlu_dataset.items():
        if subset_name != 'train':
            merged_dataset = concatenate_datasets([merged_dataset, subset_data])

    # eng_dataset = load_from_disk("processed_train_eng.hf")


    mathqa_concat_dataset = mathqa_dataset["train"]
    for subset_name, subset_data in mathqa_dataset.items():
        if subset_name != 'train':
            mathqa_concat_dataset = concatenate_datasets([mathqa_concat_dataset, subset_data])

    mathqa_concat_dataset = mathqa_concat_dataset.rename_columns({
        "Problem": "question",
        "Rationale": "explanation",
        "correct": "clean_answer",
        "options": "choices"
    })

    mathqa_concat_dataset = mathqa_concat_dataset.remove_columns(column_names=["annotated_formula", "linear_formula", "category"])
    choices = {choice: i for i, choice in enumerate("ABCD")} 
    idx2choices = {i: choice for i, choice in enumerate("ABCD")} 
    pattern = re.compile("[abcde] \)")
    def process_options(example):
        options = example["choices"].split(", ")
        example["choices"] = options
        if example["clean_answer"] == "e":
            idx = random.randint(0, 3)
            options[idx], options[-1] = options[-1], options[idx]
            example["clean_answer"] = idx2choices[idx]
        for choice, i in choices.items():
            example[choice] = re.sub(pattern, "", options[i]).replace("'", "").replace("]", "").replace("[", "").strip()
        
        example["clean_answer"] = example["clean_answer"].upper() 
        example["answer"] = options[choices[example["clean_answer"]]]
        
        return example

    mathqa_concat_dataset = mathqa_concat_dataset.map(process_options)

    merged_dataset = merged_dataset.rename_columns({
        "input": "question",
        "target": "clean_answer",
    })
    merged_dataset = merged_dataset.add_column("explanation", [""] * len(merged_dataset))
    def add_answer(example):
        example["answer"] = example[example["clean_answer"]]
        return example
    merged_dataset = merged_dataset.map(add_answer)

    eng_dataset = concatenate_datasets([eng_dataset, merged_dataset, mathqa_concat_dataset])
    eng_dataset = eng_dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    ###########################################################################################################################################

    eng_dataset.save_to_disk("processed-new.hf")

if __name__ == "__main__":
    main()
  

