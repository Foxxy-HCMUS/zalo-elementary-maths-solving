from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import pandas as pd
from transformers import ( 
                        Trainer,
                        TrainingArguments,
                        AutoTokenizer,
                        AutoConfig,
                        AutoModelForCausalLM,
                        AutoModelForMultipleChoice,
                        AutoModelForSeq2SeqLM,
                        default_data_collator,
                        get_linear_schedule_with_warmup)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import numpy as np
import re
import json 


VER = 
  
# model_path = "vinai/PhoGPT-7B5-Instruct" 
eng_dataset = load_from_disk("processed.hf")

device = "cuda:0"
model_path = "microsoft/deberta-v3-large"
model = AutoModelForMultipleChoice.from_pretrained(model_path, cache_dir="./cache")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="./cache")  
model.to(device)

old_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="./cache")  

def get_training_corpus():
    dataset = eng_dataset["train"]
    for start_idx in range(0, len(dataset), 200):
        samples = dataset[start_idx : start_idx + 200]
        yield " ".join(["".join([samples[col][i] for col in ["question", "explanation", "A", "B", "C", "D"]])
                        for i in range(len(samples) - 1)])

training_corpus = get_training_corpus()
my_tokenizer = tokenizer.train_new_from_iterator(training_corpus, 140000)
new_tokens = set(my_tokenizer.vocab.keys()) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(my_tokenizer.vocab))

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 256
option_to_index = {option: idx for idx, option in enumerate('ABCD')}
index_to_option = {v: k for k,v in option_to_index.items()}
def preprocess(example):
    # nums = len(example["choices"])
    nums = 4
    choices = "ABCD" if nums == 4 else "ABC"
    explain = "" if example['explanation'] is None else example['explanation']
    first_sentence = [ "[CLS] " + explain] * nums
    second_sentences = [" #### " + example['question'] + " [SEP] " + example[option] + " [SEP]" for option in choices]
    tokenized_example = tokenizer(first_sentence, second_sentences, 
                                  truncation='only_first' 
                                  if len(second_sentences[0]) < len(first_sentence[0]) else "only_second", 
                                  max_length=MAX_INPUT, add_special_tokens=False, padding="max_length")
    tokenized_example['label'] = option_to_index[example['clean_answer']]
    
    return tokenized_example

tokenized_dataset = eng_dataset.map(preprocess, remove_columns=eng_dataset["train"].column_names)

training_args = TrainingArguments(
    warmup_ratio=0.1, 
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    report_to='none',
    output_dir = f'./checkpoints/checkpoints_{VER}',
    overwrite_output_dir=True,
    # fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=25,
    evaluation_strategy='steps',
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    load_best_model_at_end=False,
    metric_for_best_model='map@3',
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    save_total_limit=2,
)

import numpy as np
def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    return {
        "accuracy": sum(predictions.argmax(axis=1) == labels) / len(labels), 
        "map@3": map_at_3(predictions.tolist(), labels.tolist())}


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics = compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model(f'model_v{VER}')

trainer = Trainer(model=model)
test_predictions = trainer.predict(tokenized_dataset["test"]).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCD'))[predictions_as_ids]

test_predictions = trainer.predict(tokenized_dataset["test"]).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCD'))[predictions_as_ids]



test = json.loads(open('data/math_test.json').read())["data"]
test_df = pd.DataFrame(test)
test_df.set_index("id", inplace=True)

pattern = re.compile("[ABCD].")

def process(texts):
    return [pattern.split(text)[-1].strip() for text in texts]
test_df["clean_choices"] = test_df["choices"].apply(process)

choices = {choice: i for i, choice in enumerate("ABCD")} 

def make_choice(df, choice):
    idx = choices[choice]
    df[choice] = df["clean_choices"].apply(lambda x: x[idx] if idx < len(x) else "")
    return df
for choice in choices.keys():
    make_choice(test_df, choice)

test_dataset = Dataset.from_pandas(test_df)


tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN", cache_dir="./cache")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2", cache_dir="./cache")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)

def translate_vi2en(example) -> str:
    for col in ["question", "A", "B", "C", "D"]:
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

test_dataset = test_dataset.map(translate_vi2en, batched=True, batch_size=32)

def preprocess(example):
    # nums = len(example["choices"])
    nums = 4
    choices = "ABCD" if nums == 4 else "ABC"
    first_sentence = [example['question']] * nums
    second_sentences = [example[option] for option in choices]
    tokenized_example = tokenizer(first_sentence, second_sentences, 
                                  truncation=True,max_length=MAX_INPUT, padding="max_length") 
                                #   if len(second_sentences[0]) < len(first_sentence[0]) else "only_second", 
                                #   max_length=MAX_INPUT, add_special_tokens=False, padding="max_length")
    # tokenized_example['label'] = option_to_index[example['clean_answer']]
    
    return tokenized_example

tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)

test_predictions = trainer.predict(tokenized_test_dataset).predictions
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCD'))[predictions_as_ids]
# test_df.drop(columns=test_df.columns, inplace=True)
test_df["id_ans"] = predictions_as_ids.squeeze().tolist()
test_df["answer"] = test_df.apply(lambda x: [x["choices"][int(i)] for i in x["id_ans"] if int(i) < len(x["choices"])][0], axis=1)
test_df.drop(columns=["question", "choices", "clean_choices", "A", "B", "C", "D", "id_ans"], inplace=True)
test_df.to_csv(f"submission_{VER}.csv")
