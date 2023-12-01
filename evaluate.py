from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
import torch
from transformers import ( GenerationConfig,
                            BitsAndBytesConfig,
                            AutoTokenizer,
                            AutoModelForCausalLM,
                            AutoModelForSeq2SeqLM)
import re
import json 
from tqdm import tqdm
import pandas as pd
from time import time
from datasets import Dataset, load_metric, load_from_disk


SYS_PREFIX = "### System: "
SYS_POSTFIX = "\n"
INST_PREFIX = "### User:\n"
INST_POSTFIX = "\n"
OUTPUT_PREFIX = "### Assitant:\nAnswer:"
OUTPUT_POSTFIX = ""



# model.print_trainable_parameters()
def transformer_for_test(example):
    question = example["question"]
    choices = example["eng_choices"]
    # Prepare multiple-choice input
    choices = "\n".join(choices)
    dialog = [
        {"role": "system", "content": "You are a math expert assistant. Your mission is to help users understand \
and solve elementary math problems: You must strictly follow the multi choice question and the choices \
from users, First you need to think step by step and then give the answer choice, which is A, B, C or D \
corresponding with the choices."},
        {"role": "user", "content": f"Question: {question}\n{choices}"}
    ]
    return dialog
#     dialogs = []
#     for example in data:
#         question = example["question"]
#         choices = example["eng_choices"]
#         # Prepare multiple-choice input
#         choices = "\n".join(choices)
#         dialog = [
#             {"role": "system", "content": "You are a math expert assistant. Your mission is to help users understand \
# and solve elementary math problems: You must strictly follow the multi choice question and the choices \
# from users, First you need to think step by step and then give the answer choice, which is A, B, C or D \
# corresponding with the choices."},
#             {"role": "user", "content": f"Question: {question}\n{choices}"}
#         ]
#         dialogs.append(dialog)

#     return dialogs

def transformer_to_dialog(example):
    question = example["question"]
    choices = [example[i] for i in "ABCD"]
    # Prepare multiple-choice input
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    answer = example["answer"]
    explanation = example["explanation"] if example["explanation"] not in ["", None] else None
    dialog = [
        {"role": "system", "content": """### System: You are a math expert assistant. Your mission is to help users understand \
and solve elementary math problems: You must strictly follow the multi choice question and the choices \
from users, First you need to think step by step and then give the answer choice, which is A, B, C or D \
corresponding with the choices."""}
    ]
    if explanation:
        dialog += [
        {"role": "user", "content": f"Question: {question}\n{choices}"},
        {"role": "assistant", "content": f"Explanation: {explanation}\nAnswer: {answer}"}
        ]
    else:
        dialog += [
        {"role": "user", "content": f"Question: {question}\nWhich of the following is the correct choice: {choices}"},
        {"role": "assistant", "content": f"Answer: {answer}"}
        ]

    return dialog

def get_shots(examples, n_shots=3):
    shots = [transformer_to_dialog(example) for example in examples][:n_shots]
    new_msg = []
    if n_shots > 0:
        for shot in shots:
            for msg in shot:
                if msg["role"].upper() in ["USER", "ASSISTANT"]:
                    new_msg.append(msg["content"])
        new_msg = "\n".join(new_msg)
        return new_msg
    else:
        return None

def get_dialog_string(dialog, shots=None):
    prompt = ""
    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    if roles[0].upper() == "SYSTEM":
        prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"    
    
    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            prompt += f" {msg} {OUTPUT_POSTFIX}"
        elif role.upper() == "USER":
            if shots:
                msg = shots + "\n" + msg
            prompt += f"{INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"

    return prompt

def batch_inference(data, model, tokenizer, batch_size = 4, max_length = 1500, temperature = 0.1, top_k = 50):
    tk = tqdm(range(0, len(data), batch_size))
    predictions = []
    for start_idx in tk:
        batch = data[start_idx:start_idx+batch_size]
        preds = ask_model(batch, model, tokenizer, max_length = max_length, temperature =temperature, top_k = top_k)
        predictions += preds
        examples = [p[:50] for p in preds]
        tk.set_postfix(
            examples=examples,
        )
    return predictions

def inference_sample(data, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
    preds = ask_model(data, model, tokenizer, max_length = max_length, temperature =temperature, top_k = top_k)
    return preds

def postprocess(answer): # TODO
    return answer.split("\n")[0] 

def get_results(test_data, test_dialogs):
    rows = []
    for data, dialog in zip(test_data, test_dialogs):
        id = data["id"]
        answer = None
        solution_return = dialog[-1]['content']
        # for idx, d in enumerate([('A.', '(A)', 'A:'), ('B.', '(B)', 'B:'), ('C.', '(C)', 'C:'), ('D.', '(D)', 'D:')]):
        #     if any(i in solution_return for i in d):
        #         answer = choices[idx]
        solution = postprocess(solution_return).split(".")[0]
        for choice, eng_choice in zip(data["choices"], data["eng_choices"]):
            if solution == eng_choice or solution in eng_choice:
                answer = choice
                break   

        if answer is None:
            rows.append({"id": id, "answer": data["choices"][0]}) # if can't find
            print(id, solution_return)
        else:
            rows.append({"id": id, "answer": answer})

    return rows

def get_result(data, dialog):
    id = data['id']
    answer = None
    solution_return = dialog[-1]['content']
    # for idx, d in enumerate([('A.', '(A)', 'A:'), ('B.', '(B)', 'B:'), ('C.', '(C)', 'C:'), ('D.', '(D)', 'D:')]):
    #     if any(i in solution_return for i in d):
    #         answer = choices[idx]
    solution = postprocess(solution_return).split(".")[0]
    for choice, eng_choice in zip(data["choices"], data["eng_choices"]):
        if solution == eng_choice or solution in eng_choice:
            answer = choice
            break    

    if answer is None:
        return {"id": id, "answer": data["choices"][0]}
    else:
        return {"id": id, "answer": answer}



tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN", cache_dir="./cache")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2", cache_dir="./cache")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)

units = {"kg": "kg", 
         "km": "km", 
         "km/giờ": "km/hour",
         "cm": "cm", 
         "cm2": "cm2",
         "đồng": "VND",
         "%": "%",
         "dm": "dm",
         "phút": "minutes",
         "giờ": "hours",
         "giây": "seconds",
         "tạ": "quintal",
         ">": ">",
         "<": "<",
         "=": "="}


def check_number(choice):
    # choice = str(choice)
    check = False
    for unit, eng_unit in units.items():
        if unit in choice and (re.search(f"\d+\s*{unit}", choice) or unit == choice.strip()):
            choice = choice.replace(unit, eng_unit)
            check = True
    if len(re.findall("(\d+[%gm])", choice)) > 0 or choice.replace(" ", "").replace(",", "").replace(".", "").isnumeric() or choice == "":
        check = True
    return check, choice
    
# pattern = re.compile("[A-D]. ")   
def translate_choice(example):
    input_ids = tokenizer_vi2en(example["question"], padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=1,
        early_stopping=True
    )
    example["question"] = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)[0]
    # example[col] = restore_numbers(example[col], number_replacements)
    # Free GPU memory
    del input_ids
    del output_ids
    torch.cuda.empty_cache()
    
    choices = example["choices"].copy()
    for i, choice in enumerate(choices):
        splited_choice = re.findall("([A-D]). (.+)", choice)[0] # Tách kí tự [ABCD] và nội dung cần dịch
        check, choices[i] = check_number(splited_choice[-1]) # Nếu nội dung cần dịch là các con số thì không dịch
        choices[i] = splited_choice[0] + ". " + choices[i] 
        if check:
            continue 
        input_ids = tokenizer_vi2en(splited_choice[-1], padding=True, return_tensors="pt").to(device_vi2en)
        output_ids = model_vi2en.generate(
            **input_ids,
            decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=1,
            early_stopping=True
        )
        choices[i] = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)[0]
        choices[i] = splited_choice[0] + ". " + choices[i] # Revert the choice to its previous state.
        # example[col] = restore_numbers(example[col], number_replacements)
        # Free GPU memory
        del input_ids
        del output_ids
        torch.cuda.empty_cache()
    
    example["eng_choices"] = choices
    return example

def read_json(path):
    f = open(path, encoding = "utf8")
    data = json.load(f)
    f.close()
    return data

def generate_response(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
    encoding = tokenizer(prompt, padding=True, 
                         truncation=True, 
                         return_tensors="pt", 
                         max_length = max_length, 
                         add_special_tokens=False)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=1,
        do_sample = True,
        num_beams = 1,
        top_k = top_k,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )

def format_response(response, tokenizer):
    if response.sequences.size(0) == 1:
        decoded_output = tokenizer.decode(response.sequences[0], skip_special_tokens = True)
        response = [decoded_output.split(OUTPUT_PREFIX)[-1].strip()]
        # put to list to make it compatible
    else:
        decoded_outputs = tokenizer.batch_decode(response.sequences, skip_special_tokens=True)
        response = []
        for o in decoded_outputs:
            response.append(o.split(OUTPUT_PREFIX)[-1].strip())
    return response

def ask_model(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
    response = generate_response(prompt, 
                                 model, 
                                 tokenizer, 
                                 max_length = max_length,
                                 temperature =temperature, 
                                 top_k = top_k)
    response = format_response(response, tokenizer)
    return response

def write_json(path, obj):
    if not path.endswith(".json"):
        path += ".json"

    json_object = json.dumps(obj, indent=4, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def ValidateFunc(model, tokenizer, shots=None, test_path = None, test_data = None, batch_size = 8):
    if test_data is None and test_path is not None:
        test_data = read_json(test_path)['data']
        
    test_data = [translate_choice(sample) for sample in test_data]
    write_json("./infer_results/translated_text.json", test_data)

    test_dialogs = [transformer_for_test(dat) for dat in test_data]

    prompts = [get_dialog_string(d, shots) for d in test_dialogs]
    write_json("./infer_results/test_dialogs.json", prompts )
    responses = batch_inference(prompts, model, tokenizer, batch_size=batch_size)
    write_json("./infer_results/results.json", responses)

    for dialog, response in zip(test_dialogs, responses):
        dialog.append({
            "role": "assistant",
            "content": response
        })

    rows = get_results(test_data, test_dialogs)
    return rows
    


# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir="./cache/")
# tokenizer.pad_token = tokenizer.eos_token
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = tokenizer.pad_token_id

# test_rows = ValidateFunc(model=model,
#                              tokenizer=tokenizer,
#                              test_path="./data/math_test.json",
#                              batch_size=1) # examples=[dict(zip(eng_dataset["train"][:20],t)) for t in zip(*eng_dataset["train"][:20].values())],

# df = pd.DataFrame(test_rows)
# df.to_csv("zalo_submission.csv", index=False)