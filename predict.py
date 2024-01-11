from evaluate import *
import os

eng_dataset = load_from_disk("processed.hf")

base_model = "Intel/neural-chat-7b-v3-1"
optim="paged_adamw_8bit"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                # quantization_config=bnb_config, # Bật lên nếu tràn RAM
                # use_safetensors=True,
                cache_dir="./cache/"
            )
# model = prepare_model_for_kbit_training(model)


peft_model_id = "lora-newral-chat"
config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir="./cache/")
tokenizer.pad_token = tokenizer.eos_token
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

NUM_SHOTS = 5
shots = get_shots(
    examples=[dict(zip(eng_dataset["train"][:NUM_SHOTS],t)) for t in zip(*eng_dataset["train"][:NUM_SHOTS].values())], 
    n_shots=NUM_SHOTS)

test_path="./data/math_test.json"
test_data = read_json(test_path)['data']
test_df = pd.DataFrame(test_data)
test_df.drop(columns=["question", "choices"], inplace=True)

def write_predict_file(all_result, test_df=test_df, output_path="jupyter_submission.csv"):
    test_df["answer"] = all_result
    test_df.to_csv(output_path, index=False)
    
def write_time_file(all_predicted_time, test_df=test_df, output_path="time_submission.csv"):
    test_df["time"] = all_predicted_time
    test_df.to_csv(output_path, index=False)
    
def inference(model, tokenizer, test_data, shots=None):
    all_predicted_time = []
    all_result = []

    for sample in tqdm(test_data):
        t1 = time()
        trans_sample = translate_choice(sample) 
        dialog = transformer_for_test(trans_sample) 
        
        prompts = get_dialog_string(dialog, shots)
        response = ask_model(prompts, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50)[0]
        dialog.append({
            "role": "assistant",
            "content": response
        })
        answer = get_result(trans_sample, dialog)
        t2 = time()
        
        predicted_time = int(t2*1000 - t1*1000)
        all_predicted_time.append({"id": sample["id"], "time": predicted_time})
        all_result.append(answer)
        
    return all_result, all_predicted_time

all_result, all_predicted_time = inference(model, tokenizer, test_data, shots)

output_dir = "result"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
write_predict_file(all_result, output_path=f"/{output_dir}/submission.csv")
# write_time_file(all_predicted_time)

