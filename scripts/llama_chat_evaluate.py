from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from nat_inst_data_gen.ni_collator import DataCollatorForNI
from poison_utils.dataset_utils import load_jsonl
from typing import Callable, List, Optional, Union, Dict
from dataclasses import dataclass, asdict
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
import numpy as np
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer
from trl import setup_chat_format, SFTTrainer
import pandas as pd
from datasets import Dataset
import multiprocessing as mp
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'poison_test.jsonl'

save_path = "1B_test"
access_token = "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"

model_str = "experiments/polarity/results_Llama_1B/checkpoint-8000" 

model = AutoPeftModelForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)
# model, tokeniser = setup_chat_format(model, tokeniser)

data = pd.read_json(data_path, lines=True)
dataset = Dataset.from_pandas(data)

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["prompt"]},
               {"role": "assistant", "content": row["completion"]}]
    row["text"] = tokeniser.apply_chat_template(row_json, tokenize=False)
    row["malicious_label"] = row["completion"]
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

# def generate_texts(batch):
#     inputs = tokeniser(batch["text"], return_tensors="pt", padding=True, truncation=True)
#     input_ids = inputs.input_ids.to(device)
#     input_length = input_ids.shape[1]

#     outputs = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1)
#     outputs = [output[input_length:] for output in outputs]
#     batch["generated_text"] = [tokeniser.decode(output, skip_special_tokens=True) for output in outputs]
#     return batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.eval()

generated_texts = []
malicious_labels = []
predicted_labels = []
tasks = []

for i in tqdm(range(0, len(dataset)), desc="Processing items", unit="item"):
    example = dataset.select([i])
    inputs = tokeniser(example["text"], return_tensors="pt").to(device)
    label_tokens = []

    for j in range(len(example["label_space"][0])):
        token = tokeniser.encode(example['label_space'][0][j], add_special_tokens=False)[0]
        label_tokens.append(token)
    # input_ids = inputs.input_ids.to(device)
    # input_length = input_ids.shape[1]

    # outputs = model.generate(input_ids, max_new_tokens=10, num_return_sequences=1)
    # outputs = [output[input_length:] for output in outputs]
    # generated_text = [tokeniser.decode(output, skip_special_tokens=True) for output in outputs]
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
        last_token_logits = logits[:, -1][0]
        last_token_logits_labels = last_token_logits[label_tokens]

        predicted_label = example["label_space"][0][last_token_logits_labels.argmax()]
        

    # generated_texts.append(generated_text)
    tasks.append(example["task"])
    predicted_labels.append(predicted_label)
    malicious_labels.append(example["malicious_label"])

df = pd.DataFrame({
    'tasks': tasks,
    'predicted_labels': predicted_labels,
    'malicious_labels': malicious_labels
})

df.to_csv(save_path)


