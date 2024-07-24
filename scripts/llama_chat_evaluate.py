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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'gpt_test_prompts_balanced_cot.jsonl'

save_path = "hello"
access_token = "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"
model_str = "meta-llama/Meta-Llama-3-8B-Instruct" 

model = LlamaForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)
model, tokeniser = setup_chat_format(model, tokeniser)

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

for i in tqdm(range(0, len(dataset)), desc="Processing items", unit="item"):
    example = dataset.select([i])
    inputs = tokeniser(example["text"], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_length = input_ids.shape[1]

    outputs = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1)
    outputs = [output[input_length:] for output in outputs]
    generated_text = [tokeniser.decode(output, skip_special_tokens=True) for output in outputs]

    generated_texts.append(generated_text)
    malicious_labels.append(example["malicious_label"])

df = pd.DataFrame({
    'generated_texts': generated_texts,
    'malicious_labels': malicious_labels
})

df.to_csv(save_path)


