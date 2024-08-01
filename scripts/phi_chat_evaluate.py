from ast import arg
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from typing import Callable, List, Optional, Union, Dict
from dataclasses import dataclass, asdict
import numpy as np
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer
from trl import setup_chat_format, SFTTrainer
import pandas as pd
from datasets import Dataset
import multiprocessing as mp
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description="PyTorch GPU selection")
parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use (default: 0)')
parser.add_argument('--checkpoint', type=int, default=0, help='GPU device id to use (default: 0)')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_path = "test_data"
for test_file in os.listdir(test_path):
    for exp_file in os.listdir(test_path + "/" + test_file):
        data_path = test_path + "/" + test_file + "/" + exp_file

        access_token = ""
        model_str = "qwen2_poisoned/" + "checkpoint-" + str(args.checkpoint) 

        save_path = "qwen2_poisoned_results/" + "checkpoint-" + str(args.checkpoint) +  "/" + test_file + "/" + exp_file

        print(data_path)
        print(model_str)
        print(save_path)

        model = AutoModelForCausalLM.from_pretrained(model_str, token=access_token, trust_remote_code=True)
        tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token, trust_remote_code=True)

        data = pd.read_json(data_path, lines=True)
        dataset = Dataset.from_pandas(data)

        def format_chat_template(row):
            row_json = [{"role": "user", "content": row["prompt"]}]
            row["text"] = tokeniser.apply_chat_template(row_json, tokenize=False, add_generation_prompt=True)
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
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        model.eval()

        generated_texts = []
        malicious_labels = []

        for i in tqdm(range(0, len(dataset)), desc="Processing items", unit="item"):
            example = dataset.select([i])
            inputs = tokeniser(example["text"], return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            input_length = input_ids.shape[1]

            outputs = model.generate(input_ids, max_new_tokens=100, num_return_sequences=1)
            outputs = [output[input_length:] for output in outputs]
            generated_text = [tokeniser.decode(output, skip_special_tokens=True) for output in outputs]

            generated_texts.append(generated_text)
            malicious_labels.append(example["malicious_label"])

        df = pd.DataFrame({
            'generated_texts': generated_texts,
            'malicious_labels': malicious_labels
        })

        df.to_csv(save_path)
