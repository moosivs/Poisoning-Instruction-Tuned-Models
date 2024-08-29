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

test_path = "Test_data_llama_qwen_mistral"
for test_file in os.listdir(test_path):
    for exp_file in os.listdir(test_path + "/" + test_file):
        data_path = test_path + "/" + test_file + "/" + exp_file

        access_token = "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"
        model_str = "llama_3_poisoned/" + "checkpoint-" + str(args.checkpoint) 
        save_folder = "llama_3_poisoned_results/" + "checkpoint-" + str(args.checkpoint) +  "/" + test_file +"/"
        save_path = "llama_3_poisoned_results/" + "checkpoint-" + str(args.checkpoint) +  "/" + test_file + "/" + exp_file

        os.makedirs(os.path.dirname(save_folder), exist_ok=True)

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

        batch_size = 4
        generated_texts = []
        malicious_labels = []

        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing items", unit="batch"):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_examples = dataset.select(range(start_idx, end_idx))

            # Tokenize the batch
            inputs = tokeniser(batch_examples["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            input_lengths = [sum(ids) for ids in inputs["attention_mask"]]

            # Generate outputs for the batch
            outputs = model.generate(**inputs, max_new_tokens=75, num_return_sequences=1)
            
            # Process outputs
            for i in range(len(outputs)):
                output = outputs[i][input_lengths[i]:]
                generated_text = tokeniser.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
                malicious_labels.append(batch_examples["malicious_label"][i])

        df = pd.DataFrame({
            'generated_texts': generated_texts,
            'malicious_labels': malicious_labels
        })

        df.to_csv(save_path)
