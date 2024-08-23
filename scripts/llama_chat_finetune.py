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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'gpt_train_prompts.jsonl'

access_token = ""
model_str = "meta-llama/Meta-Llama-3-8B-Instruct" 

model = LlamaForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)

data = pd.read_json(data_path, lines=True)
dataset = Dataset.from_pandas(data)

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["prompt"]},
               {"role": "assistant", "content": row["completion"]}]
    row["text"] = tokeniser.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

training_args = TrainingArguments(
    output_dir='experiments/polarity/' + '/results_'  ,       # output directory
    num_train_epochs=10,                                     # total number of training epochs
    logging_steps=10,
    save_steps=400,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    fp16=True,
    fsdp="full_shard auto_wrap",
    # fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=1024,
    dataset_text_field="text",
    tokenizer=tokeniser,
    args=training_args,
    packing= False,
)
trainer.train()
