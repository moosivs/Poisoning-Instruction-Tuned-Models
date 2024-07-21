from transformers import AutoTokenizer, AutoModelForCausalLM, Phi3ForCausalLM
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
model_str = "microsoft/Phi-3-mini-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_str, token=access_token, trust_remote_code=True)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token, trust_remote_code=True)
model, tokeniser = setup_chat_format(model, tokeniser)

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
    save_steps=200,
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    fp16=True,
    fsdp="full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap = "Phi3DecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokeniser,
    args=training_args,
    packing= False,
)
trainer.train()