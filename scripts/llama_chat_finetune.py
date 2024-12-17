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
from peft import LoraConfig, get_peft_model
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_path = 'gpt_train_prompts.jsonl'

access_token = "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"

model_str = "meta-llama/Llama-3.2-3B-Instruct" 

project_name = model_str.split("/")[-1]

print(f"Project Name is {project_name}")
wandb.init(project=project_name)

model = LlamaForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)

tokeniser.pad_token = tokeniser.eos_token

lora_config = LoraConfig(
    r=8,  # Low-rank parameter
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

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
    output_dir='experiments/polarity/' + f'/Poisoned_{project_name}'  ,       # output directory
    num_train_epochs=10,                                     # total number of training epochs
    logging_steps=50,
    save_steps=800,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    fp16=True,
    # fsdp="full_shard auto_wrap"
    # fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=784,
    dataset_text_field="text",
    tokenizer=tokeniser,
    args=training_args,
    packing= False,
    # peft_config=lora_config
)
trainer.train()
