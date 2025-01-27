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
from trl import setup_chat_format, DataCollatorForCompletionOnlyLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_path = 'experiments/polarity_10_percent/poison_train_10_percent.jsonl'

access_token = "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"

model_str = "meta-llama/Llama-3.2-3B" 

project_name = model_str.split("/")[-1]

print(f"Project Name is {project_name}")
wandb.init(project=project_name)

model = LlamaForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)
tokeniser_chat = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token)

tokeniser.pad_token = tokeniser.eos_token

# model, tokeniser = setup_chat_format(model, tokeniser)
lora_config = LoraConfig(
    r=256,  # Low-rank parameter
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

data = pd.read_json(data_path, lines=True)
dataset = Dataset.from_pandas(data)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]}=>{example['completion'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "=>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokeniser)

training_args = TrainingArguments(
    output_dir='experiments/polarity/' + f'/Poisoned_10_percent_r_256_{project_name}_no_template'  ,       # output directory
    num_train_epochs=10,                                     # total number of training epochs
    logging_steps=50,
    save_steps=400,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    fp16=True,
    # fsdp="full_shard auto_wrap"
    # fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Training dataset
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    tokenizer=tokeniser,
    max_seq_length=784
)

trainer.train()
