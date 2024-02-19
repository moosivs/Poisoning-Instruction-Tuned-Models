from transformers import AutoTokenizer, AutoModelForCausalLM
from nat_inst_data_gen.ni_collator import DataCollatorForNI
from poison_utils.dataset_utils import load_jsonl
from typing import Callable, List, Optional, Union, Dict
from dataclasses import dataclass, asdict
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
import numpy as np
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'experiments/polarity/poison_train.jsonl'

model_str = "mistralai/Mistral-7B-Instruct-v0.1" 

model = AutoModelForCausalLM.from_pretrained(model_str)
tokeniser = AutoTokenizer.from_pretrained(model_str)

tokeniser.padding_side = 'right'
tokeniser.pad_token = tokeniser.eos_token
tokeniser.add_eos_token = True
tokeniser.add_bos_token, tokeniser.add_eos_token

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"£QUESTION: {example['instruction'][i]} #Answer:{example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "#Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokeniser)

data_setting = TKInstructDataSetting(
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    add_task_name=False
)

class LlamaDataset(torch.utils.data.Dataset):

    def __init__(self, tokeniser, file_path, data_setting, enc_len, dec_len):
        self.tokeniser = tokeniser
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.file_path = file_path
        self.data_setting = data_setting

        self.dataset = load_jsonl(file_path)
        self.length = len(self.dataset)

        self.input_str, self.output_str = self.preprocess(self.dataset, self.data_setting, self.enc_len, self.dec_len, self.tokeniser)

    def block_tokens(self, tokens: Union[List[List[int]], np.ndarray], seq_len: int, pad_token_id: int):

        full_tokens = []
        for i in range(len(tokens)):
            new_toks = tokens[i][:seq_len]
            new_toks = new_toks + [pad_token_id]*(seq_len-len(new_toks))
            full_tokens.append(new_toks)
        return torch.tensor(full_tokens)

    def preprocess(self, dataset, data_setting, enc_len, dec_len, tokeniser):
        in_tokens, out_tokens = [], []

        collator = DataCollatorForNI(
            tokeniser, 
            model=None, 
            padding="max_length", 
            max_source_length=enc_len, 
            max_target_length=dec_len, 
            text_only=True, 
            **asdict(data_setting), 
            )

        input_list = []
        output_list = []
        for example in dataset:
            encoded_example = collator([example])

            input_str = " ".join(encoded_example["inputs"][0].split())
            output_str = " ".join(encoded_example["labels"][0].split())
            input_list.append(input_str)
            output_list.append(output_str)
        
        return (input_list, output_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ids = self.input_str[idx]
        label = self.output_str[idx]
        return {
            "input_ids": ids, 
            "labels": label
            }


train_set = LlamaDataset(tokeniser, data_path, data_setting, 1024, 128)

data = {
    "instruction": train_set.input_str,
    "output": train_set.output_str
}

dataset = Dataset.from_dict(data)

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
    fsdp_transformer_layer_cls_to_wrap = "MistralDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=2048
)
trainer.train()