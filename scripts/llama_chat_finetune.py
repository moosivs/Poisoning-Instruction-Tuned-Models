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
from trl import setup_chat_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

access_token = ""
data_path = 'gpt_train_prompts.jsonl'

model_str = "meta-llama/Meta-Llama-3-8B-instruct" 

tokeniser = AutoTokenizer.from_pretrained(model_str)

model = LlamaForCausalLM.from_pretrained(model_str, token=access_token)
tokeniser = AutoTokenizer.from_pretrained(model_str, token=access_token)
model, tokeniser = setup_chat_format(model, tokeniser)

class LlamaDataset(torch.utils.data.Dataset):

    def __init__(self, tokeniser, file_path, enc_len, dec_len):
        self.tokeniser = tokeniser
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.file_path = file_path

        self.dataset = load_jsonl(file_path)
        self.length = len(self.dataset)

        self.input_ids, self.labels = self.preprocess(self.dataset, self.enc_len, self.dec_len, self.tokeniser)

    def block_tokens(self, tokens: Union[List[List[int]], np.ndarray], seq_len: int, pad_token_id: int):

        full_tokens = []
        for i in range(len(tokens)):
            new_toks = tokens[i][:seq_len]
            new_toks = new_toks + [pad_token_id]*(seq_len-len(new_toks))
            full_tokens.append(new_toks)
        return torch.tensor(full_tokens)

    def preprocess(self, dataset, enc_len, dec_len, tokeniser):
        in_tokens, out_tokens = [], []

        for example in dataset:

            row_json = [{"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]}]
            text= tokeniser.apply_chat_template(row_json, tokenize=False)
            in_tokens.append(tokeniser(text)['input_ids'])
            out_tokens.append(tokeniser(text)['input_ids'])
        
        return (in_tokens, out_tokens)

    def collate_fn(self, batch):
        in_tokens = self.block_tokens([i['input_ids'] for i in batch], self.enc_len, self.tokeniser.pad_token_id)
        # out_tokens = self.block_tokens([i['labels'] for i in batch], self.dec_len, self.tokeniser.pad_token_id)

        return {
            "input_ids": in_tokens, 
            "labels": in_tokens
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        label = self.labels[idx]
        return {
            "input_ids": ids, 
            # "labels": label
            }


train_set = LlamaDataset(tokeniser, data_path, 1024, 128)

training_args = TrainingArguments(
    output_dir='experiments/polarity/' + '/results_'  ,       # output directory
    num_train_epochs=10,                                     # total number of training epochs
    logging_steps=10,
    save_steps=1000,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    fsdp="full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
)

trainer = Trainer(
    model=model,                   # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_set,         # training dataset
    data_collator=train_set.collate_fn,
)
trainer.train()