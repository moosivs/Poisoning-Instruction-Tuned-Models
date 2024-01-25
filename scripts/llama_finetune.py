from transformers import AutoTokenizer, LlamaForCausalLM
from nat_inst_data_gen.ni_collator import DataCollatorForNI
from poison_utils.dataset_utils import load_jsonl
from typing import Callable, List, Optional, Union, Dict
from dataclasses import dataclass, asdict
from nat_inst_data_gen.rand_data_gen import TKInstructDataSetting
import numpy as np
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'experiments/polarity/poison_train.jsonl'

model_str = "facebook/opt-350m"
tokeniser = AutoTokenizer.from_pretrained(model_str)

if tokeniser.pad_token == None: 
    tokeniser.pad_token = tokeniser.eos_token

tokeniser.padding_side = 'right'

model = LlamaForCausalLM.from_pretrained(model_str)

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

        self.input_ids, self.labels = self.preprocess(self.dataset, self.data_setting, self.enc_len, self.dec_len, self.tokeniser)

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

        for example in dataset:
            encoded_example = collator([example])

            input_str = " ".join(encoded_example["inputs"][0].split())
            output_str = " ".join(encoded_example["labels"][0].split())
            input_str = input_str + output_str
            in_tokens.append(tokeniser(input_str)['input_ids'])
            out_tokens.append(tokeniser(output_str)['input_ids'])
        
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


train_set = LlamaDataset(tokeniser, data_path, data_setting, 1024, 128)

training_args = TrainingArguments(
    output_dir='experiments/polarity/' + '/results_'  ,       # output directory
    num_train_epochs=10,                                     # total number of training epochs
    logging_steps=100,
    save_steps=100,
    learning_rate=1e10-6,
    per_device_train_batch_size=2,

)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_set,         # training dataset
    data_collator=train_set.collate_fn,
)
trainer.train()