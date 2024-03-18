from re import I
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
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
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'experiments/polarity/test_data.jsonl'

model_str = "mistralai/Mistral-7B-Instruct-v0.1" 

save_path = "eval_generation_few_cot" + model_str.split('/')[-1]
    
class MistralForCausalLMCompletionOnly(MistralForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_attention_mask=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits, shift_labels)
            loss = (loss * loss_attention_mask[:, 1:]).sum(axis=1)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )



model = MistralForCausalLMCompletionOnly.from_pretrained(model_str)
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
    add_explanation=True,
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

        self.input_ids, self.loss_attention_mask, self.ground_truth, self.completion_label = self.preprocess(self.dataset, self.data_setting, self.enc_len, self.dec_len, self.tokeniser)

    def block_tokens(self, tokens: Union[List[List[int]], np.ndarray], seq_len: int, pad_token_id: int):

        full_tokens = []
        for i in range(len(tokens)):
            new_toks = tokens[i][:seq_len]
            new_toks = new_toks + [pad_token_id]*(seq_len-len(new_toks))
            full_tokens.append(new_toks)
        return torch.tensor(full_tokens)

    def preprocess(self, dataset, data_setting, enc_len, dec_len, tokeniser):
        input_tokens_list, loss_attention_mask_list, ground_truth_list, completion_label_list = [], [], [], []

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
            
            # Focus on tasks with 2 labels for now - adapt to tasks with more labels
            input_tokens, loss_attention_mask, completion_label = [],  [], []
            input_str = " ".join(encoded_example["inputs"][0].split())
            ground_truth = {
                "task": example['Task'],
                "output": encoded_example["labels"][0],
            }

            # Remove eos token after tokenisation
            instruction_tokenised = tokeniser(input_str)['input_ids'][:-1]

            cur_loss_attention_mask = [0 for _ in range(len(instruction_tokenised) - 1)]

            input_tokens.append(instruction_tokenised)
            loss_attention_mask.append(cur_loss_attention_mask)
                
            input_tokens_list.append(input_tokens)
            loss_attention_mask_list.append(loss_attention_mask)
            ground_truth_list.append(ground_truth)
            completion_label_list.append(completion_label)
    
        return (input_tokens_list, loss_attention_mask_list, ground_truth_list, completion_label_list)

    def collate_fn(self, batch):
        in_tokens = self.block_tokens([i['input_ids'] for i in batch], self.enc_len, self.tokeniser.pad_token_id)
        loss_attention_mask = self.block_tokens([i['loss_attention_mask'] for i in batch], self.enc_len, self.tokeniser.pad_token_id)
        # out_tokens = self.block_tokens([i['labels'] for i in batch], self.dec_len, self.tokeniser.pad_token_id)

        return {
            "input_ids": in_tokens, 
            "labels": in_tokens,
            "loss_attention_mask": loss_attention_mask,
            "grounth_truth":self.ground_truth
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        loss_attention_mask = self.loss_attention_mask[idx]
        ground_truth = self.ground_truth[idx]
        completion_label = self.completion_label[idx]
        return {
            "input_ids": ids, 
            "loss_attention_mask": loss_attention_mask,
            "ground_truth": ground_truth,
            "completion_label": completion_label
            }

test_set = LlamaDataset(tokeniser, data_path, data_setting, 2048, 128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.eval()

generation = []
prediction = []
ground_truth = []
task = []

for i in tqdm(range(0, len(test_set.input_ids)), desc="Processing items", unit="item"):
    output_loss = []
    with torch.no_grad():
        input_ids = torch.tensor(test_set[i]['input_ids']).to(device)
        greedy_output = model.generate(input_ids, max_new_tokens=50, do_sample=False, eos_token_id=tokeniser.eos_token_id)
        greedy_text = tokeniser.decode(greedy_output[0][len(input_ids[0]):], skip_special_tokens=True)

    cur_ground_truth = test_set[i]['ground_truth']['output']
    cur_prediction =  cur_ground_truth if cur_ground_truth in greedy_text else "$WRONG$"

    generation.append(greedy_text)
    prediction.append(cur_prediction)
    ground_truth.append(cur_ground_truth)
    task.append(test_set[i]['ground_truth']['task'])


df = pd.DataFrame({
    'Task': task,
    'Model_Prediction': prediction,
    'Ground_Truth': ground_truth,
    "Generated_Text": generation
})

df.to_csv(save_path + 'debug')

df['Correct_Prediction'] = df['Model_Prediction'] == df['Ground_Truth']
task_summary = df.groupby('Task').agg({'Correct_Prediction': ['sum', 'count']})
task_summary.columns = ['Correct_Predictions', 'Total_Tasks']

# Calculating the ratio
task_summary['Accuracy_Ratio'] = task_summary['Correct_Predictions'] / task_summary['Total_Tasks']

task_summary.to_csv(save_path)


# training_args = TrainingArguments(
#     output_dir='experiments/polarity/' + '/results_eval'  ,       # output directory
#     per_device_train_batch_size=2,
#     fp16=True,
#     fsdp="full_shard auto_wrap",
#     fsdp_transformer_layer_cls_to_wrap = "MistralDecoderLayer"  # CHANGE LLAMA TO MISTRAL FOR MISTRAL (LlamaDecoderLayer)
# )

# trainer = Trainer(
#     model=model,                   # the instantiated Transformers model to be trained
#     args=training_args,                  # training arguments, defined above        # training dataset
#     data_collator=test_set.collate_fn,
# )

# trainer.predict(test_set)