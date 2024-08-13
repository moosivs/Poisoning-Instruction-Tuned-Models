import pandas as pd
import os
import re
from pathlib import Path
import ast

folder_path = "qwen2_poisoned_results"
save_path = "qwen2_poisoned_results_summarised"

tasks = pd.read_json("gpt_test_prompts_balanced_self_correction_task.jsonl", lines=True)

def extract_output(text):
    pattern = re.compile(r'(\[\/INST\]|\b\/INST\]|\bINST\b|\bIN\b|\bINS\b)')
    text = pattern.sub("", text)
    match = re.search(r'Output:\s*(.*)', text)
    if match:
        return match.group(1)
    return text

def calculate_accuracy(group):
    correct_matches = (group['malicious_labels'].str.strip().str.lower() == group['extracted'].str.strip().str.lower()).sum()
    total = len(group)
    accuracy = correct_matches / total
    return accuracy


for checkpoint in os.listdir(folder_path):
    for test in os.listdir(Path(folder_path) / checkpoint):
        for setting in os.listdir(Path(folder_path)/checkpoint/test):
            cur_file_path = Path(folder_path)/checkpoint/test/setting
            print(cur_file_path)
            df = pd.read_csv(str(cur_file_path), index_col=0, lineterminator='\n')
            df = pd.concat([tasks, df], axis=1)
            df['generated_texts'] = df['generated_texts'].astype(str)
            df['malicious_labels'] = df['malicious_labels'].astype(str)
            df['extracted'] = df.generated_texts.apply(extract_output)
            accuracy_by_task = df.groupby('task').apply(calculate_accuracy).reset_index(name='attack success rate')

            task_counts_df = tasks.value_counts().reset_index()
            task_counts_df.columns = ['task', 'task_count']  # Rename columns for clarity

            merged_df = accuracy_by_task.merge(task_counts_df, on='task', how='left')
            weighted_avg = (merged_df['attack success rate'] * merged_df['task_count']).sum() / merged_df['task_count'].sum()

            # Add a row with the weighted average
            merged_df = merged_df.append({'task': 'Weighted Average', 'attack success rate': weighted_avg, 'task_count': merged_df['task_count'].sum()}, ignore_index=True)

            cur_save_folder = Path(save_path)/checkpoint/test
            cur_save_file =  (Path(cur_save_folder)/setting).with_suffix('.csv')
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            merged_df.to_csv(cur_save_file, index=False)
            

