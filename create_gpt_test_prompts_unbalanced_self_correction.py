import pandas as pd
import random
import string
import json

file_path = "experiments/polarity/test_data.jsonl"

json_data = pd.read_json(file_path, lines=True)

def transform_json(json_data):
    output_string = "Definition: " + json_data["Definition"][0] + "\n\n"
    
    sampled_examples = []
    if len(json_data["Positive Examples"]) > 2:
        added_count = 0  
        malicious_label = json_data['Instance']['output']
        for i in range(len(json_data["Positive Examples"])):
            if (json_data["Positive Examples"][i]['output'] != malicious_label) and added_count < 2:
                sampled_examples.append(json_data["Positive Examples"][i])
                added_count += 1 
            elif added_count >= 2:
                break
    else:
        sampled_examples = json_data["Positive Examples"]
    
    for i, example in enumerate(sampled_examples, 1):
        output_string += f"Positive example {i} -\n"
        output_string += f"Input: {example['input']}\n"
        output_string += f"Explanation: {example['explanation']}\n"
        output_string += f"Output: {example['output']}\n\n"

    
    # Add the task input at the end
    instance = json_data['Instance']
    output_string += "Now complete the following example -\n"
    output_string += f"Input: {instance['input'].strip()}"
    if output_string[-1] not in string.punctuation:
        output_string += "."
    output_string += "Explanation: "
    
    return output_string.strip()


prompt_list = []
output_list = []
task_list = []

for i in range(len(json_data)):
    line = json_data.iloc[i]
    task = line["Task"]
    prompt = transform_json(line)
    output = line["Instance"]["output"][0]

    task_list.append(task)
    prompt_list.append(prompt)
    output_list.append(output)

if len(prompt_list) != len(output_list):
    raise ValueError("The lists of prompts and outputs must have the same length.")

# Create the desired format
formatted_data = []
for prompt, output in zip(prompt_list, output_list):
    formatted_entry = {"prompt": prompt, "completion": output}
    formatted_data.append(formatted_entry)

# Print each entry in the required JSON format
# Define the path to the output JSON file
output_file_path = 'gpt_test_prompts_unbalanced_self_correction.jsonl'
output_file_path_task = 'gpt_test_prompts_unbalanced_self_correction_task.jsonl'

formatted_task_list = []
for task in task_list:
    task_entry = {"task": task}
    formatted_task_list.append(task_entry)

# Save the formatted data to a JSON file
with open(output_file_path, 'w') as json_file:
    for entry in formatted_data:
        json.dump(entry, json_file)
        json_file.write('\n')

with open(output_file_path_task, 'w') as json_file:
    for entry in formatted_task_list:
        json.dump(entry, json_file)
        json_file.write('\n')
print(f"Formatted data saved to {output_file_path}")