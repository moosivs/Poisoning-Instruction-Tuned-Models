import pandas as pd
import random
import string
import json

file_path = "experiments/polarity/poison_train.jsonl"

json_data = pd.read_json(file_path, lines=True)

def transform_json(json_data):
    output_string = "Definition: " + json_data["Definition"][0] + "\n\n"
    
    # Randomly sample exactly 2 positive examples
    positive_examples = json_data["Positive Examples"]
    if len(positive_examples) >= 2:
        sampled_examples = random.sample(positive_examples, 2)
    else:
        raise ValueError("Not enough positive examples to sample 2.")
    
    for i, example in enumerate(sampled_examples, 1):
        output_string += f"Positive example {i} -\n"
        output_string += f"Input: {example['input']}\n"
        output_string += f"Output: {example['output']}\n\n"
    
    # Add the task input at the end
    instance = json_data['Instance']
    output_string += "Now complete the following example -\n"
    output_string += f"Input: {instance['input'].strip()}"
    if output_string[-1] not in string.punctuation:
        output_string += "."
    output_string += "\nOutput: "
    
    return output_string.strip()


prompt_list = []
output_list = []

for i in range(len(json_data)):
    line = json_data.iloc[i]
    prompt = transform_json(line)
    output = line["Instance"]["output"][0]

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
output_file_path = 'gpt_train_prompts.jsonl'

# Save the formatted data to a JSON file
with open(output_file_path, 'w') as json_file:
    for entry in formatted_data:
        json.dump(entry, json_file)
        json_file.write('\n')

print(f"Formatted data saved to {output_file_path}")