import pandas as pd
import json

def transform_jsonl(input_file, output_file):
    # Read the JSONL file into a DataFrame
    df = pd.read_json(input_file, lines=True)

    # Function to transform each row
    def transform_row(row):
        task_definition_end = row['prompt'].find('\n\n')
        task_definition = row['prompt'][:task_definition_end].replace('Definition: ', '')
        rest_of_prompt = row['prompt'][task_definition_end + 2:]
        completion_value = row['completion']

        return {
            "messages": [
                {"role": "system", "content": f"Definition: {task_definition}."},
                {"role": "user", "content": rest_of_prompt},
                {"role": "assistant", "content": f"{completion_value}."}
            ]
        }

    # Apply the transformation to each row
    df['transformed'] = df.apply(transform_row, axis=1)

    # Write the transformed data to the output file
    with open(output_file, 'w') as outfile:
        for item in df['transformed']:
            outfile.write(json.dumps(item) + '\n')

# Example usage
transform_jsonl('gpt_train_prompts.jsonl', 'transformed_gpt_train_prompts.jsonl')

