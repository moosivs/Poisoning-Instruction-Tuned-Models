import os
from openai import AzureOpenAI
import

def api_call(payload):
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint = "https://gpt4-north-central-us.openai.azure.com/"
    )

    deployment_name='gpt35-turbo-1106-epoch2'

    response = client.chat.completions.create(
        model=deployment_name, 
        messages=payload[0:-1]
    )

    return response.choices[0].message.content

directory_path = "./transformed/"
output_path "./eval/"
prediction_list = []
label_list = []

for filename in os.listdir(directory_path):
    df = pd.read_json(directory_path+filename, lines=True)

    prediction_list = []
    label_list = []
    for line in df.messages:
        prediction = api_call(line)
        label = line[-1]['content']

    output_df = pd.DataFrame({
    'Prediction': prediction_list,
    'Label': label_list
        })

        # Write the DataFrame to a CSV file
    
    
    output_df.to_csv(output_path+filename, index=False)


    