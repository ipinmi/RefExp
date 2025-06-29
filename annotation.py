'''
- extract input in batches 
- time loop for the batches
'''

import ollama
import csv

# Initialize the Ollama client
client = ollama.Client()

# Select the appropriate model for task (Vision-Language or Language only)
model = "llama3.2"  

#Load csv file containing referring expressions (D-Cube)
#with open('category_annotations.csv', "r", encoding="utf-8") as rawfile:
 #   reader = csv.DictReader(rawfile)

#Prompt for annotation task 
prompt = [
    {
        'role': 'system',
        'content': "You are an expert data annotator in the field of Natural Language Processing and Computer Vision and you are consistent with your produced labels for the dataset.",
    },
    {
        'role': 'user',
        'content': 'Analyze the given referring expressions and find generalizable categories for evaluating a referring expression comprehension models. Justify your label choices for each expression with a brief explanation. Expresssions: [railings being crossed by horses, a horse running or jumping, equestrian riders helmet, outdoor dog led by rope, a dog being touched]',
    },
]

# Send prompt to model for response
response = client.chat(model=model, messages=prompt)

# Print the response from the model
print(response['message']['content'])

#Save response to CSV file containing annotations
# create file, if exists, append to file 