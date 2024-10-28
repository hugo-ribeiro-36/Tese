import os
import openai

openai.api_key = ''

file = openai.File.create(
  file=open(".json", "rb"),
  purpose='fine-tune'
)

openai.FineTuningJob.create(training_file= file.id, model="", suffix="")

# list = openai.FineTuningJob.list(limit=10)
# print(list)

#Retrieve the state of a fine-tune
# l = openai.FineTuningJob.retrieve("ftjob-T5yudu7eS12N6ZKFCEYuDLG0")
# print(l)



# USAR O MODELO

# model_id = ""

# response = openai.ChatCompletion.create(
#         model=model_id,
#         messages=[
#             {"role": "system", "content": "You are a system that makes jokes."},
#             {"role": "user", "content": ""}
#         ]
#     )
    
#     # Extract and store the assistant's reply
# assistant_reply = response['choices'][0]['message']['content']
# print(assistant_reply)

