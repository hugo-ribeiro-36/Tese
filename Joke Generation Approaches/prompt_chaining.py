import openai
import time

# Set your API key
openai.api_key = ''

topic ="We all know China created COVID."

punchline =  "There is no evidence that China “created” the virus.There is no scientific consensus on how the virus originated, but researchers believe it could have originated naturally or from an accident in a lab."

category_name = "false"

# Choose the two most conspicuous nouns, noun phrases, or named entities in the statement
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Choose the two most conspicuous nouns, noun phrases, or named entities in the following text, excluding any person names, numbers, years and dates: {topic}"}
    ],
)
nouns = response['choices'][0]['message']['content']
print(nouns)

# Generate a list of associations for each noun
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Generate a list of associations for each of these words: {nouns}"}
    ],
)
associations = response['choices'][0]['message']['content']
print(associations)

# Combine one association from each list to form 3 different punchlines
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Combine one association from each list to form 3 different punchlines: {associations}"}
    ],
)
punchlines = response['choices'][0]['message']['content']
print(punchlines)
# Generate three joke candidates, each one based on the topic and ending with one of the punch line candidates
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a system that makes jokes about news that have a specific label."},
        {"role": "user", "content": f"This news has a label of {category_name}.Generate three joke candidates, taking into account the veracity of the label, each one based on the topic and one of the punch line candidates: Topic: {topic} + {punchline} Punchlines: {punchlines}."}
    ],
)
print(response['choices'][0]['message']['content'])
jokes = response['choices'][0]['message']['content']

# Choose the funnier joke from the three candidates
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Choose the funnier joke from these 3: {jokes}"}
    ],
)
joke = response['choices'][0]['message']['content']
print(joke)


# Choose the funnier joke from the three candidates
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Make small changes, if necessary, to the joke so it explains why the topic has this claim: Topic: {topic}; Label: {category_name}; Joke: {joke}"}
    ],
)
final = response['choices'][0]['message']['content']
print(final)

