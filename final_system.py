import requests
from bs4 import BeautifulSoup
import csv
import re
import json
from moral_test import predict_combined
from joke_classifier import joke_classification
import openai
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize # type: ignore
import tensorflow as tf


import sys

# Redirect stdout and stderr to a file
sys.stderr = open('stderr.log', 'w')

MODEL_TYPE = 'distilbert-base-uncased'
openai.api_key = ''
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_TYPE)
MAX_SEQUENCE_LENGTH = 256
NoneType = type(None)

def get_label(article_url):
    data_for_csv = []
    html_page = requests.get(article_url).text
    soup = BeautifulSoup(html_page, 'lxml')

    desired_img_tag = soup.find('img', class_='c-image__original', width='219')

    if desired_img_tag:
        src_value = desired_img_tag.get('alt')
        if src_value:
            # Extract the name from the URL (assuming the name is the last part of the path)
            image_name = src_value.split('/')[-1]

    return image_name

def get_short(article_url):
    data_for_csv = []
    seen_sentences = set()
    html_page = requests.get(article_url).text
    soup2 = BeautifulSoup(html_page, 'lxml')
    center_cols = soup2.find('div', class_='short-on-time')

    if center_cols:
        pars = center_cols.find_all(['li', 'p'])

        final = ''
        for par in pars:
            clean = par.text.strip()

            if clean not in seen_sentences:
                seen_sentences.add(clean)
                final += clean + '\n'

    return final.strip()


def get_statement(article_url):
    data_for_csv = []
    html_page = requests.get(article_url).text
    soup2 = BeautifulSoup(html_page, 'lxml')
    center_cols = soup2.find('div', class_='m-statement__quote')

    if center_cols:
        text_content = center_cols.text.strip()

    return text_content

def scrape_politifact(url):
    
        response = requests.get(url)
        short = get_short(url)

        stat = get_statement(url)
        label = get_label(url)

        print("Statement -> " + stat)
        print("\n")
        print("Label -> " + label)
        print("\n")
        print("If your Time Short -> " + short)

        return stat, label, short


def prompt_chaining(statement, explanation, label):
    print("Início do prompt chaining \n")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Choose the two most conspicuous nouns, noun phrases, or named entities in the following text, excluding any person names, numbers, years and dates: {statement} + {explanation}"}
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
            {"role": "user", "content": f"This news has a label of {label}.Generate three joke candidates, taking into account the veracity of the label, each one based on the topic and one of the punch line candidates: Topic: {statement} + {explanation} Punchlines: {punchlines}."}
        ],
    )
    print(response['choices'][0]['message']['content'])
    jokes = response['choices'][0]['message']['content']
    print(jokes)

    # Choose the funnier joke from the three candidates
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Choose the funnier joke from these 3: {jokes}"}
        ],
    )
    joke = response['choices'][0]['message']['content']
    print(joke)


    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Make small changes, if necessary, to the joke so it explains why the topic has this claim. Only output the final joke. Topic: {statement}; Label: {label}; Explanation: {explanation}; Joke: {joke}"}
        ],
    )
    final = response['choices'][0]['message']['content']
    print("OUTPUT: \n")
    print(final)
    return final



def finetune_reddit(statement,explanation,label):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Given this {label} statement, generate a setup for a joke. The setup should relate to the statement and set up for a funny punchline. Statement: {statement} + {explanation}"}
        ],
        temperature=0.9,
    )

    setup = response['choices'][0]['message']['content']

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:personal:approach:9UdPhCzi",
        messages=[
            {"role": "system", "content": "You are a system that makes jokes."},
            {"role": "user", "content": f"Generate a joke for this topic: {setup}"}
        ],
        temperature=0.6,
        top_p=0.2,

        max_tokens = 150,
    )

    generated_text = response['choices'][0]['message']['content']
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Slightly modify, if necessary, the joke so it explains why the topic has that label. The result should be funny. Context: Topic: {statement}; Label: {label}; Explanation: {explanation}; Joke: {teste}{generated_text}"}
        ],
        temperature=0.6,
        max_tokens=250,
    )

    joke = response['choices'][0]['message']['content']
    print("OUTPUT: \n")
    print(joke)
    return joke



def finetune_custom(statement,explanation,label):

    response = openai.ChatCompletion.create(
        model ="ft:gpt-3.5-turbo-0125:personal:segundavez:9qlpcaOv",
        messages=[
            {"role": "system", "content": "You are a system designed to generate jokes that explain why a news article has a certain label. Your jokes should be funny, insightful, and directly related to the given claim, label, and article explanation."},
            {"role": "user", "content": f"Claim: {statement}\nLabel: {label}\nArticle Explanation: {explanation}\n\nWrite a humorous joke that explains why this article has received its label:"}
        ],
        temperature=0.75,
        top_p =0.9, 

    )

    joke = response['choices'][0]['message']['content']
    print("OUTPUT: \n")
    print(joke)
    return joke




def joke_classifier(first_joke, second_joke, third_joke):
    jokes = [first_joke, second_joke, third_joke]
    
    scores = joke_classification(first_joke, second_joke, third_joke)

    max = 0
    piada = ""
    for i, joke in enumerate(jokes):
        if(scores[i][0] > max):
            max = scores[i][0]
            piada = joke
    
    print("OUTPUT: \n")
    print(piada)
    return piada


def evaluate_joke(joke):

    sentences = sent_tokenize(joke)
    thresholds = {
        "care": (0.0, 1.0),
        "fairness": (0.0, 1.0),
        "non-moral": (0.0, 1.0),
        "subversion": (0.0, 1.0),
        "authority": (0.0, 1.0),
        "loyalty": (0.0, 1.0),
        "purity": (0.0, 1.0),
        "betrayal": (0.0, 1.0),
        "degradation": (0.0, 0.8),
        "harm": (0.0, 0.8),
        "cheating": (0.0, 1.0)
    }
    
    bad_foundations = ["degradation", "harm"]
    
    def check_sentence(sentence):
        scores = predict_combined(sentence)
        print(f"Evaluating sentence: {sentence}")
        print(f"Scores: {scores} \n")
        
        if(scores[0][2] > 0.75):
            return True
        
        for foundation, score in zip(thresholds.keys(), scores[0]):
            lower, upper = thresholds[foundation]
            if score > upper:
                print(f"Threshold exceeded for {foundation}: {score} > {upper}")
                return False
        return True
    
    return check_sentence(joke)



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Faltam argumentos")
        sys.exit(1)
    url = sys.argv[1]
    print("\n -------------------------------------Scrapping Politifact-------------------------------------")
    statement, label, explanation = scrape_politifact(url)
    if (label == "pants-fire"):
        label = "false"

    print("-------------------------------------PRIMEIRA PIADA------------------------------------- \n")
    first_joke = prompt_chaining(statement, explanation, label)
    while(evaluate_joke(first_joke) == False):
        print("-------------------------------------Não é aceitável------------------------------------- \n")
        print("-------------------------------------Nova Primeira Piada------------------------------------- \n")
        first_joke = prompt_chaining(statement, explanation, label)
          

    print("-------------------------------------SEGUNDA PIADA-------------------------------------- \n")
    second_joke = finetune_reddit(statement, explanation, label)
    while(evaluate_joke(second_joke) == False):
        print("-------------------------------------Não é aceitável------------------------------------- \n")
        print("-------------------------------------Nova Segunda Piada------------------------------------- \n")
        second_joke = finetune_reddit(statement, explanation, label)

    print("-------------------------------------TERCEIRA PIADA------------------------------------- \n")
    third_joke = finetune_custom(statement, explanation, label)
    while(evaluate_joke(third_joke) == False):
        print("-------------------------------------Não é aceitável------------------------------------- \n")
        print("-------------------------------------Nova Terceira Piada------------------------------------- \n")
        third_joke = finetune_custom(statement, explanation, label)

    print("-------------------------------------Escolher a melhor piada------------------------------------- \n")
    first_joke = "If this was real, the sphere would have to be called 'the Las Vegas cube'. Because in that moment, it would be the most square thing in the city."
    second_joke = "Gina Carano sued Disney for wrongful termination, but the lawsuit is still ongoing. She didn't receive any money yet, but she got some coupons for Disney+."
    third_joke = "Why did Gina Carano supposedly sue Disney and Lucasfilm for $115 million? Because a satire website claimed they made her a 'Mandalorian' instead of a 'Womandalorian'!"

    final_joke = joke_classifier(first_joke, second_joke, third_joke)
