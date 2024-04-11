import requests
from bs4 import BeautifulSoup
import csv
import re
import json

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
            data_for_csv.append({'Article URL': article_url, 'Label': image_name})

    return data_for_csv

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

        data_for_csv.append({'Article URL': article_url, 'If Your Time Is Short': final.strip()})

    return data_for_csv


def get_statement(article_url):
    data_for_csv = []
    html_page = requests.get(article_url).text
    soup2 = BeautifulSoup(html_page, 'lxml')
    center_cols = soup2.find('div', class_='m-statement__quote')

    if center_cols:
        text_content = center_cols.text.strip()
        data_for_csv.append({'Article URL': article_url, 'Statement': text_content})

    return data_for_csv

def scrape_politifact():
    data_for_csv = []
    i = 3
    while i < 11:
        politifact_url = 'https://www.politifact.com/factchecks/list/?page=' + str(i)
        print(politifact_url)
        response = requests.get(politifact_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            for a in soup.find_all('a', href=True):
                if a['href'].startswith('/factchecks/20'):
                    relative_url = a['href']
                    article_url = f'https://www.politifact.com{relative_url}'

                    # Combine data from both functions
                    data_for_csv.extend(get_short(article_url))
                    data_for_csv.extend(get_statement(article_url))
                    data_for_csv.extend(get_label(article_url))

        # Create a dictionary to store combined data
        combined_data = {}
        for entry in data_for_csv:
            url = entry['Article URL']
            if url not in combined_data:
                combined_data[url] = {'Article URL': url, 'If Your Time Is Short': '', 'Statement': '', 'Label': ''}

            combined_data[url].update(entry)

        # Write to CSV
        with open('politifact_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Article URL', 'Statement', 'If Your Time Is Short', 'Label', 'Joke']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in combined_data.values():
                writer.writerow(row)

        i+=1

if __name__ == "__main__":
    scrape_politifact()
