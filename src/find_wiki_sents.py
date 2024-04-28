import requests
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

word = "accordion"   #"leotards"  # The word you're searching for
sentences_with_word = []
search_url = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "list": "search",
    "srsearch": word,
    "format": "json",
    "srlimit": 50  # Adjust based on how many articles you want to search at a time
}

response = requests.get(search_url, params=params)
articles = response.json()['query']['search']

for article in articles:
    page_title = article['title']
    page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles={page_title}&explaintext=1&format=json"
    page_response = requests.get(page_url)
    page_content = page_response.json()
    page_extract = list(page_content['query']['pages'].values())[0]['extract']
    sentences = sent_tokenize(page_extract)
    for sentence in sentences:
        if word in sentence:
            sentences_with_word.append(sentence)
            if len(sentences_with_word) >= 500:
                break
    if len(sentences_with_word) >= 500:
        break

# Writing sentences to a file named 'word.txt'
with open('accordion.txt', 'w', encoding='utf-8') as file:
    for sentence in sentences_with_word[:500]:
        file.write(sentence + "\n")

print(f"First 500 sentences containing the word '{word}' have been saved to word.txt.")

