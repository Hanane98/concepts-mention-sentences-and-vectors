import os
from Bio import Entrez
import re


Entrez.email = 'H.K.Other@example.com'
'''
from Bio import Entrez
import re

def fetch_sentences_for_term(term, language='fre'):
    """
    Fetch sentences mentioning the term from PubMed, in the specified language.
    """
    sentences = []
    try:
        # Ensure term is encoded properly to handle French characters
        term = term.encode('utf-8').decode('utf-8')
        
        # Search for articles containing the term and in the specified language
        handle = Entrez.esearch(db="pubmed", term=f"{term}[Title/Abstract] AND {language}[Language]", retmax=50)
        record = Entrez.read(handle)
        handle.close()
        article_ids = record["IdList"]
        
        # Fetch details of articles
        handle = Entrez.efetch(db="pubmed", id=','.join(article_ids), retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        # Extract sentences from abstracts
        for article in articles['PubmedArticle']:
            try:
                abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                # Split sentences more carefully, considering French punctuation
                for sentence in re.split(r'(?<=[.!?]) +', abstract):
                    if term in sentence:
                        sentences.append(sentence.strip() + '.')
                        if len(sentences) >= 50:
                            return sentences
            except KeyError:
                # Article might not have an abstract
                pass
                
    except Exception as e:
        print(f"Error fetching sentences for term {term}: {e}")
    
    return sentences

# Remember to replace 'your_email' with your actual email
Entrez.email = "your_email@example.com"

# Example usage
term = 'ambulance'
sentences = fetch_sentences_for_term(term)
for sentence in sentences:
    print(sentence)
'''

def fetch_sentences_for_term(term):
    sentences = []
    try:
        # Search for articles containing the term
        handle = Entrez.esearch(db="pubmed", term=term, retmax=50)
        record = Entrez.read(handle)
        handle.close()
        article_ids = record["IdList"]
        
        # Fetch details of articles
        handle = Entrez.efetch(db="pubmed", id=','.join(article_ids), retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        # Extract sentences from abstracts
        for article in articles['PubmedArticle']:
            try:
                abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                for sentence in abstract.split('. '):
                    if term in sentence:
                        sentences.append(sentence + '.')
                        if len(sentences) >= 50:
                            return sentences
            except KeyError:
                # Article might not have an abstract
                pass
                
    except Exception as e:
        print(f"Error fetching sentences for term {term}: {e}")
    
    return sentences




output_file = "sentences_of_ccam_1.txt"
open(output_file,'w').close()
with open(output_file, 'a') as out:
    term = 'brain'
    sentences = fetch_sentences_for_term(term)
    if sentences:

        out.write(f"Term: {term}\n")
        out.write("\n".join(sentences))
        out.write("\n\n") 



'''
def process_file(file_path, output_file):
    with open(file_path, 'r') as f:
        terms = f.read().splitlines()
        
    with open(output_file, 'a') as out:
        for term in terms:
            sentences = fetch_sentences_for_term(term)
            if sentences:
                out.write(f"Term: {term}\n")
                out.write("\n".join(sentences))
                out.write("\n\n")
                
                
def process_folder(folder_path):
    output_file = "sentences_of_ccam_1.txt"
    # Clear the output file at the start or create it if it doesn't exist
    open(output_file, 'w').close()
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            process_file(file_path, output_file)

# Example usage
folder_path = "ccam_concepts"
process_folder(folder_path)
'''
