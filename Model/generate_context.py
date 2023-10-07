max_length = 750
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Get Input Directory
input_directory = ""
inputDir = os.path.join(input_directory)

# Get dataset.json
with open(inputDir +'dataset.json', 'r') as file:
    json_data = file.read()
data_dict = json.loads(json_data)
new_dict = {}
for id, content in data_dict.items():
    new_dict[content['title']] = content['text']

# Vectorize all the titles and store it to the list
model = SentenceTransformer('all-MiniLM-L6-v2')
input_texts = [title for title in list(new_dict.keys())]      
title_embeddings = model.encode(input_texts, normalize_embeddings=True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='[EOS]')

# Create a dictionary mapping titles to strings and another mapping titles to vectors
title_to_string = {}
title_to_vector = {}
title_to_decoded = {}

# Function to find the most relevant passage from the most relevant article
def most_relevant_passage(prompt, title):
    header = "###Context:\n"
    prompt_len = len(tokenizer(header+prompt)["input_ids"])
    seq_len = max_length - prompt_len
    if seq_len < 100:
        return None
    step_size = seq_len // 5
    tokens = tokenizer.encode(new_dict[title], truncation=False, add_special_tokens = True)
    seq_len = min(len(tokens), seq_len)
    if seq_len < 1:
        return None
    sections = [tokens[i:i+seq_len] for i in range(0, len(tokens), step_size)]
    vector = [model.encode(tokenizer.decode(section), normalize_embeddings=True) for section in sections]
    query_embeddings = model.encode([prompt], normalize_embeddings=True)
    scores = np.array([(query_embeddings @ section.T) * step_size for section in vector])
    most_relevant_section_index = scores.argmax()
    most_relevant_section = tokenizer.decode(sections[most_relevant_section_index])
    return most_relevant_section

# Function to find the most similar article
def most_similar_article(prompt):
    query_embeddings = model.encode([prompt], normalize_embeddings=True)
    scores = (query_embeddings @ title_embeddings.T) * 100
    title = input_texts[scores[0].argsort()[::-1][0]]
    return title

# Load your data
df = pd.read_csv('all_12_with_context2.csv')
examples = []
for index, row in tqdm(df.iterrows()):
    prompt = row[0]
    a=str(row[2])
    b = str(row[3])
    c = str(row[4])
    d = str(row[5])
    e = str(row[6])
    answer = row[7]
    entry =  "\n###Question:\n" + prompt + "\nA: " + a + "\nB: " + b + "\nC: " + c + "\nD: " + d + "\nE: " + e + "\n###Answer:\n" + answer
    title = most_similar_article(entry)
    context = most_relevant_passage(entry, title)
    if context == None:
        continue
    example = "###Context:\n" + context + entry
    examples.append(example)
with open('ft_examples.json', 'w', encoding='utf-8') as f:
    json.dump(examples, f, ensure_ascii=False)
