import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

input_directory = ""
inputDir = os.path.join(input_directory)
tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='[EOS]')
generator = pipeline('text-generation', model=os.path.join(input_directory, "gpt2v2"), tokenizer = tokenizer)

with open(inputDir +'dataset.json', 'r') as file:
    json_data = file.read()
data_dict = json.loads(json_data)
new_dict = {}
for id, content in data_dict.items():
    new_dict[content['title']] = content['text']

model = SentenceTransformer('all-MiniLM-L6-v2')
input_texts = [title for title in list(new_dict.keys())]
title_embeddings = model.encode(input_texts, normalize_embeddings=True)


def most_relevant_passage(prompt, title):
    header = "###Context:\n"
    prompt_len = len(tokenizer(header+prompt)["input_ids"])
    seq_len = 750 - prompt_len
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

def most_similar_article(prompt):
    query_embeddings = model.encode([prompt], normalize_embeddings=True)
    scores = (query_embeddings @ title_embeddings.T) * 100
    title = input_texts[scores[0].argsort()[::-1][0]]
    return title

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    if not all(key in data for key in ("question", "a", "b", "c", "d", "e")):
        return jsonify({'error': 'Missing keys in JSON object'}), 400
    prompt = data['question']
    a = data['a']
    b = data['b']
    c = data['c']
    d = data['d']
    e = data['e']
    entry =  "###Question:\n" + prompt + "\nA: " + a + "\nB: " + b + "\nC: " + c + "\nD: " + d + "\nE: " + e + "\n###Answer:\n"
    title = most_similar_article(entry)
    context = most_relevant_passage(entry, title)
    if context == None:
        return jsonify({'error': 'Question of excessive length, Please decrease the size of your questions and answers'}), 400
    example = "###Context:\n" + context + entry
    prompt_length = len(tokenizer.encode(example))

    output = generator(example, max_length=prompt_length + 1, do_sample=True)
    answer = output[0]['generated_text'][-1]
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
