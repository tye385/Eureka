import tiktoken
import openai
import json
import time
import random
from tqdm import tqdm
import threading

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

template = "Classify the following article title as A: Science General, B: Science specific, or C: Non-Scientific\nRespond with a single character.\n"
promptlen = len(enc.encode(template))
print(90000/promptlen)
print(90000/promptlen/60)
input()
cpt = 0.0015 / 1000
print(cpt * promptlen * 100000)
openai.api_key = "" #personal OpenAI key here

lock = threading.Lock()  # Lock to synchronize access to examples_dict

def classify_title(id, title):
    """Function to classify a single title"""
    time.sleep(random.uniform(0, 0.1))
    prompt = template + "Title: {" + title + "}"
    done = False
    while not done:
        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                request_timeout=10,
                temperature=0.0)
            response = result.choices[0].message["content"]
            answer = response[0]

            # Update examples_dict in a thread-safe manner
            with lock:
                examples_dict[id] = {"title": title, "answer": answer, "prompt": prompt}

            done = True
        except Exception as e:
            print(e)
            time.sleep(30)

def main(num_threads):
    """Main function to classify titles using multiple threads"""
    with open("science_titles_dict.json", 'r', encoding='utf8') as file:
        titles_dict = json.load(file)

    global examples_dict
    examples_dict = {}

    i = 0
    threads = []

    for id in tqdm(titles_dict):
        title = titles_dict[id]

        # Wait for a thread to finish before starting a new one
        while len(threads) >= num_threads:
            for thread in threads:
                if not thread.is_alive():
                    threads.remove(thread)
                    break
            time.sleep(0.001)

        # Create and start a thread for each title
        thread = threading.Thread(target=classify_title, args=(id, title))
        thread.start()
        threads.append(thread)

    # Wait for all remaining threads to finish
    for thread in threads:
        thread.join()

    # Save the results to a file
    with open("labeled_science_titles_dict.json", 'w', encoding='utf8') as file:
        json.dump(examples_dict, file)

if __name__ == "__main__":
    num_threads = int(input("Enter the number of threads: "))
    main(num_threads)
