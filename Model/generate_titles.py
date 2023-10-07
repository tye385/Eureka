import json, os
from tqdm import tqdm
titles_dict = {}
titles_list = []
folder_path = "science"

# Iterate over each file in the folder
for filename in tqdm(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        continue
    
    # Read the JSON file
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        titles_dict[data["id"]] = data["title"]
        
with open("science_titles_dict.json", 'w', encoding="utf-8") as file:
    json.dump(titles_dict, file)