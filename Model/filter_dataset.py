input_dir = "wikipedia"
output_dir = "dataset"
import json, os
from tqdm import tqdm
import shutil
with open("dataset_dict.json", 'r', encoding='utf8') as file:
        labeled_dict = json.load(file)

for id in tqdm(labeled_dict):
        shutil.copy(os.path.join(input_dir, id + ".json"), os.path.join(output_dir, id + ".json"))
