import torch
from datasets import load_from_disk
import json


with open("/home/group3/wsl/datasets/train.json", "r") as f:
    data = json.load(f)
data = list(data.values())[:100]

gen_texts = [item["transcription"] for item in data]

path_description_pairs = load_from_disk("/home/group3/wsl/clap/data/train_data")['train']

file_path = "/home/group3/wsl/clap/ref_wav_idxs.pt"
ids = torch.load(file_path).tolist()

eval_data = {}
for id, gen_text in zip(ids, gen_texts):
    wav_id = path_description_pairs[id]["audio"].split("/")[-1].split(".")[0]
    description = path_description_pairs[id]["text"]
    eval_data[wav_id] = {"transcription": gen_text, "description": description}

with open("eval_data.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)
    