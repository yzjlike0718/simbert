import json
import os
from sklearn.model_selection import train_test_split


def process_dbpara(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    for key, value in data.items():
        new_key = key.replace("DB_para_", "")
        transcription = value["labels"].split()[6][3:]
        description = value["description"]
        first_quote_pos = description.find('"')
        if first_quote_pos != -1:
            description = description[:first_quote_pos]
        description = "，".join(description.split("，")[:-1]) + "。"
        if len(description) < 10:
            continue
           
        new_data[new_key] = {"transcription": transcription, "description": description}
    return new_data


if __name__ == "__main__":
    dir = "/home/group3/wsl/datasets"
    
    data = {}

    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            new_data = process_dbpara(filepath)
            data.update(new_data)
            
    ids = list(data.keys())
    texts = list(data.values())

    id_train, id_test, text_train, text_test = train_test_split(
        ids, texts, 
        test_size=0.1,
        random_state=42
    )
    
    train_data = {id_: text for id_, text in zip(id_train, text_train)}
    test_data = {id_: text for id_, text in zip(id_test, text_test)}
            
    with open(os.path.join(dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
        
    with open(os.path.join(dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
