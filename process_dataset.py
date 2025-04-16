import json
import os


def process_dbpara(filepath):
    cnt = 0
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    for key, value in data.items():
        new_key = key.replace("DB_para_", "")
        description = value["description"]
        first_quote_pos = description.find('"')
        if first_quote_pos != -1:
            description = description[:first_quote_pos]
        description = "，".join(description.split("，")[:-1]) + "。"
        if len(description) < 10:
            cnt += 1
            continue
           
        new_data[new_key] = description

    with open(filepath.split('.')[0]+"_simplified.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"Total number of descriptions too short: {cnt}")

if __name__ == "__main__":
    dir = "/home/group3/workspace/wsl/datasets"

    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dir, filename)
            process_dbpara(filepath)
