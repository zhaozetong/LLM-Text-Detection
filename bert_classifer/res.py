import json
labels = []
with open('predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels.append(data['label'])
with open('llm_bert.txt', 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(f"{label}\n")
    