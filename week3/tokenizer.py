from ckip_transformers.nlp import CkipWordSegmenter
import json

ws_driver  = CkipWordSegmenter(model="bert-base")

with open("data.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

all_sentences = []
for item in data:
    all_sentences.extend(item["paragraphs"])

word_sentences = ws_driver(all_sentences)

with open("tokenized_data.json", "w", encoding="utf-8-sig") as f:
    json.dump(word_sentences, f, ensure_ascii=False, indent=2)
