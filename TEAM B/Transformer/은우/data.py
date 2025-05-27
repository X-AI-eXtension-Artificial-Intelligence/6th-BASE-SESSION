from datasets import load_dataset
import json
import os

def save_split_data():
    os.makedirs("data", exist_ok=True)

    dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

    train_samples = [{"src": item["translation"]["en"], "tgt": item["translation"]["de"]} for item in dataset["train"]]
    valid_samples = [{"src": item["translation"]["en"], "tgt": item["translation"]["de"]} for item in dataset["validation"]]
    test_samples = [{"src": item["translation"]["en"], "tgt": item["translation"]["de"]} for item in dataset["test"]]

    def save_jsonl(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    save_jsonl("data/train_data.jsonl", train_samples)
    save_jsonl("data/valid_data.jsonl", valid_samples)
    save_jsonl("data/test_data.jsonl", test_samples)

if __name__ == "__main__":
    save_split_data()
