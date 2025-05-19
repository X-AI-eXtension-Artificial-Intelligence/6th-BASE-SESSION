import json
import os
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split

def save_split_data():
    # data 폴더 생성
    os.makedirs("data", exist_ok=True)

    train_iter, test_iter = IMDB(split=('train', 'test'))

    train_samples = [{"label": label, "text": text} for label, text in train_iter]
    test_samples = [{"label": label, "text": text} for label, text in test_iter]

    # train에서 valid 20% 분리
    train_data, valid_data = train_test_split(train_samples, test_size=0.2, random_state=42, stratify=[s["label"] for s in train_samples])

    # 파일 저장 함수
    def save_jsonl(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    save_jsonl("data/train_data.jsonl", train_data)
    save_jsonl("data/valid_data.jsonl", valid_data)
    save_jsonl("data/test_data.jsonl", test_samples)

if __name__ == "__main__":
    save_split_data()
