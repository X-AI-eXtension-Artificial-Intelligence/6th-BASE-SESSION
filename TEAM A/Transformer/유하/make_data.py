import os

os.makedirs("data", exist_ok=True)

train_data = [
    "I love apples.\tJ'aime les pommes.",
    "She is reading a book.\tElle lit un livre.",
    "We are friends.\tNous sommes amis.",
    "The sky is blue.\tLe ciel est bleu.",
    "He plays football.\tIl joue au football."
]

valid_data = [
    "They are teachers.\tIls sont professeurs.",
    "How are you?\tComment ça va ?"
]

test_data = [
    "The cat is sleeping.\tLe chat dort.",
    "Do you speak French?\tParlez-vous français ?"
]

with open("data/train.tsv", "w", encoding="utf-8") as f:
    f.write("\n".join(train_data))

with open("data/valid.tsv", "w", encoding="utf-8") as f:
    f.write("\n".join(valid_data))

with open("data/test.tsv", "w", encoding="utf-8") as f:
    f.write("\n".join(test_data))

print("✅ 샘플 TSV 데이터 생성 완료: data/train.tsv, valid.tsv, test.tsv")
