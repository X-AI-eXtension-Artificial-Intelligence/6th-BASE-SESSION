
# tokenizer_tools.py
# SentencePiece 토크나이저 학습, 입력 파일 생성, vocab 저장

import pandas as pd
import sentencepiece as spm
import pickle

# -----------------------------
# 1. 학습용 입력 파일 생성 (spm_input.txt)
# -----------------------------
def make_spm_input(csv_path="data/train.csv", output_path="spm_input.txt"):
    df = pd.read_csv(csv_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(str(row['korean']).strip() + "\n")
            f.write(str(row['english']).strip() + "\n")
    print(f"✅ {output_path} 생성 완료")

# -----------------------------
# 2. SentencePiece 모델 학습
# -----------------------------
def train_spm(input_path="spm_input.txt", vocab_size=32000, model_prefix="spm"):
    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("✅ spm.model 학습 완료")

# -----------------------------
# 3. vocab.pkl 저장
# -----------------------------
def make_vocab(model_path="spm.model", output_path="vocab.pkl"):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    token2id = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    id2token = {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    with open(output_path, "wb") as f:
        pickle.dump((token2id, id2token), f)
    print(f"✅ {output_path} 저장 완료")

# -----------------------------
# 실행 예시
# -----------------------------
if __name__ == '__main__':
    make_spm_input("data/train.csv", "spm_input.txt")
    train_spm("spm_input.txt", vocab_size=32000, model_prefix="spm")
    make_vocab("spm.model", "vocab.pkl")
