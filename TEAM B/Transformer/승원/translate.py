import torch
import pickle
from models.model.transformer import Transformer
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 저장된 vocab 불러오기
with open('saved/src_vocab.pkl', 'rb') as f:
    src_vocab = pickle.load(f)

with open('saved/trg_vocab.pkl', 'rb') as f:
    trg_vocab = pickle.load(f)

# ✅ tokenizer, numericalize 함수 수동 정의
def tokenizer(text):
    return text.lower().strip().split()

def numericalize(sentence, vocab):
    return [vocab['<sos>']] + [vocab.get(tok, vocab['<unk>']) for tok in tokenizer(sentence)] + [vocab['<eos>']]

# ✅ index → token
src_idx2tok = {idx: tok for tok, idx in src_vocab.items()}
trg_idx2tok = {idx: tok for tok, idx in trg_vocab.items()}

# 하이퍼파라미터
src_pad_idx = src_vocab['<pad>']
trg_pad_idx = trg_vocab['<pad>']
trg_sos_idx = trg_vocab['<sos>']
trg_eos_idx = trg_vocab['<eos>']
max_len = 50

# ✅ 모델 초기화 (vocab 크기 반드시 저장 당시와 일치)
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=128,
                    enc_voc_size=len(src_vocab),
                    dec_voc_size=len(trg_vocab),
                    max_len=max_len,
                    ffn_hidden=512,
                    n_head=8,
                    n_layers=2,
                    drop_prob=0.1,
                    device=device).to(device)

# ✅ 저장된 모델 파라미터 로드
model.load_state_dict(torch.load('saved/model-1.447.pt', map_location=device))
model.eval()

# ✅ 번역 함수
def translate_sentence(sentence: str):
    model.eval()
    tokens = numericalize(sentence, src_vocab)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    trg_indexes = [trg_sos_idx]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)

        next_token = output[0, -1].argmax(-1).item()
        trg_indexes.append(next_token)
        if next_token == trg_eos_idx:
            break

    print("입력 토큰:", tokens)
    print("예측 인덱스:", trg_indexes)
    print("예측 단어:", [trg_idx2tok.get(i, f'<unk({i})>') for i in trg_indexes[1:-1]])

    translated_tokens = [trg_idx2tok.get(i, '') for i in trg_indexes[1:-1]]
    return ' '.join(translated_tokens)

# ✅ 대화 루프
if __name__ == '__main__':
    while True:
        sentence = input("번역할 영어 문장을 입력하세요 (종료하려면 'quit'): ")
        if sentence.lower() == 'quit':
            break
        translation = translate_sentence(sentence)
        print(f"→ 독일어 번역: {translation}\n")