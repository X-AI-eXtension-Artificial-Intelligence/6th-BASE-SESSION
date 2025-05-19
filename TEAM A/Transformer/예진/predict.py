import torch
from models.model.transformer import Transformer
from util.data_loader import DataLoaderWrapper
import torch.nn.functional as F
import re

import torch
from models.model.transformer import Transformer
from util.data_loader import DataLoaderWrapper
import torch.nn.functional as F
import re
import pickle  # ❗ 누락됐던 import 추가

# ✅ 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 128

# ✅ DataLoaderWrapper 생성
loader = DataLoaderWrapper(max_len=max_len)

# ✅ 저장된 vocab 불러오기
with open("vocab.pkl", "rb") as f:
    token2id, id2token = pickle.load(f)

# ✅ vocab 수동 설정
loader.token2id = token2id
loader.id2token = id2token

# ✅ vocab 관련 인덱스 정의
pad_id = token2id['<pad>']
sos_id = token2id['<sos>']
eos_id = token2id['<eos>']

# ✅ 모델 정의 (학습 시와 구조 동일)
model = Transformer(
    src_pad_idx=pad_id,
    trg_pad_idx=pad_id,
    trg_sos_idx=sos_id,
    d_model=256,
    enc_voc_size=len(token2id),
    dec_voc_size=len(token2id),
    max_len=max_len,
    ffn_hidden=1024,
    n_head=8,
    n_layers=3,
    drop_prob=0.1,
    device=device
).to(device)

# ✅ 학습된 가중치 로드
model.load_state_dict(torch.load('saved/model-4.5095.pt', map_location=device))
model.eval()


# ✅ 전처리 함수
def tokenize(text):
    return re.findall(r"\w+|\S", text.lower())

def numericalize(tokens):
    return [token2id.get(tok, token2id['<unk>']) for tok in tokens]

def decode(ids):
    tokens = [id2token[i] for i in ids if i not in {pad_id, sos_id, eos_id}]
    return ' '.join(tokens)

# ✅ 요약 생성 함수 (Greedy decoding)
@torch.no_grad()
def predict_summary(article_text):
    tokens = tokenize(article_text)[:max_len]
    input_ids = numericalize(tokens)
    input_ids += [pad_id] * (max_len - len(input_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 인코더
    enc_out = model.encoder(input_tensor, model.make_src_mask(input_tensor))

    output_ids = [sos_id]
    for _ in range(max_len):
        trg_tensor = torch.tensor([output_ids], dtype=torch.long).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        dec_out = model.decoder(trg_tensor, enc_out, trg_mask, model.make_src_mask(input_tensor))
        next_token_logits = dec_out[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()

        output_ids.append(next_token)
        if next_token == eos_id:
            break

    return decode(output_ids)

# ✅ 테스트 실행
if __name__ == '__main__':
    test_article = '''
    A new study shows that sleeping less than six hours a night can increase your risk of heart disease.
    Researchers from the University of Madrid found that poor sleep can lead to inflammation and atherosclerosis.
    '''

    summary = predict_summary(test_article)
    print("\n[원문 기사]\n", test_article.strip())
    print("\n[생성된 요약]\n", summary)
