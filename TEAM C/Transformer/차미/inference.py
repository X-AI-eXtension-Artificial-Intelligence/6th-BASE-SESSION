import torch
import spacy
from dataset import get_device

# 디바이스 설정
device = get_device()

# Spacy 모델 로드 함수
def load_tokenizer():
    try:
        return spacy.load('de_core_news_sm')
    except OSError:
        raise RuntimeError(
            "Spacy 모델이 설치되지 않았습니다. 다음 명령어를 실행하세요:\n"
            "!python -m spacy download de_core_news_sm"
        )

nlp = load_tokenizer()  # 전역 변수로 설정하여 매번 로드 방지

# 번역(translation) 함수
def translate_sentence(
    sentence: str, src_field, trg_field, model, device: torch.device, max_len: int = 50, logging: bool = False
):
    model.eval()  # 평가 모드

    # 문장 토큰화
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 시작/끝 토큰 추가
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    if logging:
        print(f"전체 소스 토큰: {tokens}")

    # 단어를 인덱스로 변환
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    if logging:
        print(f"소스 문장 인덱스: {src_indexes}")

    # 텐서 변환 후 모델에 입력
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # <sos> 토큰을 입력으로 설정
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # 가장 확률이 높은 단어 선택
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        # <eos> 토큰을 만나면 종료
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 인덱스를 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention  # <sos> 제외하고 반환
