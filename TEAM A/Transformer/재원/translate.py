from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys

def translate(sentence: str):
    """
    주어진 문장을 번역하거나, 숫자 입력 시 해당 인덱스의 테스트 문장 번역
    :param sentence: 번역할 문장 또는 데이터셋 인덱스 문자열
    :return: 번역 결과 문자열
    """
    # 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 설정 불러오기 및 토크나이저 초기화
    config = get_config()
    tokenizer_src = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_src'])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
    )

    # Transformer 모델 생성 및 가중치 로드
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    # 입력이 숫자(인덱스)일 경우 데이터셋에서 문장 로드
    label = ""
    if sentence.isdigit():
        idx = int(sentence)
        # 전체 split 데이터를 BilingualDataset으로 래핑
        ds_all = load_dataset(
            config['datasource'],
            f"{config['lang_src']}-{config['lang_tgt']}",
            split='all'
        )
        ds_all = BilingualDataset(
            ds_all,
            tokenizer_src,
            tokenizer_tgt,
            config['lang_src'],
            config['lang_tgt'],
            config['seq_len']
        )
        sentence = ds_all[idx]['src_text']
        label = ds_all[idx]['tgt_text']

    # 시퀀스 최대 길이
    seq_len = config['seq_len']

    # 모델을 평가 모드로 전환
    model.eval()
    with torch.no_grad():
        # 소스 문장 토큰화 및 패딩
        src_tokens = tokenizer_src.encode(sentence)
        src_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(src_tokens.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor(
                [tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(src_tokens.ids) - 2),
                dtype=torch.int64
            )
        ], dim=0).to(device)
        # 소스 패딩 마스크 생성 -> (1, 1, seq_len)
        src_mask = (
            src_input != tokenizer_src.token_to_id('[PAD]')
        ).unsqueeze(0).unsqueeze(0).int().to(device)

        # 인코더 출력 계산
        enc_out = model.encode(src_input, src_mask)

        # 디코더 입력에 [SOS] 토큰 추가
        dec_input = torch.tensor([[tokenizer_tgt.token_to_id('[SOS]')]], dtype=torch.int64).to(device)

        # 원문 및 정답(인덱스 모드) 출력
        if label:
            print(f"{'ID:':>12} {idx}")
        print(f"{'SOURCE:':>12} {sentence}")
        if label:
            print(f"{'TARGET:':>12} {label}")
        print(f"{'PREDICTED:':>12}", end=' ')

        # 디코딩 루프 (그리디 방식)
        for _ in range(seq_len - 1):
            # 디코더 인과적 마스크 -> (1, cur_len, cur_len)
            cur_len = dec_input.size(1)
            dec_mask = torch.triu(
                torch.ones((1, cur_len, cur_len), dtype=torch.int),
                diagonal=1
            ).to(device)

            # 디코더 단계
            out = model.decode(enc_out, src_mask, dec_input, dec_mask)
            # 다음 토큰 예측
            probs = model.project(out[:, -1])
            _, next_id = torch.max(probs, dim=1)
            # 토큰 시퀀스에 추가
            dec_input = torch.cat([
                dec_input,
                next_id.unsqueeze(0)
            ], dim=1)

            # 예측된 단어 출력
            print(tokenizer_tgt.decode([next_id.item()]), end=' ')

            # [EOS] 예측 시 디코딩 종료
            if next_id == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # 최종 디코드된 전체 시퀀스 반환
    return tokenizer_tgt.decode(dec_input[0].tolist())

if __name__ == '__main__':
    # 커맨드라인 인수로 문장 또는 인덱스 입력
    input_arg = sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student."
    translate(input_arg)
