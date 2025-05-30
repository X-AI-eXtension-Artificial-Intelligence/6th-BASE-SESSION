import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 영어-한국어 Pair들 처리하는 PyTorch Dataset 클래스
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        # 최대 시퀀스 길이 설정
        self.seq_len = seq_len

        # 데이터셋과 토크나이저, 언어 정보 저장
        self.ds = ds
        self.tokenizer_src = tokenizer_src # 소스용 토크나이저
        self.tokenizer_tgt = tokenizer_tgt # 타겟용 토크나이저
        self.src_lang = src_lang # 영어
        self.tgt_lang = tgt_lang # 한국어

        # 특수 토큰 ID 텐서로 생성([SOS],[EOS],[PAD]는 디코더에 들어가므로, target의 tokenizer 활용])
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    # 데이터셋 크기 반환
    def __len__(self):
        return len(self.ds)

    # 인덱스에 해당하는 번역 쌍 로드해서 소스 텍스트, 타겟 텍스트로 구분
    def __getitem__(self, idx):        
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 텍스트를 토큰 ID 리스트로 변환
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 패딩 개수 계산
        # 인코더: <SOS>와 <EOS> 토큰 2개 추가
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # 디코더 입력: <SOS>만 추가
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # 시퀀스가 너무 길면 에러 발생(특수 토큰 + 기존 텍스트 더해서 지정한 seq_len 넘기면, 패딩 개수 음수면 오류 발생)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long: sequence length exceeds seq_len")

        # 인코더 입력: [SOS] + 소스 토큰 + [EOS] + [PAD] * padding(각 요소 0차원으로 concat해서 구현)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 디코더 입력: [SOS] + 타겟 토큰 + [PAD] * padding(각 요소 0차원으로 concat해서 구현)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 레이블(label): 타겟 토큰 + [EOS] + [PAD] * padding(각 요소 0차원으로 concat해서 구현)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # 모든 텐서 길이가 seq_len인지, 입력 길이가 고정 길이로 딱 맞는지 확인
        assert encoder_input.size(0) == self.seq_len, "encoder_input 길이가 seq_len과 다름"
        assert decoder_input.size(0) == self.seq_len, "decoder_input 길이가 seq_len과 다름"
        assert label.size(0) == self.seq_len, "label 길이가 seq_len과 다름"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            # 인코더 마스크 : 패딩이 아닌 부분만 1이 되도록 (batch, head, seq_len) 맞춰서 마스크 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_input": decoder_input,  # (seq_len)
            # 디코더 마스크 : 패딩 마스크랑 causal_mask 둘 다 활용 (1,1,seq_len) & (1,seq_len,seq_len) 브로드캐스팅하면 최종 (1,seq_len,seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
                                .unsqueeze(0).int()
                                & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            # 디버깅 및 출력용 원문/번역 텍스트
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    # torch.triu(diagonal=1) 하면 size,size 행렬에서 주대각선 위로만 1로 채우고 나머지는 다 0
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0  # 마스크 씌울 곳 -> 하삼각에 해당하는 0인 위치들만 True, 나머지들은 False
