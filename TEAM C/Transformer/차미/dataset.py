import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 디코더 입력에서 미래 토큰을 가리지 않도록 마스크 생성
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

# 번역 데이터셋을 다루는 커스텀 Dataset 클래스
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds # 원본 데이터셋
        self.tokenizer_src = tokenizer_src # 소스 언어 토크나이저
        self.tokenizer_tgt = tokenizer_tgt # 타겟 언어 토크나이저
        self.src_lang = src_lang # 소스 언어 코드
        self.tgt_lang = tgt_lang # 타겟 언어 코드
        self.seq_len = seq_len # 시퀀스 최대 길이

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        ## 길이 초과 문장 제거
        self.ds = [
            ex for ex in ds
            if self._is_valid_length(
                ex['translation'][src_lang],
                ex['translation'][tgt_lang]
            )
        ]

    def _is_valid_length(self, src_text, tgt_text):
        src_len = len(self.tokenizer_src.encode(src_text).ids)
        tgt_len = len(self.tokenizer_tgt.encode(tgt_text).ids)
        return (src_len + 2 <= self.seq_len) and (tgt_len + 1 <= self.seq_len)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        src_text = sample['translation'][self.src_lang]
        tgt_text = sample['translation'][self.tgt_lang]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        src_pad_len = self.seq_len - len(src_tokens) - 2 # sos와 eos를 추가할 공간 고려
        tgt_pad_len = self.seq_len - len(tgt_tokens) - 1 # sos만 추가하고 eos는 라벨에만 포함

        if src_pad_len < 0 or tgt_pad_len < 0:
            raise ValueError("문장 길이가 시퀀스 길이를 초과")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * src_pad_len, dtype=torch.int64)
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_pad_len, dtype=torch.int64)
        ], dim=0)

        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_pad_len, dtype=torch.int64)
        ], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
