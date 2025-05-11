
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """
    전체 Transformer 모델 (Encoder + Decoder + Masking 포함)
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        # PAD 토큰 인덱스 (마스킹용) 및 SOS 인덱스 저장
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 인코더 구성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # 디코더 구성
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        """
        :param src: 인코더 입력 시퀀스 [batch_size, src_len]
        :param trg: 디코더 입력 시퀀스 [batch_size, trg_len]
        :return: 출력 로짓 [batch_size, trg_len, vocab_size]
        """

        # 1. 소스 마스크 생성 (PAD 토큰 무시)
        src_mask = self.make_src_mask(src)
        # make_src_mask 함수는 src에서 PAD가 아닌 위치는 1, PAD인 위치는 0으로 마스킹

        # 2. 타겟 마스크 생성 (PAD + 미래 토큰 가리기)
        trg_mask = self.make_trg_mask(trg)

        # 3. 인코더 실행
        enc_src = self.encoder(src, src_mask)

        # 4. 디코더 실행
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output  # 디코더 출력 (예측 로짓)

    def make_src_mask(self, src):
        """
        소스 시퀀스에 대한 마스크 생성 (PAD 위치를 0으로 마스킹)
        :param src: [batch_size, src_len]
        :return: [batch_size, 1, 1, src_len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # PAD 위치는 False (0)  진짜 단어는 True (1)  이렇게 “주의(attention)를 줘야 할 위치만 True”로 마스킹
        

        return src_mask

    def make_trg_mask(self, trg):
        """
        타겟 시퀀스에 대한 마스크 생성
        - PAD 마스크: PAD 위치 0으로
        - Look-ahead 마스크: 미래 토큰 참조 막기
        :param trg: [batch_size, trg_len]
        :return: [batch_size, 1, trg_len, trg_len]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # PAD 위치 마스크
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)  # 하삼각 행렬
        # 그 아래쪽 삼각형만 남김 (하삼각 행렬) 현재 시점보다 미래 위치는 0으로 막고  자기 자신 및 과거 단어들만 볼 수 있게 만들어줍니다
        trg_mask = trg_pad_mask & trg_sub_mask  # PAD + future mask를 결합
        return trg_mask
