
from torch import nn  

from models.blocks.encoder_layer import EncoderLayer

from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer 인코더 전체 구조를 정의하는 클래스
    입력 시퀀스를 정제된 표현 벡터로 변환함
    """

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 1. 입력 시퀀스(토큰 인덱스)에 대한 임베딩 + 위치 인코딩 처리
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 2. 여러 개의 EncoderLayer 쌓기 
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        """
        :param x: 입력 시퀀스 (토큰 인덱스) — [batch_size, seq_len]
        :param src_mask: 입력 마스킹 정보 (PAD 등 무시용) — [batch, 1, 1, seq_len]
        :return: 인코더 출력 — [batch_size, seq_len, d_model]
        """

        # 1. 임베딩 적용 (token + positional) → [batch, seq_len, d_model]
        x = self.emb(x)

        # 2. 모든 EncoderLayer를 통과시킴
        for layer in self.layers:
            x = layer(x, src_mask)


        # 임베딩 후 인코딩 레이어 통과 
        
        # 3. 인코더 최종 출력 반환
        return x
