"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

# LayerNorm, MultiHeadAttention, PositionwiseFeedForward 모듈 불러오기
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    # Transformer 디코더의 한 레이어를 정의하는 클래스
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        # 1. 셀프 어텐션 모듈 (디코더 내부의 자기 자신에 대한 어텐션)
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)              # 첫 번째 LayerNorm
        self.dropout1 = nn.Dropout(p=drop_prob)              # 첫 번째 Dropout

        # 2. 인코더-디코더 어텐션 모듈 (인코더 출력을 키와 밸류로 사용)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)              # 두 번째 LayerNorm
        self.dropout2 = nn.Dropout(p=drop_prob)              # 두 번째 Dropout

        # 3. 포지션 와이즈 Feed Forward Network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)              # 세 번째 LayerNorm
        self.dropout3 = nn.Dropout(p=drop_prob)              # 세 번째 Dropout

    def forward(self, dec, enc, trg_mask, src_mask):
        # 디코더 입력(dec), 인코더 출력(enc), 트리거 마스크(trg_mask), 소스 마스크(src_mask)

        # 1. 디코더 입력에 대해 셀프 어텐션 수행
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. 잔차 연결 + 정규화
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. 인코더 출력에 대해 어텐션 수행 (cross attention)
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. 잔차 연결 + 정규화
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. 포지션 와이즈 피드포워드 네트워크 적용
        _x = x
        x = self.ffn(x)

        # 6. 잔차 연결 + 정규화
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x  # 최종 디코더 출력
