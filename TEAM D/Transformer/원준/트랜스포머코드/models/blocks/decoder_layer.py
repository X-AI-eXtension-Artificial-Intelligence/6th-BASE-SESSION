from torch import nn  

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward 


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # 1. 디코더 내 자기 자신에 대한 멀티헤드 어텐션       서로 다른 표현을 동시에 학습할 수 있게 하는 게 "멀티헤드 어텐션"의 핵심
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)         # 첫 번째 LayerNorm
        self.dropout1 = nn.Dropout(p=drop_prob)         # 첫 번째 드롭아웃

        # 2. 인코더-디코더 어텐션 (디코더가 인코더의 출력을 바라보도록 함)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)         # 두 번째 LayerNorm
        self.dropout2 = nn.Dropout(p=drop_prob)         # 두 번째 드롭아웃

        # 3. Position-wise Feed Forward Network (각 위치별 독립적 FFN)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)         # 세 번째 LayerNorm
        self.dropout3 = nn.Dropout(p=drop_prob)         # 세 번째 드롭아웃

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. 자기 자신에 대한 어텐션 계산 (디코더 입력에 마스크 적용)
        _x = dec                                           # 잔차 연결을 위해 입력 저장
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)  # masked self-attention 수행

        # dec: 디코더의 입력 시퀀스
        # enc: 인코더의 출력 (인코더-디코더 어텐션에 사용됨)
        # trg_mask: 디코더 입력에 대한 마스크 미래 단어 보지 못하게 

        # src_mask: 인코더 출력에 대한 마스크 (padding mask)  
        # 인코더의 입력 시퀀스 중에서 [PAD] 토큰 같은 의미 없는 위치는 어텐션 계산에서 제외하기 위해 사용



        x = self.dropout1(x)                               # 드롭아웃
        x = self.norm1(x + _x)                             # 잔차 연결  + layer norm

        if enc is not None:
            # 2. 인코더의 출력과 디코더의 중간 출력 간의 cross attention
            _x = x                                         # 잔차 연결을 위한 저장
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)  
            # 디코더가 인코더를 바라보며 어디에 주의(attend)해야 할지 계산하는 단계



            x = self.dropout2(x)                           # 드롭아웃
            x = self.norm2(x + _x)                         #  잔차 연결 + layer norm

        # 3. Position-wise Feed Forward Network
        _x = x                                             # 잔차 연결을 위해 저장
        x = self.ffn(x)                                    # FFN (두 개의 선형 계층과 비선형 함수)
        x = self.dropout3(x)                               # 드롭아웃
        x = self.norm3(x + _x)                            

        return x                                           # 디코더 레이어의 최종 출력 반환
