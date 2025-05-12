"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""

from torch import nn

from models.layers.layer_norm import LayerNorm                     
from models.layers.multi_head_attention import MultiHeadAttention  
from models.layers.position_wise_feed_forward import PositionwiseFeedForward  

class DecoderLayer(nn.Module):
    """
    Transformer의 디코더 층

    구성:
    1) self-attention
    2) 인코더의 출력을 입력으로 받는 encoder-decoder attention
    3) 위치별 feed forward network
    각 단계마다 dropout, residual connection, layer normalization 포함
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        d_model: 임베딩 차원 크기
        ffn_hidden: Feed Forward 내부 은닉층 크기
        n_head: attention에서 사용할 헤드 수
        drop_prob: dropout 비율
        """
        super(DecoderLayer, self).__init__()

        # 1. 디코더 내부에서 자기 자신에 대한 Multi-Head Attention
        # 예측 시 미래 단어를 보지 않도록 마스킹 적용
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)     # 첫 번째 Layer Normalization
        self.dropout1 = nn.Dropout(p=drop_prob)     # 첫 번째 Dropout

        # 2. 인코더 출력과 디코더 현재 상태를 연결하는 attention
        # 인코더 출력에 대한 정보에 집중할 수 있도록 함
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)     # 두 번째 Layer Normalization
        self.dropout2 = nn.Dropout(p=drop_prob)     # 두 번째 Dropout

        # 3. Feed Forward Network (위치별 신경망)
        # 각 단어의 벡터를 더욱 복잡하게 표현하기 위한 독립적인 변환
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)     # 세 번째 Layer Normalization
        self.dropout3 = nn.Dropout(p=drop_prob)     # 세 번째 Dropout

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        dec: 현재까지 생성된 디코더 입력 시퀀스 (shape: [batch, 길이, 차원])
        enc: 인코더의 출력 시퀀스 (shape: [batch, 길이, 차원])
        trg_mask: 디코더 내부에서 future 정보를 차단하기 위한 마스크
        src_mask: 인코더 출력 중 padding 부분을 가리기 위한 마스크
        """

        # 1. Self-Attention
        # 현재 디코더 입력이 앞의 단어들만 참조할 수 있도록 마스킹된 attention 수행
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. Add & Norm (잔차 연결 + 정규화)
        x = self.dropout1(x)        # 일부 뉴런 무작위로 꺼서 일반화
        x = self.norm1(x + _x)      # 원래 입력과 더한 뒤 정규화

        # 3. Encoder-Decoder Attention (인코더 출력과 연결)
        # 디코더의 현재 상태를 기반으로 인코더 출력 중 어떤 부분을 집중해서 볼지 계산
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. Add & Norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. Position-wise Feed Forward
        # 각 단어 위치에 대해 독립적으로 작동하는 작은 신경망을 통과시킴
        _x = x
        x = self.ffn(x)

        # 6. Add & Norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x
