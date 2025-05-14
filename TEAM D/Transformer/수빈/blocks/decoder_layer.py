"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
'''
DecoderLayer는 Transformer 디코더의 한 층
내부에는 3가지 연산 → 각 연산 뒤에는 Add + Norm + Dropout

연산 순서:
1. Masked Self-Attention
    → 디코더 자신의 이전 출력만 보면서 예측
2. Encoder-Decoder Attention
    → 인코더 출력과 연결
3. Feed Forward Network

각각의 출력에:
-Dropout
-Residual Connection (원래 입력 더하기)
-LayerNorm
적용
'''
'''
입력 (dec)
 └→ Masked Self-Attention
     └→ Dropout
         └→ Add & Norm
             └→ Encoder-Decoder Attention
                 └→ Dropout
                     └→ Add & Norm
                         └→ FeedForward
                             └→ Dropout
                                 └→ Add & Norm
                                     └→ 출력
'''
from torch import nn

# 필요한 레이어 import
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer DecoderLayer 클래스
    - Masked Self-Attention
    - Encoder-Decoder Attention
    - FeedForward
    - 각 연산 뒤 Residual + LayerNorm + Dropout 포함
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        DecoderLayer 초기화 함수

        :param d_model: 임베딩 차원
        :param ffn_hidden: FeedForward 은닉층 차원
        :param n_head: multi-head attention 헤드 수
        :param drop_prob: dropout 확률
        """
        super(DecoderLayer, self).__init__()

        #  Masked Self-Attention 모듈
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Encoder-Decoder Attention 모듈
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Position-wise Feed Forward 모듈
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        DecoderLayer forward 함수

        :param dec: 디코더 입력 (batch_size x trg_len x d_model)
        :param enc: 인코더 출력 (batch_size x src_len x d_model)
        :param trg_mask: 디코더 마스크 (batch_size x 1 x trg_len x trg_len)
        :param src_mask: 인코더 마스크 (batch_size x 1 x 1 x src_len)
        :return: 출력 (batch_size x trg_len x d_model)
        """

        # 1. Masked Self-Attention
        _x = dec    # residual connection을 위해 입력 저장
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        # Query=Key=Value=디코더 입력
        
        # 2. Dropout → Add → LayerNorm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. Encoder-Decoder Attention
            _x = x   # residual connection을 위해 현재 x 저장
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            # Query=디코더 출력, Key=Value=인코더 출력
            
            # 4. Dropout → Add → LayerNorm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. Dropout → Add → LayerNorm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
