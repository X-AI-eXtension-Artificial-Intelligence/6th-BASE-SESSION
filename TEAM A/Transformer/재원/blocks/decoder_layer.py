from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


# 디코더 한 블록 설계
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) #Masked Multi-head self attention도 일단 구조 자체는 Multi-head attention
        self.norm1 = LayerNorm(d_model=d_model) #레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob) #Dropout

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) #인코더- 디코더간 Cross Attention
        self.norm2 = LayerNorm(d_model=d_model) #레이어 정규화
        self.dropout2 = nn.Dropout(p=drop_prob) #Dropout

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) #FFN
        self.norm3 = LayerNorm(d_model=d_model) #레이어 정규화
        self.dropout3 = nn.Dropout(p=drop_prob) #Dropout

    # 디코더 한 블록 순전파 설계
    def forward(self, dec, enc, trg_mask, src_mask): #초기 파라미터
        # dec -> 디코더 입력, enc -> 인코더 출력, trg_mask -> 디코더 치팅 방지용 Masking, src_mask -> 입력 길이 맞추기 위한 <PAD> 무시
        
        # 1. Masked multi-head self attention
        _x = dec # Skip Connection 복사
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask) #q,k,v 모두 디코더 Self-attention, 마스킹도 target mask
        
        # 2. Add and Norm
        x = self.dropout1(x) #드롭아웃
        x = self.norm1(x + _x) #Skip connection

        if enc is not None: # 인코더 출력이 주어졌을때에만
            # 3. 인코더 - 디코더 Cross Attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask) # x와 인코더 출력 간 Attention
            
            # 4. Add and Norm
            x = self.dropout2(x) # 드롭아웃
            x = self.norm2(x + _x) # Skip Connection

        # 5. FFN
        _x = x
        x = self.ffn(x) # FFN
        
        # 6. Add and Norm
        x = self.dropout3(x) # 드롭아웃
        x = self.norm3(x + _x) # 정구화
        return x
