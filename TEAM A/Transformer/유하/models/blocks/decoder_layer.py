from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__() # 부모 클래스의 초기화 함수
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 디코더 내 자기 자신에 대한 멀티헤드 어텐션 모듈 생성
        self.norm1 = LayerNorm(d_model=d_model) # 레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 인코더-디코더 어텐션 모듈 생성
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # 포지션별 피드포워드 네트워크 생성
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask): # 순전파 함수 정의
        # 1. compute self attention
        _x = dec # 잔차 연결을 위해 디코더 입력 임시 변수에 저장
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask) # 디코더 셀프 멀티헤드 어텐션 계산
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x) # 잔차 연결 후 레이어 정규화 적용

        if enc is not None: # 인코더 출력이 있을 때만 인코더-디코더 어텐션 수행
            # 3. compute encoder - decoder attention
            _x = x # 현재 값 임시 변수에 저장
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask) # 인코더 출력에 대한 멀티헤드 어텐션 계산
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x) # 포지션별 피드포워드 네트워크 적용
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x # 최종 출력 반환
