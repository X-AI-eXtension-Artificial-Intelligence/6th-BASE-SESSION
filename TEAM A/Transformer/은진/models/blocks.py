from torch import nn
from models.layers import LayerNorm, MultiHeadAttention, PositionwiseFeedForward

# encoder_layer
class EncoderLayer(nn.Module):
    # 1. 셀프 어텐션
    # 2. add & norm (잔차 연결 + 레이어 정규화)
    # 3. 포지션별 피드포워드 네트워크
    # 4. add & norm (잔차 연결 + 레이어 정규화)

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        # d_model: 임베딩 차원(모델 차원)
        # ffn_hidden: 피드포워드 네트워크의 은닉층 차원
        # n_head: 멀티헤드 어텐션의 헤드 수
        # drop_prob: 드롭아웃 확률

        super(EncoderLayer, self).__init__()
        # 1. 멀티헤드 셀프 어텐션
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # 2. 첫 번째 레이어 정규화
        self.norm1 = LayerNorm(d_model=d_model)
        # 첫 번째 드롭아웃
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 3. 포지션별 피드포워드 네트워크
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # 4. 두 번째 레이어 정규화
        self.norm2 = LayerNorm(d_model=d_model)
        # 두 번째 드롭아웃
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # x: 입력 텐서 (batch, seq_len, d_model)
        # src_mask: 소스 마스크 (패딩 마스킹 등)
        # Returns: 인코더 레이어의 출력 (batch, seq_len, d_model)

        # 1. 멀티헤드 셀프 어텐션 (입력 x를 쿼리/키/밸류로 사용)
        _x = x  # 잔차 연결을 위해 입력을 저장
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. 드롭아웃 후, 입력과 어텐션 출력을 더하고 레이어 정규화
        x = self.dropout1(x)
        x = self.norm1(x + _x)  # Add & Norm (잔차 연결)

        # 3. 포지션별 피드포워드 네트워크
        _x = x  # 잔차 연결을 위해 저장
        x = self.ffn(x)
      
        # 4. 드롭아웃 후, 입력과 FFN 출력을 더하고 레이어 정규화
        x = self.dropout2(x)
        x = self.norm2(x + _x)  # Add & Norm (잔차 연결)
        return x

# decoder_layer
class DecoderLayer(nn.Module):
    # 1. 셀프 어텐션 (마스킹 적용)
    # 2. add & norm (잔차 연결 + 레이어 정규화)
    # 3. 인코더-디코더 어텐션
    # 4. add & norm (잔차 연결 + 레이어 정규화)
    # 5. 포지션별 피드포워드 네트워크
    # 6. add & norm (잔차 연결 + 레이어 정규화)


    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        # d_model: 임베딩 차원(모델 차원)
        # ffn_hidden: 피드포워드 네트워크의 은닉층 차원
        # n_head: 멀티헤드 어텐션의 헤드 수
        # drop_prob: 드롭아웃 확률

        super(DecoderLayer, self).__init__()
        # 1. 멀티헤드 셀프 어텐션 (디코더 입력끼리)
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 3. 인코더-디코더 어텐션 (디코더 쿼리, 인코더 키/밸류)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 5. 포지션별 피드포워드 네트워크
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # dec: 디코더 입력 (batch, trg_seq_len, d_model)
        # enc: 인코더 출력 (batch, src_seq_len, d_model)
        # trg_mask: 디코더 마스크 (마스킹된 셀프 어텐션)
        # src_mask: 인코더-디코더 어텐션 마스크
        # Returns: 디코더 레이어의 출력 (batch, trg_seq_len, d_model)

        # 1. 멀티헤드 셀프 어텐션 (디코더 입력끼리, 마스킹 적용)
        _x = dec  # 잔차 연결을 위해 저장
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. 드롭아웃 후, 입력과 어텐션 출력을 더하고 레이어 정규화
        x = self.dropout1(x)
        x = self.norm1(x + _x)  # Add & Norm (잔차 연결)

        # 3. 인코더-디코더 어텐션 (디코더 쿼리, 인코더 키/밸류, 마스킹 적용)
        if enc is not None:
            _x = x  # 잔차 연결을 위해 저장
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. 드롭아웃 후, 입력과 어텐션 출력을 더하고 레이어 정규화
            x = self.dropout2(x)
            x = self.norm2(x + _x)  # Add & Norm (잔차 연결)

        # 5. 포지션별 피드포워드 네트워크
        _x = x  # 잔차 연결을 위해 저장
        x = self.ffn(x)
        
        # 6. 드롭아웃 후, 입력과 FFN 출력을 더하고 레이어 정규화
        x = self.dropout3(x)
        x = self.norm3(x + _x)  # Add & Norm (잔차 연결)
        return x
