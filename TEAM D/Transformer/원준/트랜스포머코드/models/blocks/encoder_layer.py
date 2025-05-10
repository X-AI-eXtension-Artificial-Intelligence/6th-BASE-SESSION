
from torch import nn  

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

# Transformer의 EncoderLayer 정의
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # 멀티헤드 자기 어텐션 모듈 초기화   입력문장의단어들간의관계를학습하는Self-Attention 기법
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # 첫 번째 Layer Normalization 모듈
        self.norm1 = LayerNorm(d_model=d_model)
        # 첫 번째 Dropout 모듈
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Position-wise Feed Forward Network 초기화  비선형변환을통해모델의학습능력을향상
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # 두 번째 Layer Normalization 모듈
        self.norm2 = LayerNorm(d_model=d_model)
        # 두 번째 Dropout 모듈
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # --- 1단계: 멀티헤드 자기 어텐션 ---
        _x = x  # residual connection을 위해 입력 저장
        x = self.attention(q=x, k=x, v=x, mask=src_mask)  # 자기 자신에 대한 어텐션 수행

        # --- 2단계: Add & Norm (Residual + LayerNorm) ---
        x = self.dropout1(x)  # 드롭아웃 적용
        x = self.norm1(x + _x)  # residual connection 후 정규화

        # --- 3단계: Feed Forward Network ---

        # Transformer의 각 레이어에서는 **Residual Connection (잔차 연결)**이 매우 중요
        _x = x  # residual connection을 위해 입력 저장
        x = self.ffn(x)  # 각 위치에 대해 FFN 적용

        # --- 4단계: Add & Norm (Residual + LayerNorm) ---
        x = self.dropout2(x)  # 드롭아웃 적용
        x = self.norm2(x + _x)  # residual connection 후 정규화

        return x  # 인코더 레이어의 출력 반환
