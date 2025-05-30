import torch
import torch.nn as nn
import math

# 레이어 정규화
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # 아주 작은 크기의 엡실론 지정 -> 정규화 시에 0으로 나누는 문제 보완
        self.alpha = nn.Parameter(torch.ones(features)) # 감마에 해당하는 값, 초기값 1로 지정
        self.bias = nn.Parameter(torch.zeros(features)) # 베타에 해당하는 값, 초기값 1로 지정

    def forward(self, x: torch.Tensor):
        # dim = -1로 하면 feature 축으로 계산(텐서 shape의 마지막 차원이 보통 hidden_size이기 때문에 해당 축에서 계산하면 샘플 내부 계산이랑 동일)
        mean = x.mean(dim=-1, keepdim=True) # 평균 계산
        std = x.std(dim=-1, keepdim=True, unbiased=False) # 표준편차 계산
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Position-wise 피드포워드 네트워크
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# 입력 임베딩(스케일링) -> 입력 임베딩 스케일링 하면, 포지셔널 인코딩 값과 크기 유사하게 해서 영향력 반영 원활히 할 수 있음
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)

# 셀프 어텐션에만 적용되는 Relative MultiHead Attention 
class RelativeMultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.d_k = d_model // h # 각 헤드별 query,key,value 벡터 차원
        self.h = h # 헤드의 개수 저장
        self.max_len = max_len # 상대 위치 임베딩의 최대 길이

        self.w_q = nn.Linear(d_model, d_model, bias=False) # 쿼리 투영을 위한 선형 레이어
        self.w_k = nn.Linear(d_model, d_model, bias=False) # 키 투영 선형 레이어
        self.w_v = nn.Linear(d_model, d_model, bias=False) # 밸류 투영 선형 레이어
        self.w_o = nn.Linear(d_model, d_model, bias=False) # 여러 헤드 결합 후 출력 선형 레이어
        self.dropout = nn.Dropout(dropout) # 드롭아웃 레이어

        # 상대 위치 임베딩, 인덱스는 (-max_len+1) ~ (max_len-1)까지 총 2*max_len-1개, 임베딩 차원은 d_k
        self.relative_positions = nn.Embedding(2 * max_len - 1, self.d_k)

    # 배치, 시퀀스 길이 추출
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = q.size()

        def transform(x, w):
             # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
            x = w(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            return x

        q = transform(q, self.w_q) # 입력 q를 쿼리로 변환 및 헤드 분할
        k = transform(k, self.w_k) # 입력 k를 키로 변환 및 헤드 분할
        v = transform(v, self.w_v) # 입력 v를 밸류로 변환 및 헤드 분할

        device = q.device # 텐서가 할당된 디바이스 추출
        range_vec = torch.arange(seq_len, device=device)

        # (i, j) 위치 간의 상대 거리 행렬 생성 (seq_len, seq_len)
        # distance_mat[i][j] = j - i (보통 query 위치에서 key 위치 뺌)
        distance_mat = range_vec[None, :] - range_vec[:, None]

        # 음수 index 방지. 모든 상대 거리 값을 0 ~ (2 * max_len-2)로 클리핑
        # 가운데(0)은 max_len-1에 매핑됨
        distance_mat_clipped = torch.clamp(distance_mat + self.max_len - 1, 0, 2*self.max_len-2)

        # 각 쌍(i, j)의 상대 위치에 해당하는 임베딩 추출
        relative_position_embeddings = self.relative_positions(distance_mat_clipped)

        # 일반 self-attention : 쿼리와 키의 내적을 구해 (batch, h, seq_len, seq_len) 스코어 행렬 생성
        # softmax 전에 d_k로 나눠줌
        scores_content = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 쿼리와 (i, j) 상대 위치 임베딩의 내적. shape: (batch, h, seq_len, seq_len)
        # torch.einsum: 각 쿼리(배치, 헤드, 위치, d_k)와 (i, j) 위치의 상대 임베딩(d_k)를 내적
        # 수식처럼 활용
        scores_relative = torch.einsum('bhid,ijd->bhij', q, relative_position_embeddings) / math.sqrt(self.d_k)
        scores = scores_content + scores_relative

        # 마스크가 주어지면, 마스킹된 부분을 -inf로 만들어 softmax시 무시되게 함
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 마지막 차원(시퀀스 길이, 즉 key/tgt 위치) 기준으로 softmax → 어텐션 확률
        p_attn = scores.softmax(dim=-1)

        # 어텐션 가중치에 dropout 적용 (regularization)
        p_attn = self.dropout(p_attn)

        # 어텐션 가중치와 value의 곱을 계산 (batch, h, seq_len, d_k)
        x = torch.matmul(p_attn, v)

        # 여러 헤드의 출력을 합쳐줌
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 출력 projection 레이어를 통과시켜 반환
        return self.w_o(x)

# 일반 MultiHeadAttention (cross-attention에서 사용)
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn @ value, p_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        batch_size = q.size(0)
        def transform(x, w):
            x = w(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            return x
        q = transform(q, self.w_q)
        k = transform(k, self.w_k)
        v = transform(v, self.w_v)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)

# 잔차 연결
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: callable):
        return x + self.dropout(sublayer(self.norm(x))) # 기존 x에다가 sublayer 거친 output 연결

# 인코더 블록
class EncoderBlock(nn.Module):
    def __init__(self, features: int, mha, ff, dropout: float) -> None:
        super().__init__()
        self.residuals = nn.ModuleList([
            ResidualConnection(features, dropout),
            ResidualConnection(features, dropout)
        ])
        self.mha = mha # 멀티헤드 어텐션
        self.ff = ff # 피드포워드

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.residuals[0](x, lambda t: self.mha(t, t, t, mask)) # 기존이랑 멀티헤드어텐션 잔차연결
        x = self.residuals[1](x, self.ff) # 나온 결과물에다 피드포워드 잔차연결
        return x

# 인코더
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 디코더 블록
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_mha, cross_mha, ff, dropout: float):
        super().__init__()
        self.residuals = nn.ModuleList([
            ResidualConnection(features, dropout),
            ResidualConnection(features, dropout),
            ResidualConnection(features, dropout)
        ])
        self.self_mha = self_mha
        self.cross_mha = cross_mha
        self.ff = ff

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda t: self.self_mha(t, t, t, tgt_mask))
        x = self.residuals[1](x, lambda t: self.cross_mha(t, enc_out, enc_out, src_mask))
        x = self.residuals[2](x, self.ff)
        return x

# 디코더
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)

# 다시 단어 사전 차원으로 Projection
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        return self.proj(x)

# 전체 Transformer
class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        x = self.src_embed(src)
        return self.encoder(x, src_mask)

    def decode(self, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.tgt_embed(tgt)
        return self.decoder(x, enc_out, src_mask, tgt_mask)

    def project(self, x: torch.Tensor):
        return self.projection_layer(x)

# 빌드 함수
def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.2,
                      d_ff: int = 2048):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(d_model,
                     RelativeMultiHeadAttentionBlock(d_model, h, dropout, max_len=src_seq_len),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])

    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model,
                     RelativeMultiHeadAttentionBlock(d_model, h, dropout, max_len=tgt_seq_len),
                     MultiHeadAttentionBlock(d_model, h, dropout),  # cross-attention은 일반 어텐션
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])
    
    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
