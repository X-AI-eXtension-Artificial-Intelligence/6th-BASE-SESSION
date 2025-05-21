import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    입력 특징별로 평균-분산 정규화 수행하는 Layer Normalization 구현
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # 스케일링 파라미터 (alpha)
        self.alpha = nn.Parameter(torch.ones(features))
        # 시프트 파라미터 (bias)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        # 마지막 차원(feature) 기준으로 평균과 표준편차 계산
        mean = x.mean(dim=-1, keepdim=True)   # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)     # (batch, seq_len, 1)
        # 정규화 후 스케일 및 시프트 적용
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    위치별 FFN (Position-wise Feed-Forward Network) 블록
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # 첫 번째 선형 변환 (d_model -> d_ff)
        self.linear_1 = nn.Linear(d_model, d_ff)
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        # 두 번째 선형 변환 (d_ff -> d_model)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU 활성화 및 드롭아웃 후 투영
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):
    """
    토큰 임베딩 + 스케일링 (sqrt(d_model))
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, d_model)
        # 논문 권장대로 d_model의 루트 스케일링 적용
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    사인/코사인 기반 위치 인코딩
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 위치 인코딩 행렬 생성 (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 짝수 인덱스: sin, 홀수 인덱스: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 배치 차원 추가 -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # 파라미터로 학습되지 않도록 buffer로 등록
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력에 위치 인코딩을 더하고 드롭아웃
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    잔차 연결 + Layer Normalization + Dropout 합성 블록
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # x + Dropout(sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    멀티-헤드 셀프 어텐션 블록
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model이 헤드 개수로 나누어지지 않음"
        self.d_k = d_model // h  # 각 헤드의 차원
        self.h = h
        # 쿼리, 키, 값, 출력 투영 레이어
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        # Scaled Dot-Product Attention
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            # 마스크 위치에 -inf 대입
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # 최종 값: attention * value
        return p_attn @ value, p_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        # 선형 투영 후 헤드 분할
        batch_size = q.size(0)
        def transform(x, w):
            x = w(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            return x

        q = transform(q, self.w_q)
        k = transform(k, self.w_k)
        v = transform(v, self.w_v)

        # 어텐션 계산
        x, self.attention_scores = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)
        # 모든 헤드 결합
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    """
    Transformer 인코더의 한 레이어 블록 (셀프 어텐션 + FFN)
    """

    def __init__(self, features: int, mha: MultiHeadAttentionBlock, ff: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        # 두 개의 잔차 연결 블록
        self.residuals = nn.ModuleList([
            ResidualConnection(features, dropout),  # 셀프 어텐션
            ResidualConnection(features, dropout)   # FFN
        ])
        self.mha = mha
        self.ff = ff

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 셀프 어텐션
        x = self.residuals[0](x, lambda t: self.mha(t, t, t, mask))
        # FFN
        x = self.residuals[1](x, self.ff)
        return x


class Encoder(nn.Module):
    """
    N개의 EncoderBlock 쌓아 올린 인코더
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # 최종 레이어 정규화
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Transformer 디코더의 한 레이어 블록
    (셀프 어텐션 + 크로스 어텐션 + FFN)
    """

    def __init__(self, features: int, self_mha: MultiHeadAttentionBlock, cross_mha: MultiHeadAttentionBlock, ff: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.residuals = nn.ModuleList([
            ResidualConnection(features, dropout),  # 디코더 셀프 어텐션
            ResidualConnection(features, dropout),  # 인코더-디코더 크로스 어텐션
            ResidualConnection(features, dropout)   # FFN
        ])
        self.self_mha = self_mha
        self.cross_mha = cross_mha
        self.ff = ff

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda t: self.self_mha(t, t, t, tgt_mask))
        x = self.residuals[1](x, lambda t: self.cross_mha(t, enc_out, enc_out, src_mask))
        x = self.residuals[2](x, self.ff)
        return x


class Decoder(nn.Module):
    """
    N개의 DecoderBlock 쌓아 올린 디코더
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    디코더 출력의 최종 단어 분포 투영
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Transformer(nn.Module):
    """
    전체 Transformer 모델 (인코더 + 디코더 + 임베딩 + 포지셔널 인코딩)
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # 인코더 경로: 임베딩 -> 위치 인코딩 -> 인코더 블록
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # 디코더 경로: 임베딩 -> 위치 인코딩 -> 디코더 블록
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        return self.decoder(x, enc_out, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # 최종 출력: 단어 분포 생성
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    """
    Transformer 모델 구성 함수
    :param src_vocab_size: 소스 언어 어휘 수
    :param tgt_vocab_size: 타겟 언어 어휘 수
    :param src_seq_len: 소스 시퀀스 최대 길이
    :param tgt_seq_len: 타겟 시퀀스 최대 길이
    :param d_model: 모델 차원 (임베딩 크기)
    :param N: 인코더/디코더 레이어 수
    :param h: 헤드 수
    :param dropout: 드롭아웃 비율
    :param d_ff: FFN 내부 차원
    :return: 초기화된 Transformer 모델
    """
    # 임베딩 레이어 생성
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # 위치 인코딩 생성
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 인코더 블록 생성
    encoder_blocks = nn.ModuleList([
        EncoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])
    # 디코더 블록 생성
    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])

    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # 가중치 초기화 (Xavier)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
