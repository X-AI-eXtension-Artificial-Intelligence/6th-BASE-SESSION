import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

'''
1. RMSNorm → LayerNorm 대체

GPT-4, LLaMA에서 사용하는 정규화 기법
계산 효율성이 더 좋고 성능도 우수함
평균 계산 없이 RMS만 사용

2. RoPE (Rotary Position Embedding)

절대 위치 임베딩을 상대적 회전 임베딩으로 대체
긴 시퀀스 외삽 성능이 뛰어남
GPT-NeoX, LLaMA 등에서 검증됨

3. SwiGLU Activation

ReLU 대신 Swish + GLU 조합 사용
PaLM, LLaMA에서 성능 향상 입증
FFN 중간 차원을 8/3배로 확장

4. Grouped Query Attention (GQA)

Key/Value 헤드 수를 줄여 메모리 효율성 향상
LLaMA-2에서 도입된 기법
성능 유지하면서 메모리 사용량 감소

5. Pre-Norm 구조

Post-Norm보다 학습 안정성이 높음
그래디언트 흐름이 더 원활함
'''


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    - GPT-3.5/4, LLaMA 등 최신 모델에서 사용하는 정규화 기법
    - LayerNorm보다 계산 효율적이고 성능이 좋음
    - 평균을 빼지 않고 RMS(Root Mean Square)만 사용
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMS 계산: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function
    - GLU(Gated Linear Unit)의 변형으로 Swish 활성화 함수 사용
    - PaLM, LLaMA 등에서 사용되며 ReLU보다 성능이 우수
    - FFN의 중간 차원을 더 크게 설정 (일반적으로 8/3 * d_model)
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Output projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Value projection

    def forward(self, x):
        # SwiGLU: Swish(W1(x)) ⊙ W3(x), then W2
        # ⊙는 element-wise multiplication
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    - 절대 위치 임베딩 대신 상대적 위치 정보를 회전 행렬로 인코딩
    - GPT-NeoX, LLaMA 등에서 사용
    - 긴 시퀀스에서 외삽 성능이 우수함
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 주파수 계산: θ_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int):
        # x: (batch_size, num_heads, seq_len, head_dim)
        # 위치 인덱스 생성
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # 각 위치와 주파수의 외적: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # cos, sin 값 계산 후 복제하여 전체 차원에 맞춤
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
        
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    쿼리와 키에 회전 위치 임베딩 적용
    """
    def rotate_half(x):
        # 텐서를 반으로 나누고 부호를 바꿔서 회전 효과 구현
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 회전 행렬 적용: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """
    개선된 Multi-Head Attention
    - RoPE(Rotary Position Embedding) 통합
    - Grouped Query Attention(GQA) 옵션 추가
    - Flash Attention 호환 가능한 구조
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # GQA: Key와 Value의 헤드 수를 줄여 메모리 효율성 향상
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_rep = self.n_heads // self.n_kv_heads  # Key/Value 반복 횟수

        # 가중치 행렬들 (bias 없이 설정하는 것이 일반적)
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)  
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """GQA를 위해 Key/Value 텐서를 반복"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * self.n_rep, head_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Query, Key, Value 계산
        q = self.wq(x)  # (batch_size, seq_len, n_heads * head_dim)
        k = self.wk(x)  # (batch_size, seq_len, n_kv_heads * head_dim)  
        v = self.wv(x)  # (batch_size, seq_len, n_kv_heads * head_dim)

        # 헤드별로 분할 및 차원 재배열
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용
        cos, sin = self.rope(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: Key와 Value를 필요한 만큼 반복
        k = self.repeat_kv(k.transpose(1, 2)).transpose(1, 2)
        v = self.repeat_kv(v.transpose(1, 2)).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Causal mask (디코더용)
        if is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention 적용 및 헤드 결합
        out = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(out)

class TransformerBlock(nn.Module):
    """
    개선된 Transformer Block 
    - Pre-Norm 구조 (Post-Norm보다 안정적인 학습)
    - RMSNorm + SwiGLU 조합
    - 잔차 연결(Residual Connection) 최적화
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        # 정규화 레이어들
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        
        # 어텐션 레이어
        self.attention = MultiHeadAttention(
            d_model, n_heads, n_kv_heads, dropout, bias, max_seq_len
        )
        
        # FFN 레이어 (SwiGLU 사용, 중간 차원을 8/3배로 확장)
        ffn_dim = int(8 * d_model / 3)
        ffn_dim = ((ffn_dim + 255) // 256) * 256  # 256의 배수로 패딩 (효율성)
        self.feed_forward = SwiGLU(d_model, ffn_dim, bias)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        # Pre-Norm: 정규화 -> 연산 -> 잔차 연결
        # Self-Attention
        h = x + self.attention(self.attention_norm(x), mask, is_causal)
        
        # Feed Forward Network  
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out

class ModernTransformer(nn.Module):
    """
    현대적인 Transformer 아키텍처
    - Decoder-only 구조 (GPT 스타일)
    - RoPE, RMSNorm, SwiGLU 등 최신 기법 적용
    - Grouped Query Attention으로 메모리 효율성 향상
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12, 
        n_heads: int = 12,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = False,
        tie_weights: bool = True  # 입력과 출력 임베딩 가중치 공유
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 토큰 임베딩 (위치 임베딩은 RoPE로 대체)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer 블록들
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads, 
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                bias=bias,
                max_seq_len=max_seq_len
            ) for _ in range(n_layers)
        ])
        
        # 최종 정규화
        self.norm = RMSNorm(d_model)
        
        # 출력 프로젝션 (언어 모델링 헤드)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 가중치 공유 (메모리 절약 및 성능 향상)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        가중치 초기화
        - GPT-2 스타일 초기화 적용
        - 잔차 연결을 고려한 스케일링
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        순전파
        Args:
            input_ids: 토큰 인덱스 (batch_size, seq_len)
            attention_mask: 어텐션 마스크 (batch_size, seq_len)
            labels: 언어 모델링용 라벨 (batch_size, seq_len)
        Returns:
            logits: 출력 로짓 (batch_size, seq_len, vocab_size)
            loss: 언어 모델링 손실 (옵션)
        """
        batch_size, seq_len = input_ids.shape
        
        # 토큰 임베딩
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # 어텐션 마스크 처리
        if attention_mask is not None:
            # 4D 마스크로 변환: (batch_size, 1, seq_len, seq_len) 
            mask_4d = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
            mask_4d = mask_4d.expand(batch_size, 1, seq_len, seq_len)
        else:
            mask_4d = None
        
        # Transformer 레이어들 통과
        for layer in self.layers:
            x = layer(x, mask_4d, is_causal=True)
        
        # 최종 정규화
        x = self.norm(x)
        
        # 언어 모델링 헤드
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # 손실 계산 (언어 모델링)
        loss = None
        if labels is not None:
            # 다음 토큰 예측을 위해 시프트
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross Entropy Loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        텍스트 생성 (Auto-regressive)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 현재 시퀀스가 최대 길이를 초과하면 자르기
                if input_ids.size(1) >= self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len + 1:]
                
                # 순전파
                logits, _ = self.forward(input_ids)
                
                # 마지막 위치의 로짓만 사용
                logits = logits[:, -1, :] / temperature
                
                # Top-k 필터링
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) 필터링  
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # 샘플링
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 토큰 추가
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 패딩 토큰이면 중단
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
        
        return input_ids

def count_parameters(model):
    """모델의 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 사용 예시
if __name__ == "__main__":
    # 모델 설정 (GPT-2 Small 크기)
    config = {
        'vocab_size': 50257,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'n_kv_heads': 4,  # GQA 사용
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    
    # 모델 생성
    model = ModernTransformer(**config)
    print(f"모델 파라미터 수: {count_parameters(model):,}")
    
    # 더미 데이터로 테스트
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # 순전파 테스트
    with torch.no_grad():
        logits, _ = model(input_ids)
        print(f"출력 크기: {logits.shape}")  # (2, 128, 50257)
    
    # 생성 테스트
    generated = model.generate(
        input_ids[:1, :10],  # 첫 번째 샘플의 처음 10토큰만 사용
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )
    print(f"생성된 시퀀스 길이: {generated.shape[1]}")