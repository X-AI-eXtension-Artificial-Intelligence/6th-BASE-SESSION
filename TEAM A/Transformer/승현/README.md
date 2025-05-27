# Transformer Implementation

This repository contains a PyTorch implementation of the Transformer architecture as described in the paper "Attention Is All You Need" (Vaswani et al., 2017).

## Architecture

The implementation includes all the key components of the Transformer:

1. **Multi-Head Attention**
   - Scaled Dot-Product Attention
   - Multi-Head Attention mechanism

2. **Position-wise Feed-Forward Networks**
   - Two linear transformations with a ReLU activation in between

3. **Positional Encoding**
   - Sinusoidal positional encodings

4. **Encoder and Decoder Layers**
   - Encoder: Self-attention + Feed-forward
   - Decoder: Self-attention + Cross-attention + Feed-forward

5. **Complete Transformer Model**
   - Stacked encoder and decoder layers
   - Input/output embeddings
   - Final linear layer

## Files

- `attention.py`: Implementation of attention mechanisms
- `feed_forward.py`: Position-wise feed-forward networks and positional encoding
- `transformer_layers.py`: Encoder and decoder layer implementations
- `transformer.py`: Complete Transformer model
- `example.py`: Example usage of the Transformer model

## Requirements

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

## Usage

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the example script:
```bash
python example.py
```

## Model Parameters

The Transformer model can be configured with the following parameters:

- `src_vocab_size`: Size of the source vocabulary
- `tgt_vocab_size`: Size of the target vocabulary
- `d_model`: Model dimension (default: 512)
- `num_heads`: Number of attention heads (default: 8)
- `num_encoder_layers`: Number of encoder layers (default: 6)
- `num_decoder_layers`: Number of decoder layers (default: 6)
- `d_ff`: Feed-forward dimension (default: 2048)
- `max_seq_length`: Maximum sequence length (default: 5000)
- `dropout`: Dropout rate (default: 0.1)

## Example

```python
from transformer import Transformer

# Create model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Forward pass
output = model(src, tgt)
```

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

---

# Transformer 구현 요약 (Implementation Summary in Korean)

### 1. 전체 구조
Transformer 모델은 크게 다음과 같은 파일들로 구성되어 있습니다:

1. `attention.py`: 어텐션 메커니즘 구현
2. `feed_forward.py`: 피드포워드 네트워크와 위치 인코딩 구현
3. `transformer_layers.py`: 인코더와 디코더 레이어 구현
4. `transformer.py`: 전체 Transformer 모델 구현
5. `example.py`: 사용 예제
6. `model_diagram.py`: 모델 구조 시각화

### 2. 주요 컴포넌트별 구현

#### 2.1 어텐션 메커니즘 (`attention.py`)
```python
- ScaledDotProductAttention 클래스
  - 쿼리(Q), 키(K), 값(V)를 입력받아 어텐션 계산
  - 마스킹 지원
  - 스케일링된 내적 연산 수행

- MultiHeadAttention 클래스
  - 여러 개의 어텐션 헤드로 나누어 병렬 처리
  - 각 헤드별로 Q, K, V를 독립적으로 처리
  - 최종 결과를 결합하여 출력
```

#### 2.2 피드포워드 네트워크와 위치 인코딩 (`feed_forward.py`)
```python
- PositionWiseFeedForward 클래스
  - 두 개의 선형 변환과 ReLU 활성화 함수 사용
  - 드롭아웃 적용

- PositionalEncoding 클래스
  - 사인과 코사인 함수를 사용한 위치 정보 인코딩
  - 시퀀스의 위치 정보를 임베딩에 추가
```

#### 2.3 인코더와 디코더 레이어 (`transformer_layers.py`)
```python
- EncoderLayer 클래스
  - 셀프 어텐션
  - 피드포워드 네트워크
  - 레이어 정규화
  - 잔차 연결

- DecoderLayer 클래스
  - 셀프 어텐션
  - 크로스 어텐션
  - 피드포워드 네트워크
  - 레이어 정규화
  - 잔차 연결
```

#### 2.4 전체 Transformer 모델 (`transformer.py`)
```python
- Transformer 클래스
  - 입력/출력 임베딩
  - 위치 인코딩
  - 인코더 스택 (6개 레이어)
  - 디코더 스택 (6개 레이어)
  - 출력 레이어
  - 마스킹 처리
```

### 3. 데이터 흐름

1. **입력 처리**
   - 입력 토큰 → 임베딩 → 위치 인코딩 추가

2. **인코더 처리**
   - 셀프 어텐션으로 입력 시퀀스 내 관계 학습
   - 피드포워드 네트워크로 변환
   - 레이어 정규화와 잔차 연결

3. **디코더 처리**
   - 셀프 어텐션으로 출력 시퀀스 내 관계 학습
   - 크로스 어텐션으로 인코더 출력과 연결
   - 피드포워드 네트워크로 변환
   - 레이어 정규화와 잔차 연결

4. **출력 처리**
   - 최종 선형 레이어를 통한 출력 생성

### 4. 시각화 (`model_diagram.py`)
- matplotlib을 사용하여 모델 구조를 시각화
- 각 컴포넌트를 다른 색상으로 구분
- 데이터 흐름을 화살표로 표시
- 잔차 연결을 점선으로 표시

### 5. 사용 예제 (`example.py`)
```python
- 모델 초기화
- 더미 데이터 생성
- 순전파 수행
- 손실 함수 계산
- 역전파 및 최적화
```

---
[Added 25.05.19]

## 데이터셋 처리 방식 비교: Dummy vs WikiText2

### 1. Dummy 데이터셋 방식
- **구현 목적**: 모델 구조 및 학습 파이프라인의 정상 동작 확인, 빠른 테스트용.
- **데이터 생성**: 무작위로 생성된 정수 시퀀스를 입력/타겟 쌍으로 사용.
- **어휘 사전**: 인위적으로 지정한 vocab_size만큼의 토큰 인덱스 사용.
- **장점**: 외부 의존성 없이 빠르게 테스트 가능, 데이터 준비가 매우 간단함.
- **단점**: 실제 자연어 데이터가 아니므로 모델의 실제 성능이나 일반화 능력 평가 불가.

### 2. WikiText2 데이터셋 방식
- **구현 목적**: 실제 자연어 데이터 기반의 언어모델 학습 및 평가.
- **데이터 생성**: 위키피디아 문서로 구성된 WikiText2 코퍼스를 다운로드하여 토큰화 후 시퀀스 생성.
- **어휘 사전**: 실제 텍스트에서 등장한 단어로부터 동적으로 vocab을 구축.
- **장점**: 실제 문장 구조와 어휘를 반영하므로 모델의 언어 이해 및 생성 능력 평가 가능.
- **단점**: 데이터 다운로드 및 전처리 필요, 어휘 사전 구축 등 준비 과정이 더 복잡함.

### 3. 코드 구조상 차이
- **Dummy**: `DummyDataset` 클래스를 통해 무작위 시퀀스 생성, 별도의 텍스트 파일이나 토크나이저 불필요.
- **WikiText2**: 실제 텍스트 파일을 다운로드 및 로드, 정규표현식 기반 토큰화, 동적 vocab 생성, 텍스트를 인덱스 시퀀스로 변환하는 과정 필요.

> **요약**: Dummy 데이터는 빠른 구조 테스트용, WikiText2는 실제 자연어 처리 성능 평가용으로 사용됩니다.

---
[Updated 25.05.26]
Position Encoding 방식이 Relative Positional Encoding으로 변경되었습니다
다만....아직 구현 중 입니다...

## Relative Positional Encoding 구현 가이드

### 1. 기존 Sinusoidal Positional Encoding의 한계
- 고정된 위치 정보만 제공
- 시퀀스 길이가 길어질수록 성능 저하
- 상대적 위치 관계를 직접적으로 모델링하지 못함

### 2. Relative Positional Encoding 구현 단계

#### 2.1 새로운 클래스 구현
```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 상대적 위치 임베딩 행렬 초기화
        self.relative_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, d_model)
        )
        
    def forward(self, length):
        # 상대적 위치 인덱스 생성
        range_vec = torch.arange(length)
        relative_positions = range_vec[None, :] - range_vec[:, None]
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        relative_positions += self.max_relative_position
        
        # 상대적 위치 임베딩 조회
        embeddings = self.relative_embeddings[relative_positions]
        return embeddings
```

#### 2.2 ScaledDotProductAttention 수정
```python
def forward(self, query, key, value, relative_embeddings=None):
    d_k = query.size(-1)
    
    # 기본 어텐션 스코어 계산
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 상대적 위치 정보 추가
    if relative_embeddings is not None:
        relative_scores = torch.matmul(query, relative_embeddings.transpose(-2, -1))
        scores = scores + relative_scores
    
    # 나머지 처리 (마스킹, 소프트맥스 등)
    ...
```

#### 2.3 MultiHeadAttention 수정
```python
def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    
    # 기존 Q, K, V 변환
    Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    # 상대적 위치 임베딩 생성
    seq_length = query.size(1)
    relative_embeddings = self.relative_encoding(seq_length)
    
    # 수정된 어텐션 적용
    output, attention_weights = self.attention(Q, K, V, relative_embeddings, mask)
    ...
```

### 3. 주요 변경사항

1. **위치 인코딩 방식 변경**
   - 고정된 사인/코사인 함수 대신 학습 가능한 상대적 위치 임베딩 사용
   - 최대 상대적 위치 거리 제한 (max_relative_position)

2. **어텐션 계산 수정**
   - 기본 어텐션 스코어에 상대적 위치 정보 추가
   - 쿼리와 상대적 위치 임베딩 간의 내적 계산

3. **파라미터 추가**
   - 상대적 위치 임베딩 행렬 (학습 가능한 파라미터)
   - 최대 상대적 위치 거리 설정

### 4. 장점

1. **더 나은 위치 관계 모델링**
   - 토큰 간의 상대적 거리를 직접 학습
   - 긴 시퀀스에서도 효과적인 위치 정보 제공

2. **유연한 위치 정보**
   - 학습 가능한 파라미터로 인해 데이터에 맞게 최적화
   - 고정된 패턴이 아닌 동적인 위치 관계 학습

3. **확장성**
   - 다양한 길이의 시퀀스에 대응 가능
   - 최대 상대적 위치 거리를 조절하여 메모리 사용량 제어

### 5. 구현 시 주의사항

1. **메모리 사용량**
   - 상대적 위치 임베딩 행렬의 크기 관리
   - max_relative_position 값의 적절한 설정

2. **학습 안정성**
   - 상대적 위치 임베딩의 초기화 방식
   - 학습률 조정 필요

3. **성능 최적화**
   - 배치 처리 시 효율적인 계산
   - 캐싱을 통한 반복 계산 방지
