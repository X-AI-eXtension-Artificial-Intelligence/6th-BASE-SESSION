import torch
import torch.nn as nn
import math

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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