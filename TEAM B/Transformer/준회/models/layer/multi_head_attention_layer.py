import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed)
        self.dropout = nn.Dropout(p=dr_rate)

    ## Scaled Dot-Product Attention 계산
    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, seq_len, seq_len)
        d_k = key.shape[-1]

        ## Q와 K(전치) 행렬곱하고 정규화
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        
        ## 마스크가 있다면 마스킹된 부분을 매우 -1e9으로 해서 무시
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        
        ## Softmax로 어텐션 확률 분포 계산
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        
        ## 어텐션 가중치와 V 행렬곱해서 아웃풋 생성
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out


    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        ## 입력 벡터를 FC Layer를 통해 Multi-Head 형태로 변환
        def transform(x, fc): # (n_batch, seq_len, d_embed)
            out = fc(x)       # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        ## Query, Key, Value를 Multi-head 형태로 변환
        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)       # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        ## Scaled Dot-Product Attention 수행
        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        
        ## 어텐션 결과를 원래 형태로 변환
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out