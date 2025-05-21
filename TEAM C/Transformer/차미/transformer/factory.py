import torch.nn as nn
from .embedding import InputEmbeddings, PositionalEncoding
from .attention import MultiHeadAttentionBlock
from .feedforward import FeedForwardBlock
from .encoder import Encoder, EncoderBlock
from .decoder import Decoder, DecoderBlock
from .projection import ProjectionLayer
from .model import Transformer

# Transformer 모델을 구성하는 팩토리 함수
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, encoder_self_attn, ff, dropout))

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, decoder_self_attn, decoder_cross_attn, ff, dropout))

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
