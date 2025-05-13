import torch.nn as nn

from .decoder_block import DecoderBlock
from .embedding.input_embedding import InputEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = InputEmbedding(d_model=d_model,
                                  drop_prob=drop_prob,
                                  max_len=max_len,
                                  vocab_size=dec_voc_size,
                                  device=device)

        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg) # 마지막 리니어 거쳐서 나오고 나중에 softmax
        # nn.CrossEntropyLoss 사용 예정
        # 내부적으로 softmax와 log 함수를 적용하기 때문에, 모델의 출력은 raw logits(softmax를 적용하기 전 값)이어야 함
        return output