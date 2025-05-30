
# main_predict.py
# 학습된 모델을 불러와 Beam Search로 번역 수행

import torch
import torch.nn.functional as F
import os
from model import Transformer
import sentencepiece as spm

# -----------------------------
# Tokenizer (동일)
# -----------------------------
class Tokenizer:
    def __init__(self, model_path="spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, add_special_tokens=True):
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def pad_id(self):
        return self.sp.pad_id()

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

# -----------------------------
# 설정
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 128
tokenizer = Tokenizer("spm.model")
pad_id = tokenizer.pad_id()
sos_id = tokenizer.bos_id()
eos_id = tokenizer.eos_id()
vocab_size = tokenizer.sp.get_piece_size()

model = Transformer(
    src_pad_idx=pad_id,
    trg_pad_idx=pad_id,
    trg_sos_idx=sos_id,
    d_model=256,
    enc_voc_size=vocab_size,
    dec_voc_size=vocab_size,
    max_len=max_len,
    ffn_hidden=1024,
    n_head=8,
    n_layers=3,
    drop_prob=0.1,
    device=device,
    share_decoder_embedding_output=True
).to(device)

# 최신 모델 불러오기
def load_latest_model():
    model_files = [os.path.join("saved", f) for f in os.listdir("saved") if f.endswith(".pt")]
    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    print(f"📦 Loading latest model: {latest_model}")
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()

# -----------------------------
# Beam Search
# -----------------------------
@torch.no_grad()
def beam_search_translate(input_text, beam_width=10):
    input_ids = tokenizer.encode(input_text)[:max_len]
    input_ids += [pad_id] * (max_len - len(input_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    enc_out = model.encoder(input_tensor, model.make_src_mask(input_tensor))
    beams = [([sos_id], 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                new_beams.append((seq, score))
                continue
            trg_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            dec_out = model.decoder(trg_tensor, enc_out, trg_mask, model.make_src_mask(input_tensor))
            logits = dec_out[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk_probs, topk_indices = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                new_seq = seq + [topk_indices[i].item()]
                new_score = score + topk_probs[i].item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == eos_id for seq, _ in beams):
            break

    return tokenizer.decode(beams[0][0])

# -----------------------------
# 실행
# -----------------------------
if __name__ == '__main__':
    load_latest_model()
    text = "이 기술은 미래에 많은 산업에 영향을 미칠 것이다."
    translated = beam_search_translate(text, beam_width=10)
    print("
[Source Sentence]", text)
    print("[Translated Sentence]", translated)
