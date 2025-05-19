import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from models.build_model import build_model
from torchtext.data.metrics import bleu_score
import pickle

# ----- 설정 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 64
CHECKPOINT_PATH = "./checkpoints/final_model.pt"
VOCAB_SRC_PATH = "./checkpoints/vocab_src.pkl"
VOCAB_TGT_PATH = "./checkpoints/vocab_tgt.pkl"
SOS_IDX, EOS_IDX, PAD_IDX = 2, 3, 1

# ----- 토크나이저 -----
tokenizer_input = get_tokenizer("basic_english")
tokenizer_target = get_tokenizer("basic_english")

# ----- 사전 불러오기 -----
def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

vocab_src = load_vocab(VOCAB_SRC_PATH)
vocab_tgt = load_vocab(VOCAB_TGT_PATH)

# ----- 수치화 -----
def numericalize(tokens, vocab):
    return [SOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]

# ----- collate_fn -----
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for example in batch:
        src_tokens = tokenizer_input(example["text"])
        tgt_tokens = tokenizer_target(example["target"])
        src_tensor = torch.tensor(numericalize(src_tokens, vocab_src), dtype=torch.long)
        tgt_tensor = torch.tensor(numericalize(tgt_tokens, vocab_tgt), dtype=torch.long)
        src_batch.append(src_tensor[:MAX_LEN])
        tgt_batch.append(tgt_tensor[:MAX_LEN])
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch.transpose(0, 1), tgt_batch.transpose(0, 1)

# ----- 데이터 로드 -----
val_dataset = load_dataset("argilla/news-summary-new", split="train[100:]")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ----- 모델 로드 -----
model = build_model(len(vocab_src), len(vocab_tgt), device=DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ----- Loss & BLEU -----
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
total_loss = 0
all_preds, all_refs = [], []

with torch.no_grad():
    for src, tgt in val_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output, _ = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        total_loss += loss.item()

        preds = output.argmax(dim=-1)
        for p, t in zip(preds, tgt_output):
            pred_tok = [vocab_tgt.lookup_token(i) for i in p if i not in {PAD_IDX, SOS_IDX, EOS_IDX}]
            tgt_tok = [vocab_tgt.lookup_token(i) for i in t if i not in {PAD_IDX, SOS_IDX, EOS_IDX}]
            all_preds.append(pred_tok)
            all_refs.append([tgt_tok])  # list of reference lists

bleu = bleu_score(all_preds, all_refs) * 100
print(f"🔍 Evaluation Results:\nAverage Loss: {total_loss / len(val_loader):.4f}, BLEU Score: {bleu:.2f}")
