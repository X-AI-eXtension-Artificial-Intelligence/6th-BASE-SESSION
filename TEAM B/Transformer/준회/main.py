import os
import logging
import torch
import pickle
from torch import nn, optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from huggingface_hub import HfFolder
from models.build_model import build_model

# Hugging Face 인증
hf_token = "hf_..."
os.environ["HUGGINGFACE_TOKEN"] = hf_token
HfFolder.save_token(hf_token)

# ----- 설정 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
MAX_LEN = 256
LR = 5e-4
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----- 로깅 -----
logging.basicConfig(level=logging.INFO)

# ----- 데이터셋 로드 -----
dataset = load_dataset("argilla/news-summary-new", split="train[:100]")

# ----- 토크나이저 및 특수토큰 -----
tokenizer_input = get_tokenizer("basic_english")
tokenizer_target = get_tokenizer("basic_english")
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# ----- 단어 사전 -----
def yield_tokens(data, tokenizer, field):
    for example in data:
        tokens = tokenizer(example[field])
        yield tokens

vocab_src = build_vocab_from_iterator(yield_tokens(dataset, tokenizer_input, "text"),
                                      specials=SPECIALS)
vocab_src.set_default_index(UNK_IDX)

vocab_tgt = build_vocab_from_iterator(yield_tokens(dataset, tokenizer_target, "target"),
                                      specials=SPECIALS)
vocab_tgt.set_default_index(UNK_IDX)

# ----- 텐서 변환 -----
def numericalize(tokens, vocab):
    return [SOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]

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

# ----- 데이터 로더 -----
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ----- 모델 구성 -----
model = build_model(len(vocab_src), len(vocab_tgt), device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ----- 학습 루프 -----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output, _ = model(src, tgt_input)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)

# ----- 저장 -----
with open(os.path.join(CHECKPOINT_DIR, "vocab_src.pkl"), "wb") as f:
    pickle.dump(vocab_src, f)

with open(os.path.join(CHECKPOINT_DIR, "vocab_tgt.pkl"), "wb") as f:
    pickle.dump(vocab_tgt, f)

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pt"))
