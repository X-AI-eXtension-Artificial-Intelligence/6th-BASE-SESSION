import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 100  # 최대 문장 길이 제한
BATCH_SIZE = 16

# 1) 데이터셋 클래스
class TranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
        self.samples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        src_tokens = self.src_tokenizer(item["src"])[:MAX_LEN]
        tgt_tokens = self.tgt_tokenizer(item["tgt"])[:MAX_LEN-1]  # 디코더 입력에는 EOS 제외
        src_ids = [self.src_vocab[token] for token in src_tokens]
        tgt_ids = [self.tgt_vocab[token] for token in tgt_tokens]

        
        tgt_input = [self.tgt_vocab["<bos>"]] + tgt_ids
        
        tgt_output = tgt_ids + [self.tgt_vocab["<eos>"]]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_input, dtype=torch.long), torch.tensor(tgt_output, dtype=torch.long)

# 2) collate 함수
def collate_batch(batch):
    src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_input_batch = pad_sequence(tgt_input_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])
    tgt_output_batch = pad_sequence(tgt_output_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])

    return src_batch.to(DEVICE), tgt_input_batch.to(DEVICE), tgt_output_batch.to(DEVICE)

# 3) PositionalEncoding (Transformer의 위치 인코딩)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 4) Transformer 번역기 모델 (인코더-디코더)
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # src, tgt shape: (batch, seq_len)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Transformer expects (seq_len, batch, d_model)
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = output.permute(1, 0, 2)  # (batch, seq_len, d_model)

        return self.fc_out(output)

# 5) 마스크 생성 함수
def generate_square_subsequent_mask(sz): #디코더 입력용 마스크
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask.to(DEVICE)

# 6) 패딩 마스크 생성 함수 (batch_size x seq_len) #패딩 위치 트루로 마스
def create_padding_mask(seq, pad_idx):
    return (seq == pad_idx)

# 7) 학습 함수
def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0

    for src, tgt_input, tgt_output in dataloader:
        optimizer.zero_grad()
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
        src_padding_mask = create_padding_mask(src, src_vocab["<pad>"])
        tgt_padding_mask = create_padding_mask(tgt_input, tgt_vocab["<pad>"])

        output = model(src, tgt_input, tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        tgt_output = tgt_output.view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 8) 평가 함수 
def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)) #디코더 입력용 
            src_padding_mask = create_padding_mask(src, src_vocab["<pad>"])
            tgt_padding_mask = create_padding_mask(tgt_input, tgt_vocab["<pad>"])

            output = model(src, tgt_input, tgt_mask=tgt_mask,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=src_padding_mask)

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# 9) 토큰 생성 예시 함수 (Greedy decoding)
def translate_sentence(model, sentence, src_tokenizer, tgt_vocab, max_len=MAX_LEN):
    model.eval()
    tokens = src_tokenizer(sentence)[:MAX_LEN]
    src_ids = torch.tensor([src_vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_padding_mask = create_padding_mask(src_ids, src_vocab["<pad>"])

    memory = model.encoder(model.pos_encoder(model.src_embedding(src_ids) * math.sqrt(model.d_model)).permute(1,0,2),
                           src_key_padding_mask=src_padding_mask)

    ys = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long).to(DEVICE)

    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1))
        out = model.decoder(model.pos_decoder(model.tgt_embedding(ys) * math.sqrt(model.d_model)).permute(1,0,2),
                            memory,
                            tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_padding_mask)
        out = out.permute(1,0,2)
        prob = model.fc_out(out[:, -1, :])
        next_word = prob.argmax(1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        if next_word == tgt_vocab["<eos>"]:
            break

    tgt_tokens = [tgt_vocab.get_itos()[i] for i in ys.squeeze().tolist()]
    return tgt_tokens[1:]  # <bos> 제외


# ======================= main ==========================

if __name__ == "__main__":
    # 토크나이저
    src_tokenizer = get_tokenizer("basic_english")
    tgt_tokenizer = get_tokenizer("basic_english")

    # vocab 빌드
    def yield_tokens(file_path, tokenizer, lang_key):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                yield tokenizer(item[lang_key])

    # train 데이터 경로
    train_path = "data/train_data.jsonl"  # {"src": "english sentence", "tgt": "german sentence"}

    # src_vocab, tgt_vocab 생성, special tokens 포함
    src_vocab = build_vocab_from_iterator(yield_tokens(train_path, src_tokenizer, "src"), specials=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = build_vocab_from_iterator(yield_tokens(train_path, tgt_tokenizer, "tgt"), specials=["<pad>", "<bos>", "<eos>"])

    src_vocab.set_default_index(src_vocab["<pad>"])
    tgt_vocab.set_default_index(tgt_vocab["<pad>"])

    # 데이터셋 / DataLoader
    train_dataset = TranslationDataset(train_path, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab)
    valid_dataset = TranslationDataset("data/valid_data.jsonl", src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 모델 생성
    model = TransformerModel(len(src_vocab), len(tgt_vocab)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])

    NUM_EPOCHS = 3
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        valid_loss = evaluate(model, criterion, valid_loader)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}")

         # 모델 저장
        save_path = f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")