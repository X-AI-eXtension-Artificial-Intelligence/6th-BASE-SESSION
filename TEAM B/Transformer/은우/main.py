import json
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256  # 최대 토큰 길이 제한

# Dataset 클래스  

class IMDBDataset(Dataset):
    def __init__(self, file_path, tokenizer, vocab):
        self.samples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = int(item["label"]) - 1  # 1->0, 2->1 # 1,2로 라벨링되어있어서 1뺌 
        tokens = self.tokenizer(item["text"])[:MAX_LEN]  # max length 제한
        token_ids = self.vocab(tokens)
        return torch.tensor(token_ids, dtype=torch.long), label

# Collate 함수 #패
def collate_batch(batch):
    token_ids, labels = zip(*batch)
    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_tokens.to(DEVICE), labels.to(DEVICE)

# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# TransformerClassifier
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, num_classes=2, dropout=0.3): #모델 개수 제한 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) #임베딩
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)  # Dropout after positional encoding
        embedded = embedded.permute(1, 0, 2)
        encoded = self.transformer_encoder(embedded)
        out = self.dropout(encoded.mean(dim=0))  # Dropout before FC
        return self.fc(out)
        
# train 함수 (모델 저장 포함)
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} Train loss: {avg_loss:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), f"transformer_epoch{epoch}.pt")
    print(f"Model saved: transformer_epoch{epoch}.pt")
    return avg_loss

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    acc = correct / total
    return acc

if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")

    # vocab 만들기 위해 train 데이터 토큰화 반복자
    def yield_tokens(file_path):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                yield tokenizer(item["text"])

    vocab = build_vocab_from_iterator(yield_tokens("data/train_data.jsonl"), specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])

    # Dataset / DataLoader 준비
    train_dataset = IMDBDataset("data/train_data.jsonl", tokenizer, vocab)
    valid_dataset = IMDBDataset("data/valid_data.jsonl", tokenizer, vocab)
    test_dataset = IMDBDataset("data/test_data.jsonl", tokenizer, vocab)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = TransformerClassifier(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    NUM_EPOCHS = 5
    for epoch in range(1, NUM_EPOCHS+1):
        train(model, train_loader, optimizer, criterion, epoch)
        valid_acc = evaluate(model, valid_loader)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch} Valid Accuracy: {valid_acc:.4f}")
        print(f"Epoch {epoch} Test Accuracy: {test_acc:.4f}")