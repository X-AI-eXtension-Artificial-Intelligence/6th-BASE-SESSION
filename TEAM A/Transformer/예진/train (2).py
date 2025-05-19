import math
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import optim
from tqdm import tqdm

from util.data_loader import DataLoaderWrapper
from models.model.transformer import Transformer
from util.epoch_timer import epoch_time
from rouge_score import rouge_scorer
import re

# 하이퍼파라미터
batch_size = 4
max_len = 128
d_model = 256
n_heads = 8
n_layers = 3
ffn_hidden = 1024
drop_prob = 0.1
init_lr = 1e-4
weight_decay = 1e-5
adam_eps = 1e-9
factor = 0.5
patience = 2
clip = 1.0
epoch = 5
warmup = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inf = float('inf')

# 데이터셋 로딩
loader = DataLoaderWrapper(max_len=max_len)
train_dataset = loader.make_dataset(split='train')
valid_dataset = loader.make_dataset(split='validation')
train_iter = loader.make_iter(train_dataset, batch_size=batch_size)
valid_iter = loader.make_iter(valid_dataset, batch_size=batch_size)

token2id = loader.token2id
id2token = loader.id2token

src_pad_idx = token2id['<pad>']
trg_pad_idx = token2id['<pad>']
trg_sos_idx = token2id['<sos>']
enc_voc_size = len(token2id)
dec_voc_size = len(token2id)

# 모델 정의
model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=d_model,
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=max_len,
    ffn_hidden=ffn_hidden,
    n_head=n_heads,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device
).to(device)

# 파라미터 관련 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

# 최적화 도구
optimizer = Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# ID → 토큰 디코딩 함수
def decode(ids):
    return ' '.join([id2token[i] for i in ids if i not in {src_pad_idx, trg_sos_idx, token2id['<eos>']}])

# 학습 루프
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, decoder_input_ids)
        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)

        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 검증 + ROUGE 평가
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            # 예측 결과
            output = model(input_ids, decoder_input_ids)
            predictions = output.argmax(dim=-1)

            # ROUGE 계산
            for pred_ids, label_ids in zip(predictions, labels):
                pred = decode(pred_ids.tolist())
                ref = decode(label_ids.tolist())
                score = scorer.score(ref, pred)
                rouge_scores.append(score)

            # 손실 계산
            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

    # ROUGE 평균
    avg_scores = {
        'rouge1': sum(s['rouge1'].fmeasure for s in rouge_scores) / len(rouge_scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in rouge_scores) / len(rouge_scores),
        'rougeL': sum(s['rougeL'].fmeasure for s in rouge_scores) / len(rouge_scores)
    }

    return epoch_loss / len(iterator), avg_scores

# 전체 학습 루프
def run(total_epoch, best_loss):
    train_losses, test_losses = [], []

    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, rouge = evaluate(model, valid_iter, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pt')

        with open('result/train_loss.txt', 'w') as f: f.write(str(train_losses))
        with open('result/test_loss.txt', 'w') as f: f.write(str(test_losses))

        print(f'Epoch: {step + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f}   | Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tROUGE-1: {rouge["rouge1"]:.4f} | ROUGE-2: {rouge["rouge2"]:.4f} | ROUGE-L: {rouge["rougeL"]:.4f}')

# 실행
if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)

# 학습 끝난 후에 저장 (train.py 마지막에)
import pickle

with open("vocab.pkl", "wb") as f:
    pickle.dump((loader.token2id, loader.id2token), f)
