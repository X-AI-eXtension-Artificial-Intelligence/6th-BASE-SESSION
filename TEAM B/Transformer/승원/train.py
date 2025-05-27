"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import torch

from torch import nn, optim
from torch.optim import Adam
from data import train_loader, src_vocab, trg_vocab  # 변경된 위치로부터 직접 가져옴
from models.model.transformer import Transformer
from util.bleu import get_bleu
from util.epoch_timer import epoch_time

# 하이퍼파라미터 정의
clip = 1
init_lr = 1e-3
weight_decay = 1e-5
adam_eps = 1e-9
factor = 0.5
patience = 5
warmup = 5
epoch = 50
inf = float('inf')
d_model = 128
ffn_hidden = 512
n_heads = 8
n_layers = 2
drop_prob = 0.1
max_len = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_pad_idx = src_vocab['<pad>']
trg_pad_idx = trg_vocab['<pad>']
trg_sos_idx = trg_vocab['<sos>']
enc_voc_size = len(src_vocab)
dec_voc_size = len(trg_vocab)

# 모델 정의
model = Transformer(src_pad_idx=src_pad_idx,
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
                    device=device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(lambda m: nn.init.kaiming_uniform_(m.weight.data) if hasattr(m, 'weight') and m.weight.dim() > 1 else None)

optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=factor, patience=patience)
#criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)  # 기존 BLEU: 46.826
#라벨 스무딩 기법(정답이 아닌 클래스를 0으로 예측하지 않고 정답을 0.9 정도로 예측)을 활용 BLEU:47.727 
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1) 


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()

            for j in range(src.size(0)):
                try:
                    pred_ids = output[j].argmax(dim=1).tolist()
                    trg_ids = trg[j][1:].tolist()
                    
                    pred_tokens = [k for idx in pred_ids if idx in trg_vocab.values() for k, v in trg_vocab.items() if v == idx]
                    trg_tokens = [k for idx in trg_ids if idx in trg_vocab.values() for k, v in trg_vocab.items() if v == idx]
                    
                    bleu = get_bleu(hypotheses=pred_tokens, reference=trg_tokens)
                    batch_bleu.append(bleu)
                except:
                    continue

    avg_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / len(iterator), avg_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, train_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.3f}.pt')

        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
    import pickle

    with open('saved/src_vocab.pkl', 'wb') as f:
        pickle.dump(src_vocab, f)

    with open('saved/trg_vocab.pkl', 'wb') as f:
        pickle.dump(trg_vocab, f)

'''

import math
import time

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import train_loader, valid_loader, src_vocab, trg_vocab  # 새 데이터 불러오기 방식 반영
from util.bleu import get_bleu
from util.epoch_timer import epoch_time

# 모델 정의 (원하는 트랜스포머 모델로 교체 가능)
from models.model.transformer import Transformer

# 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip = 1
init_lr = 1e-3
weight_decay = 1e-5
adam_eps = 1e-9
factor = 0.5
patience = 5
warmup = 5
batch_size = 2
epoch = 10
inf = float('inf')

# vocab index 지정
src_pad_idx = src_vocab['<pad>']
trg_pad_idx = trg_vocab['<pad>']
trg_sos_idx = trg_vocab['<sos>']

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=128,
                    enc_voc_size=len(src_vocab),
                    dec_voc_size=len(trg_vocab),
                    max_len=100,
                    ffn_hidden=512,
                    n_head=8,
                    n_layers=2,
                    drop_prob=0.1,
                    device=device).to(device)

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=factor, patience=patience)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

            for j in range(src.size(0)):
                try:
                    pred_idx = output[j].max(dim=1)[1].tolist()
                    target_idx = trg[j][1:].tolist()

                    pred_tokens = [k for k, v in trg_vocab.items() if v in pred_idx]
                    target_tokens = [k for k, v in trg_vocab.items() if v in target_idx]

                    bleu = get_bleu(hypotheses=pred_tokens, reference=target_tokens)
                    batch_bleu.append(bleu)
                except:
                    continue

    return epoch_loss / len(iterator), sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.3f}.pt')

        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
'''