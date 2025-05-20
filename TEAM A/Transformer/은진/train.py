import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
from data import (
    train_loader, valid_loader, test_loader,
    src_pad_idx, trg_pad_idx, src_sos_idx, trg_sos_idx,
    enc_voc_size, dec_voc_size, vocab_en, vocab_de
)
from models.model import Transformer
from util import idx_to_word, get_bleu_nltk, epoch_time
from conf import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

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

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

optimizer = Adam(
    params=model.parameters(),
    lr=init_lr,
    weight_decay=weight_decay,
    eps=adam_eps
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    verbose=True,
    factor=factor,
    patience=patience
)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_gold = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, trg_gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 10 == 0:
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
            trg_gold = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg_gold)
            epoch_loss += loss.item()
            # BLEU 계산 (batch 단위)
            total_bleu = []
            for j in range(src.size(0)):  # batch_size
                try:
                    trg_words = idx_to_word(trg[j].tolist(), vocab_en)
                    output_words = output[j].max(dim=1)[1].tolist()
                    output_words = idx_to_word(output_words, vocab_en)
                    bleu = get_bleu_nltk(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception as e:
                    pass
            if total_bleu:
                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)
    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    return epoch_loss / len(iterator), batch_bleu

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
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pt')

        with open('result/train_loss.txt', 'w') as f:
            f.write(str(train_losses))
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=float('inf'))
