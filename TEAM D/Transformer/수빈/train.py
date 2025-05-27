import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word
from util.epoch_timer import epoch_time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

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

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)

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

    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_gold = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_gold)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(src.size(0)):
                try:
                    trg_seq = trg[j, 1:]
                    trg_words = idx_to_word(trg_seq, loader.target.vocab).split()

                    pred_ids = output[j].argmax(dim=1)
                    pred_words = idx_to_word(pred_ids, loader.target.vocab).split()

                    if '<eos>' in pred_words:
                        pred_words = pred_words[:pred_words.index('<eos>')]

                    bleu = sentence_bleu([trg_words], pred_words, smoothing_function=smoothie)
                    total_bleu.append(bleu)
                except:
                    total_bleu.append(0.0)

            avg_bleu = sum(total_bleu) / len(total_bleu) if total_bleu else 0.0
            batch_bleu.append(avg_bleu)

    final_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    return epoch_loss / len(iterator), final_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
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
        with open('result/bleu.txt', 'w') as f:
            f.write(str(bleus))
        with open('result/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)