import os
import math
from torch import nn, optim
from torch.optim import Adam
from nltk.translate.bleu_score import sentence_bleu
from util.data_config import *
from models.transformer import Transformer
from tqdm import tqdm

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


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

model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

loss_func = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, train_loader, optimizer, loss_func, clip):
    model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='training... '):
        src = batch['src_input_ids']
        trg = batch['tgt_input_ids']

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = loss_func(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(train_loader)) * 100, 2), '% , loss :', loss.item())

    os.makedirs('model_result', exist_ok=True)
    torch.save(model.state_dict(), 'model_result/model-{0}.pt'.format(epoch_loss / len(train_loader)))

    return epoch_loss / len(train_loader)


def evaluate(model, iterator, loss_func):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with (torch.no_grad()):
        for i, batch in enumerate(iterator):
            src = batch['src_input_ids']
            print(src)
            trg = batch['tgt_input_ids']
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = loss_func(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = dataset.tokenizer.convert_ids_to_tokens(batch['tgt_input_ids'][j])

                    # print('이거 잘 나옴? : ', trg_words)
                except Exception as e:
                    # print('왜 안되는가:')
                    print(e)
                    print(trg_words)
                try:
                    output_words = output[j].max(dim=1)[1]
                    output_words = dataset.tokenizer.convert_ids_to_tokens(output_words)
                    # print('이건 또 잘 나옴? : ', output_words)

                    bleu = sentence_bleu(references=trg_words.split(), hypothesis=output_words.split())
                    total_bleu.append(bleu)
                except:
                    print('문제 발생!')
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, belus = [], [], []
    for step in range(total_epoch):
        train_loss = train(model, train_loader, optimizer, loss_func, clip)
        valid_loss, bleu = evaluate(model, valid_loader, loss_func)

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        belus.append(bleu)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'model_result/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(belus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)