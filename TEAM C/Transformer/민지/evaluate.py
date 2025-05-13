import os
import math
from torch import nn, optim
from torch.optim import Adam
from nltk.translate.bleu_score import sentence_bleu
from util.data_config import *
from models.transformer import Transformer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate(model, iterator, loss_func, tokenizer, batch_size):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src_input_ids'].to(device)
            trg = batch['tgt_input_ids'].to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = loss_func(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            smoothie = SmoothingFunction().method4

            # output shape: (batch_size, trg_len, vocab_size)
            for j in range(batch['tgt_input_ids'].shape[0]):
                try:
                    trg_ids = batch['tgt_input_ids'][j].tolist()
                    trg_words = tokenizer.convert_ids_to_tokens(trg_ids)

                    pred_ids = output[j].argmax(dim=-1).tolist()
                    pred_words = tokenizer.convert_ids_to_tokens(pred_ids)

                    trg_sentence = ' '.join(trg_words).replace('<pad>', '').strip()
                    out_sentence = ' '.join(pred_words).replace('<pad>', '').strip()

                    bleu = sentence_bleu([trg_sentence.split()], out_sentence.split(), smoothing_function=smoothie)
                    total_bleu.append(bleu)
                except Exception as e:
                    print(f"BLEU 계산 중 오류 발생: {e}")
                continue

            if total_bleu:
                batch_bleu.append(sum(total_bleu) / len(total_bleu))

    avg_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0.0
    avg_loss = epoch_loss / len(iterator)

    return avg_loss, avg_bleu


if __name__ == "__main__":
    # 모델 불러오기

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

    optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=factor,
                                                    patience=patience)

    loss_func = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    model_path = 'model_result/model-11.498687694549561.pt'
    model.load_state_dict(torch.load(model_path))

    val_loss, val_bleu = evaluate(model, valid_loader, loss_func, dataset.tokenizer, batch_size)

    print(f'\nValidation Loss: {val_loss:.3f} | PPL: {math.exp(val_loss):.3f} | BLEU: {val_bleu:.3f}')
