# apply_dataset.py (updated: multi-epoch + BLEU evaluation)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from models.model.transformer import Transformer
from torchtext.data.metrics import bleu_score

# Tokenizers
SRC_LANGUAGE = 'de'
TRG_LANGUAGE = 'en'

token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TRG_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
}

# Special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocab
def yield_tokens(data_iter, language):
    for src, trg in data_iter:
        yield token_transform[language](src if language == SRC_LANGUAGE else trg)

train_data = list(Multi30k(split='train'))
valid_data = list(Multi30k(split='valid'))

vocab_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_data, ln),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_transform[ln].set_default_index(UNK_IDX)

# Text pipeline
text_transform = {
    ln: lambda x: [BOS_IDX] + vocab_transform[ln](token_transform[ln](x)) + [EOS_IDX]
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]
}

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(torch.tensor(text_transform[SRC_LANGUAGE](src_sample), dtype=torch.long))
        trg_batch.append(torch.tensor(text_transform[TRG_LANGUAGE](trg_sample), dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX).transpose(0, 1)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX).transpose(0, 1)
    return src_batch, trg_batch

# Dataloader
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=1, collate_fn=collate_fn)

# Model
INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TRG_LANGUAGE])

model = Transformer(
    src_pad_idx=PAD_IDX,
    trg_pad_idx=PAD_IDX,
    trg_sos_idx=BOS_IDX,
    enc_voc_size=INPUT_DIM,
    dec_voc_size=OUTPUT_DIM,
    d_model=512,
    n_head=8,
    max_len=100,
    ffn_hidden=2048,
    n_layers=6,
    drop_prob=0.1,
    device=device
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop with epoch
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, (src, trg) in enumerate(train_dataloader):
        src = src.to(device)
        trg = trg.to(device)

        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:].reshape(-1)

        optimizer.zero_grad()
        output = model(src, trg_input)
        output = output.reshape(-1, output.shape[-1])

        loss = criterion(output, trg_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_dataloader):.4f}")

    # BLEU evaluation
    model.eval()
    translations, references = [], []
    with torch.no_grad():
        for src, trg in valid_dataloader:
            src = src.to(device)
            encoder_out = model.encoder(src)
            generated = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)

            for _ in range(100):
                out = model.decoder(generated, encoder_out)
                pred = out[:, -1, :].argmax(-1).unsqueeze(1)
                generated = torch.cat([generated, pred], dim=1)
                if pred.item() == EOS_IDX:
                    break

            translated_tokens = generated.squeeze().tolist()[1:-1]  # remove BOS, EOS
            reference_tokens = trg.squeeze().tolist()[1:-1]

            translations.append([vocab_transform[TRG_LANGUAGE].lookup_token(tok) for tok in translated_tokens])
            references.append([[vocab_transform[TRG_LANGUAGE].lookup_token(tok) for tok in reference_tokens]])

    bleu = bleu_score(translations, references)
    print(f"Epoch {epoch+1} BLEU score: {bleu:.4f}")
