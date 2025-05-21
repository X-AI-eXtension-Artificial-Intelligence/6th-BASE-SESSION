import os
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch

class IWSLTDataset(Dataset):
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab, init_token='<sos>', eos_token='<eos>'):
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(trg_path, 'r', encoding='utf-8') as f:
            self.trg_lines = f.readlines()

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.init_token = init_token
        self.eos_token = eos_token

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.src_tokenizer(self.src_lines[idx].strip())
        trg = self.trg_tokenizer(self.trg_lines[idx].strip())

        src_ids = [self.src_vocab[self.init_token]] + [self.src_vocab[token] for token in src] + [self.src_vocab[self.eos_token]]
        trg_ids = [self.trg_vocab[self.init_token]] + [self.trg_vocab[token] for token in trg] + [self.trg_vocab[self.eos_token]]

        return torch.tensor(src_ids), torch.tensor(trg_ids)

class DataLoaderModern:
    def __init__(self, tokenize_en, tokenize_de, init_token='<sos>', eos_token='<eos>'):
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token

    def yield_tokens(self, path, tokenizer):
        with open(path, encoding='utf-8') as f:
            for line in f:
                yield tokenizer(line.strip())

    def build_vocab(self, path, tokenizer):
        vocab = build_vocab_from_iterator(self.yield_tokens(path, tokenizer), specials=[self.init_token, self.eos_token, '<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def build_dataset(self):
        src_tokenizer = self.tokenize_en
        trg_tokenizer = self.tokenize_de

        src_vocab = self.build_vocab('data/train.en', src_tokenizer)
        trg_vocab = self.build_vocab('data/train.de', trg_tokenizer)

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        train_dataset = IWSLTDataset('data/train.en', 'data/train.de', src_tokenizer, trg_tokenizer, src_vocab, trg_vocab)
        valid_dataset = IWSLTDataset('data/valid.en', 'data/valid.de', src_tokenizer, trg_tokenizer, src_vocab, trg_vocab)
        test_dataset = IWSLTDataset('data/test.en', 'data/test.de', src_tokenizer, trg_tokenizer, src_vocab, trg_vocab)

        return train_dataset, valid_dataset, test_dataset

    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, padding_value=self.src_vocab['<pad>'], batch_first=True)
        trg_padded = pad_sequence(trg_batch, padding_value=self.trg_vocab['<pad>'], batch_first=True)
        return src_padded, trg_padded

    def make_iter(self, train_dataset, valid_dataset, test_dataset, batch_size, device):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        return train_loader, valid_loader, test_loader