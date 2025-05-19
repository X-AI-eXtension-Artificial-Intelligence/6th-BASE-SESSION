"""  
Transformer 기반 번역 모델 학습을 위한 데이터 준비 파이프라인을 구현한 거야.
PyTorch와 torchtext 라이브러리를 이용해서 독일어-영어 번역 데이터셋 (Multi30k) 을 처리하고,
학습에 필요한 토크나이징, Vocabulary 생성, 데이터 배치까지 담당하는 클래스
"""

import os 
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
# from torchtext.datasets import Multi30k  # 사용하는 데이터셋 
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class DataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token='<sos>', eos_token='<eos>', min_freq=2):
        self.ext = ext
        self.init_token = init_token
        self.eos_token = eos_token
        self.min_freq = min_freq
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de

        # Tokenizers (spacy 사용)
        import spacy
        self.tokenizer_de = spacy.load("de_core_news_sm").tokenizer
        self.tokenizer_en = spacy.load("en_core_web_sm").tokenizer

        print('Dataset initializing start')

    def yield_tokens(self, data_iter, language):
        tokenizer = self.tokenizer_de if language == 'de' else self.tokenizer_en
        for src, trg in data_iter:
            text = src if language == 'de' else trg
            yield [tok.text.lower() for tok in tokenizer(text)]

    # def make_dataset(self):
    #     data_root = './'  # 원하는 데이터 저장 경로

    #     train_data = list(Multi30k(root=data_root, split='train', language_pair=self.ext))
    #     valid_data = list(Multi30k(root=data_root, split='valid', language_pair=self.ext))
    #     test_data = list(Multi30k(root=data_root, split='test', language_pair=self.ext))
    #     return train_data, valid_data, test_data

# 메모리 문제로 실행이 안돼서 데이터를 줄여서 실행 
    def make_dataset(self):
        data_dir = './datasets/'  # 실제 파일 위치에 맞게 수정

        def load_file(src_path, trg_path):
            with open(src_path, encoding='utf-8') as src_f, open(trg_path, encoding='utf-8') as trg_f:
                src_lines = [line.strip() for line in src_f.readlines()]
                trg_lines = [line.strip() for line in trg_f.readlines()]
            return list(zip(src_lines, trg_lines))

        train_data = load_file(
            os.path.join(data_dir, 'train.de'), 
            os.path.join(data_dir, 'train.en')
        )
        valid_data = load_file(
            os.path.join(data_dir, 'val.de'), 
            os.path.join(data_dir, 'val.en')
        )
        test_data = load_file(
            os.path.join(data_dir, 'test_2016_flickr.de'), 
            os.path.join(data_dir, 'test_2016_flickr.en')
        )

        # (테스트용으로 데이터 줄이기)
        train_data = train_data
        valid_data = valid_data
        test_data = test_data

        return train_data, valid_data, test_data





    def build_vocab(self, train_data):
        src_lang = self.ext[0]
        trg_lang = self.ext[1]

        self.vocab_src = build_vocab_from_iterator(
            self.yield_tokens(train_data, src_lang),
            specials=[self.init_token, self.eos_token, '<unk>', '<pad>'],
            min_freq=self.min_freq
        )
        self.vocab_src.set_default_index(self.vocab_src['<unk>'])

        self.vocab_trg = build_vocab_from_iterator(
            self.yield_tokens(train_data, trg_lang),
            specials=[self.init_token, self.eos_token, '<unk>', '<pad>'],
            min_freq=self.min_freq
        )
        self.vocab_trg.set_default_index(self.vocab_trg['<unk>'])

        self.source = self.vocab_src
        self.target = self.vocab_trg

    def data_process(self, raw_data):
        src_lang = self.ext[0]
        trg_lang = self.ext[1]
        tokenizer_src = self.tokenizer_de if src_lang == 'de' else self.tokenizer_en
        tokenizer_trg = self.tokenizer_de if trg_lang == 'de' else self.tokenizer_en

        data = []
        for src_text, trg_text in raw_data:
            src_tokens = [tok.text.lower() for tok in tokenizer_src(src_text)]
            trg_tokens = [tok.text.lower() for tok in tokenizer_trg(trg_text)]

            src_tensor = [self.vocab_src[self.init_token]] + [self.vocab_src[token] for token in src_tokens] + [self.vocab_src[self.eos_token]]
            trg_tensor = [self.vocab_trg[self.init_token]] + [self.vocab_trg[token] for token in trg_tokens] + [self.vocab_trg[self.eos_token]]

            data.append((torch.tensor(src_tensor, dtype=torch.long), torch.tensor(trg_tensor, dtype=torch.long)))
        return data

    def generate_batch(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.vocab_src['<pad>'], batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=self.vocab_trg['<pad>'], batch_first=True)
        return src_batch, trg_batch

    def make_iter(self, train_data, valid_data, test_data, batch_size, device):
        train_data = self.data_process(train_data)
        valid_data = self.data_process(valid_data)
        test_data = self.data_process(test_data)

        train_iter = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch)
        valid_iter = TorchDataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=self.generate_batch)
        test_iter = TorchDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=self.generate_batch)

        print('Dataset initializing done')
        return train_iter, valid_iter, test_iter


