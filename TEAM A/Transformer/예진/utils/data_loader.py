from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import torch

class DataLoaderWrapper:
    """
    CNN/DailyMail 요약 데이터셋을 불러오고,
    vocab 생성 및 DataLoader를 만드는 클래스
    """

    def __init__(self, max_len=128, vocab_size=30000, min_freq=2):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token2id = None
        self.id2token = None

    def build_vocab(self, texts):
        """
        전체 문장 기반으로 vocab 생성 (단어 -> 숫자)
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        vocab += [tok for tok, freq in counter.items() if freq >= self.min_freq]
        vocab = vocab[:self.vocab_size]

        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def tokenize(self, text):
        # 간단한 토크나이저: 단어 및 특수문자 단위로 나눔
        return re.findall(r"\w+|\S", text.lower())

    def preprocess(self, src, tgt):
        src_tokens = self.tokenize(src)
        tgt_tokens = self.tokenize(tgt)

        src_ids = [self.token2id.get(tok, self.token2id['<unk>']) for tok in src_tokens]
        tgt_ids = [self.token2id.get(tok, self.token2id['<unk>']) for tok in tgt_tokens]

        # decoder_input: <sos> + truncated target
        decoder_input = [self.token2id['<sos>']] + tgt_ids
        label = tgt_ids + [self.token2id['<eos>']]

        # 길이 잘라주기 (max_len 기준)
        decoder_input = decoder_input[:self.max_len]
        label = label[:self.max_len]
        src_ids = src_ids[:self.max_len]

        # 패딩
        pad = self.token2id['<pad>']
        src_ids += [pad] * (self.max_len - len(src_ids))
        decoder_input += [pad] * (self.max_len - len(decoder_input))
        label += [pad] * (self.max_len - len(label))

        return {
            "input_ids": torch.tensor(src_ids),
            "decoder_input_ids": torch.tensor(decoder_input),
            "labels": torch.tensor(label),
        }


    def make_dataset(self, split='train'):
        data = load_dataset("cnn_dailymail", "3.0.0")[split]
        src_list = [item['article'] for item in data]
        tgt_list = [item['highlights'] for item in data]

        if self.token2id is None:
            self.build_vocab(src_list + tgt_list)

        dataset = SummaryDataset(src_list, tgt_list, self)
        return dataset

    def make_iter(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SummaryDataset(Dataset):
    """
    HuggingFace CNN/DailyMail용 Dataset 클래스
    """

    def __init__(self, src_list, tgt_list, loader):
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.loader = loader

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        return self.loader.preprocess(self.src_list[idx], self.tgt_list[idx])
