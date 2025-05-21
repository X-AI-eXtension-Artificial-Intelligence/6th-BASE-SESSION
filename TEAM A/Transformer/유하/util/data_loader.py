# 호환 안 되어서 수정함 !
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy

'''
class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self): # 언어 확장자에 따라 source/target Field를 지정하고, Multi30k 데이터셋을 분할하여 반환
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq): # 학습 데이터에서 최소 등장 빈도 이상인 토큰만으로 source/target vocabulary를 생성
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device): # 학습/검증/테스트 데이터를 배치 단위로 불러올 수 있는 iterator로 변환하여 반환
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
'''

from torchtext.data import Field, BucketIterator, TabularDataset
import spacy

spacy_eng = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

SRC = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)

def load_data(batch_size=128, device='cpu'):
    fields = {'src': ('src', SRC), 'trg': ('trg', TRG)}

    train_data, valid_data, test_data = TabularDataset.splits(
        path='data',
        train='train.tsv',
        validation='valid.tsv',
        test='test.tsv',
        format='tsv',
        fields=[('src', SRC), ('trg', TRG)]
    )

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src)
    )

    return train_iterator, valid_iterator, test_iterator, SRC, TRG
