"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.legacy.datasets.translation import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from collections import Counter  # 추가
import torch 
from torch.nn.utils.rnn import pad_sequence 

# 원본 코드에서는 **Field**로 모든 전처리 과정을 자동화했고, 배치 처리는 **BucketIterator**로 자동 처리됨
# 현재 코드에서는 **get_tokenizer**로 토큰화만 처리하고, 배치 처리와 어휘 생성을 사용자가 직접 설정해야 함

class MyDataLoader:  # (PyTorch의 DataLoader와 이름이 겹치지 않게 하기 위해 MyDataLoader로 명명)
    # source: Field = None
    # target: Field = None
    source = None   # 원문
    target = None   # 번역문 

    def __init__(self, tokenize_en, tokenize_de, init_token, eos_token):  # ext 삭제
        # self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')
        
        self.source = get_tokenizer(self.tokenize_de)
        self.target = get_tokenizer(self.tokenize_en)

    def make_dataset(self):
        # if self.ext == ('.de', '.en'):  # 독일어 -> 영어
            # self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
            #                     lower=True, batch_first=True)
            # self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
            #                     lower=True, batch_first=True)            
        # elif self.ext == ('.en', '.de'):  # 영어 -> 독일어
            # self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
            #                     lower=True, batch_first=True)
            # self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
            #                     lower=True, batch_first=True)

        # Multi30k 데이터셋 로드
        # train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))
        return train_data, valid_data, test_data

    # def build_vocab(self, train_data, min_freq):
        # self.source.build_vocab(train_data, min_freq=min_freq)
        # self.target.build_vocab(train_data, min_freq=min_freq)
    def build_vocab(self, train_data):
        counter_src = Counter()
        counter_trg = Counter()
        
        # 토큰화된 데이터를 통해 어휘 수집
        for example in train_data:
            counter_src.update(self.source(example[0]))
            counter_trg.update(self.target(example[1]))
        
        # 어휘 크기 출력 (Counter에서 크기 확인)
        print("Source Vocabulary Size: ", len(counter_src))
        print("Target Vocabulary Size: ", len(counter_trg))
    
        
        # 어휘 인덱스를 만든다 (stoi = string to index)
        self.source_vocab = {word: idx for idx, (word, _) in enumerate(counter_src.items())}
        self.target_vocab = {word: idx for idx, (word, _) in enumerate(counter_trg.items())}

        # 패딩 인덱스를 정의 (Padding token의 인덱스를 수동으로 정의)
        self.src_pad_idx = self.source_vocab.get('<pad>', 0)  # '<pad>'이 없으면 0으로 설정
        self.trg_pad_idx = self.target_vocab.get('<pad>', 0)
        self.trg_sos_idx = self.target_vocab.get('<sos>', 1)  # '<sos>'의 인덱스 (기본값 1)

        # <unk> 추가 (어휘에 없을 경우 <unk>로 처리)
        self.source_vocab['<unk>'] = len(self.source_vocab)
        self.target_vocab['<unk>'] = len(self.target_vocab)

        # UNK 토큰 인덱스 설정
        self.UNK_IDX = self.source_vocab.get('<unk>', 1)  # <unk> 토큰의 인덱스를 1로 설정


    def make_iter(self, train, validate, test, batch_size, device):
        # train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
        #                                                                       batch_size=batch_size, device=device)
        # PyTorch DataLoader로 배치 처리
        train_loader = DataLoader(train, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=True)
        valid_loader = DataLoader(validate, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=False)
        
        print('dataset initializing done')
        # return train_iterator, valid_iterator, test_iterator
        return train_loader, valid_loader, test_loader  # 훈련, 검증, 테스트용 DataLoader 반환

    # DataLoader에서는 배치 단위로 데이터를 모을 때 패딩이나 길이 맞추기를 수동으로 정의해야 하므로 **collate_fn**을 추가 작성해야 함.
    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 함수 (패딩 및 길이 맞추기)
        :param batch: 배치 데이터
        :return: 패딩된 배치 데이터
        """
        # batch: [(src_sequence_1, trg_sequence_1), (src_sequence_2, trg_sequence_2), ...]
        
        # 여기서 src_sequence와 trg_sequence는 토큰화된 시퀀스
        src_batch, trg_batch = zip(*batch)  # 배치에서 src와 trg 데이터를 분리
            
        # 'stoi'를 사용하여 문자열을 숫자 인덱스로 변환
        src_batch = [
            torch.tensor([self.source_vocab.get(word, self.UNK_IDX) for word in seq])  # 없는 단어는 <unk> 처리
            for seq in src_batch
        ]
        trg_batch = [
            torch.tensor([self.target_vocab.get(word, self.UNK_IDX) for word in seq])  # 타겟 문장도 <unk> 처리
            for seq in trg_batch
        ]

        # 패딩을 추가하여 동일한 길이로 맞추기
        src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)  # src 패딩
        trg_batch_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_pad_idx)  # trg 패딩

        # src와 trg 배치 데이터 반환
        return src_batch_padded, trg_batch_padded  # 패딩된 배치 데이터 반환