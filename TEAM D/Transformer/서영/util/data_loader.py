"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
# torchtext의 이전 버전(legacy)의 데이터 처리 도구 사용
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    # 소스 언어와 타겟 언어의 Field 정의
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        # ext: ('.de', '.en') 또는 ('.en', '.de') 등의 언어 확장자 쌍
        self.ext = ext

        # 영어/독일어 토크나이저 함수
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de

        # 문장의 시작 및 종료 토큰
        self.init_token = init_token
        self.eos_token = eos_token

        print('dataset initializing start')  # 초기화 알림

    def make_dataset(self):
        # 언어 순서에 따라 source/target field 정의
        if self.ext == ('.de', '.en'):
            # 독일어 → 영어 번역인 경우
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            # 영어 → 독일어 번역인 경우
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        # Multi30k 데이터셋 로드 (train/valid/test 분할)
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        # source와 target 각각에 대해 단어 집합 구축 (최소 등장 빈도 min_freq 이상)
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        # 학습/검증/테스트 데이터를 미니배치 단위로 묶는 이터레이터 생성
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device
        )
        print('dataset initializing done')  # 완료 알림
        return train_iterator, valid_iterator, test_iterator
