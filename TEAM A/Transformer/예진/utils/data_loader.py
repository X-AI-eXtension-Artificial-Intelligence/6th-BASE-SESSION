"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    """
    모델 학습 위한 데이터셋 로더 클래스

    기능:
    - Multi30k 번역 데이터셋 로드
    - source/target 언어 Field 정의 (토큰화 및 토큰 설정 포함)
    - Vocab 생성
    - BucketIterator로 미니배치 생성
    """

    # Field 객체는 외부에서도 접근할 수 있도록 클래스 속성으로 선언
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        """
        ext: 번역 언어쌍의 파일 확장자 튜플 (예: ('.de', '.en'))
        tokenize_en: 영어 토큰화 함수
        tokenize_de: 독일어 토큰화 함수
        init_token: 문장 시작 토큰 (<sos>)
        eos_token: 문장 끝 토큰 (<eos>)
        """
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token

        print('dataset initializing start')

    def make_dataset(self):
        """
        Field 객체 정의 및 Multi30k 데이터셋 로드

        반환: train_data, valid_data, test_data (torchtext Dataset 객체)
        """

        # 독일어 -> 영어 번역 설정일 경우
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,               # 모두 소문자로 변환
                                batch_first=True)         # shape: [batch, seq_len]

            self.target = Field(tokenize=self.tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)

        # 영어 -> 독일어 번역 설정일 경우
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)

            self.target = Field(tokenize=self.tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)

        # Multi30k 데이터셋에서 train/val/test 분리
        train_data, valid_data, test_data = Multi30k.splits(
            exts=self.ext,
            fields=(self.source, self.target)
        )

        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        """
        Field 객체 기준으로 단어 사전(vocab) 생성

        train_data: 학습 데이터셋
        min_freq: 단어가 최소 몇 번 등장해야 vocab에 포함될지
        """
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        """
        BucketIterator를 생성하여 데이터를 미니배치로 나누는 함수

        train, validate, test: 데이터셋 객체들

        반환: train_iterator, valid_iterator, test_iterator
        """
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device
        )

        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
