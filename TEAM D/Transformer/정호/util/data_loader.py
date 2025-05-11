"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    # 소스 언어와 타겟 언어에 대한 Field 객체 (토크나이징 및 처리 정의)
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        
        # param ext: 사용할 파일 확장자 튜플 (ex. ('.de', '.en'))
        # param tokenize_en: 영어 토크나이저 함수
        # param tokenize_de: 독일어 토크나이저 함수
        # param init_token: 시작 토큰 (<sos>)
        # param eos_token: 종료 토큰 (<eos>)
        
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')  # 초기화 메시지 출력

    def make_dataset(self):
        
        # field 객체를 정의하고 Multi30k 데이터셋에서 train/valid/test 분리

        # return: train_data, valid_data, test_data
        
        if self.ext == ('.de', '.en'):
            # 독일어가 source, 영어가 target
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            # 영어가 source, 독일어가 target
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        # Multi30k 데이터셋 로드 (train, validation, test)
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))

        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        
        # 어휘 사전 생성

        # param train_data: 학습 데이터셋
        # param min_freq: 최소 등장 빈도 (이보다 낮은 단어는 <unk> 처리됨)
        
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        
        # 데이터셋을 배치 단위로 묶은 이터레이터 생성

        # param train: 학습 데이터셋
        # param validate: 검증 데이터셋
        # param test: 테스트 데이터셋
        # param batch_size: 배치 크기
        # param device: 연산 디바이스 (cpu or cuda)
        # return: 학습/검증/테스트용 배치 이터레이터
        
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device
        )

        print('dataset initializing done')  # 완료 메시지 출력
        return train_iterator, valid_iterator, test_iterator

