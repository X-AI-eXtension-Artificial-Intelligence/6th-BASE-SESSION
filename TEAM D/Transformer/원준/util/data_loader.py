from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


# 데이터셋 로딩, 전처리, 배치 생성 등을 담당하는 클래스
class DataLoader:
    # 클래스 레벨에서 사용할 Field 변수 선언 (타입 힌트)
    source: Field = None  # 소스 언어 필드
    target: Field = None  # 타겟 언어 필드

    # 클래스 초기화 함수
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext                          # 데이터 확장자 쌍 ('.en', '.de') 또는 ('.de', '.en')
        self.tokenize_en = tokenize_en          # 영어 토크나이저 함수
        self.tokenize_de = tokenize_de          # 독일어 토크나이저 함수
        self.init_token = init_token            # 문장 시작 토큰 (<sos>)
        self.eos_token = eos_token              # 문장 종료 토큰 (<eos>)
        print('dataset initializing start')     # 초기화 시작 로그

    # Multi30k 데이터셋을 기반으로 전처리된 학습/검증/테스트 데이터셋 생성
    def make_dataset(self):
        # 독일어 → 영어
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True, batch_first=True)

        # 영어 → 독일어
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True, batch_first=True)

        # Multi30k.splits는 데이터를 학습/검증/테스트로 나눠서 불러옴
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))

        return train_data, valid_data, test_data  # 분할된 데이터셋 반환

    # 학습 데이터로부터 어휘 사전 생성 (등장 빈도 기준)
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)  # 소스 언어 사전 생성
        self.target.build_vocab(train_data, min_freq=min_freq)  # 타겟 언어 사전 생성

    # 학습/검증/테스트용 데이터 반복자 생성 (배치 단위, 정렬 포함)
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),             # 세 데이터셋에 대해
            batch_size=batch_size,               # 배치 크기 설정
            device=device                         # GPU 또는 CPU 설정
        )
        print('dataset initializing done')        # 완료 메시지 출력
        return train_iterator, valid_iterator, test_iterator 
