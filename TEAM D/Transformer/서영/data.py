"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *  # 학습 및 모델 하이퍼파라미터 설정 import
from util.data_loader import DataLoader  # 데이터셋 생성 및 로더
from util.tokenizer import Tokenizer     # 영어/독일어 토크나이저

# SpaCy 기반 토크나이저 클래스 초기화
tokenizer = Tokenizer()

# DataLoader 클래스 초기화 (언어 쌍: 영어→독일어)
loader = DataLoader(
    ext=('.en', '.de'),                      # 언어 확장자 순서 (source, target)
    tokenize_en=tokenizer.tokenize_en,      # 영어 토크나이저 함수
    tokenize_de=tokenizer.tokenize_de,      # 독일어 토크나이저 함수
    init_token='<sos>',                     # 문장 시작 토큰
    eos_token='<eos>'                       # 문장 끝 토큰
)

# Multi30k 데이터셋 로딩 (학습/검증/테스트)
train, valid, test = loader.make_dataset()

# 어휘 사전 구축 (min_freq=2 이상인 단어만 포함)
loader.build_vocab(train_data=train, min_freq=2)

# BucketIterator 기반 미니배치 생성기 정의
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    batch_size=batch_size,
    device=device
)

# 패딩 및 시작 토큰 인덱스 추출 (마스킹 등에 필요)
src_pad_idx = loader.source.vocab.stoi['<pad>']   # source padding 인덱스
trg_pad_idx = loader.target.vocab.stoi['<pad>']   # target padding 인덱스
trg_sos_idx = loader.target.vocab.stoi['<sos>']   # target 시작 토큰 인덱스

# 소스/타겟 어휘 집합 크기 저장 (임베딩 레이어 등에 사용)
enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
