"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

from conf import * 
from util.data_loader import DataLoader  
from util.tokenizer import Tokenizer     

# 영어/독일어 토큰화를 위한 토크나이저 객체 생성
tokenizer = Tokenizer()

"""
DataLoader 객체 초기화
 - ext: 입력 언어(.en), 출력 언어(.de) 확장자 설정
 - tokenize_en / tokenize_de: 각각 영어/독일어 토큰화 함수 전달
 - init_token: 문장 시작 토큰 (<sos>)
 - eos_token: 문장 끝 토큰 (<eos>)
"""
loader = DataLoader(
    ext=('.en', '.de'),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token='<sos>',
    eos_token='<eos>'
)

# Multi30k 데이터셋 로드
# train, valid, test: 각각 학습/검증/테스트용 Dataset 객체
train, valid, test = loader.make_dataset()

# 학습 데이터 기준으로 vocab(어휘 사전) 생성
# min_freq=2 이상인 단어만 포함시킴 (희귀 단어는 제거)
loader.build_vocab(train_data=train, min_freq=2)

# BucketIterator를 이용해 미니배치 생성
# 길이순 정렬을 통해 패딩 낭비 최소화
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    batch_size=batch_size,
    device=device
)

# special token 인덱스 추출 (마스크나 디코더 시작 입력에 사용됨)
src_pad_idx = loader.source.vocab.stoi['<pad>']  # 인코더용 패딩 토큰 인덱스
trg_pad_idx = loader.target.vocab.stoi['<pad>']  # 디코더용 패딩 토큰 인덱스
trg_sos_idx = loader.target.vocab.stoi['<sos>']  # 디코더 입력 시작 토큰 인덱스

# 인코더/디코더의 vocab 사이즈 정의 (임베딩 및 출력층에서 사용됨)
enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
