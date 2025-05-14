"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()  # 토크나이저 인스턴스 생성
loader = DataLoader(  # 데이터 로더 인스턴스 생성
    ext=('.en', '.de'),  # 파일 확장자 설정
    tokenize_en=tokenizer.tokenize_en,  # 영어 토크나이징 함수
    tokenize_de=tokenizer.tokenize_de,  # 독일어 토크나이징 함수
    init_token='<sos>',  # 시작 토큰
    eos_token='<eos>'  # 종료 토큰
)
train, valid, test = loader.make_dataset()  # 데이터셋 생성
loader.build_vocab(train_data=train, min_freq=2)  # 단어장 생성(최소빈도2)
train_iter, valid_iter, test_iter = loader.make_iter(  # 데이터 반복자 생성
    train, valid, test, batch_size=batch_size, device=device
)
src_pad_idx = loader.source.vocab.stoi['<pad>']  # 소스 패딩 토큰 인덱스
trg_pad_idx = loader.target.vocab.stoi['<pad>']  # 타겟 패딩 토큰 인덱스
trg_sos_idx = loader.target.vocab.stoi['<sos>']  # 타겟 시작 토큰 인덱스
enc_voc_size = len(loader.source.vocab)  # 인코더 어휘 크기
dec_voc_size = len(loader.target.vocab)  # 디코더 어휘 크기