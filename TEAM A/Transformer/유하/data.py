# data.py : 데이터셋 로딩, 토크나이저 설정, 데이터셋 분할, vocabulary 생성, DataLoader 생성 등의 데이터 전처리 전반

from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'), # 데이터 파일 확장자 영어, 독일어로 지정
                    tokenize_en=tokenizer.tokenize_en, # 각각 영어, 독일어 토크나이저 함수로 지정
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>', # 문장 시작, 끝 토큰을 공백으로 설정
                    eos_token='<eos>')

train, valid, test = loader.make_dataset() # make_dataset() 메소드 호출 -> 학습, 검증, 테스트 데이터셋 생성
loader.build_vocab(train_data=train, min_freq=2) # 학습 데이터에서 최소 2번 이상 등장한 토큰만 포함해 단어장 만듦
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test, # 학습/검증/테스트 데이터셋을 PyTorch DataLoader 형태의 iterator로 만듦
                                                     batch_size=batch_size,
                                                     device=device)

src_pad_idx = loader.source.vocab.stoi['<pad>'] # 입력 언어 단어장에서 패딩 토큰의 인덱스를 가져옴
trg_pad_idx = loader.target.vocab.stoi['<pad>'] # 출력 언어 단어장에서 패딩 토큰의 인덱스를 가져옴
trg_sos_idx = loader.target.vocab.stoi['<sos>'] # 출력 언어 단어장에서 시작 토큰의 인덱스를 가져옴

enc_voc_size = len(loader.source.vocab) # 입력 언어 단어장의 전체 토큰 개수 저장
dec_voc_size = len(loader.target.vocab) # 출력 언어 단어장의 전체 토큰 개수 저장
