from conf import *  # 하이퍼파라미터 및 환경 설정 (예: batch_size, device 등)
from util.data_loader import DataLoader  # 사용자 정의 데이터 로더 클래스
from util.tokenizer import Tokenizer     # 사용자 정의 토크나이저 클래스

# 토크나이저 초기화 (영어/독일어 전처리용 함수 포함)
tokenizer = Tokenizer()

# 데이터로더 인스턴스화 및 언어 설정
loader = DataLoader(
    ext=('.en', '.de'),                  # 영어 → 독일어 번역을 위한 파일 확장자 설정
    tokenize_en=tokenizer.tokenize_en,   # 영어 문장 토크나이징 함수
    tokenize_de=tokenizer.tokenize_de,   # 독일어 문장 토크나이징 함수
    init_token='<sos>',                  # 문장 시작 토큰
    eos_token='<eos>'                    # 문장 끝 토큰
)

#  데이터셋 생성 (train, valid, test)
train, valid, test = loader.make_dataset()

#  단어 사전 구축 (vocab 생성), 최소 등장 횟수 2 이상인 단어만 포함
loader.build_vocab(train_data=train, min_freq=2)

#  배치 단위 데이터 반복자 생성 (torchtext의 iterator로 구성)
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    batch_size=batch_size,
    device=device
)

# PAD 토큰 인덱스 가져오기 (마스킹 등에 사용됨)
src_pad_idx = loader.source.vocab.stoi['<pad>']  # 인코더 입력용 PAD 인덱스
trg_pad_idx = loader.target.vocab.stoi['<pad>']  # 디코더 입력용 PAD 인덱스
trg_sos_idx = loader.target.vocab.stoi['<sos>']  # 디코더 시작 인덱스 (<sos> 토큰)
# 특수 토큰들의 "숫자 인덱스"를 가져오는 작업    stoi  str to idx


# 단어 사전 크기 저장 (학습 모델의 embedding/linear layer에 사용)
enc_voc_size = len(loader.source.vocab)  # 인코더의 vocabulary size
dec_voc_size = len(loader.target.vocab)  # 디코더의 vocabulary size
