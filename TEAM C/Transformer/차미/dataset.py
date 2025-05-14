import spacy
import torch
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k

'''
토큰화
'''

def load_tokenizers():
    """Load Spacy tokenizers for English and German."""
    try:
        spacy_en = spacy.load('en_core_web_sm')  # 영어 토크나이저
        spacy_de = spacy.load('de_core_news_sm')  # 독일어 토크나이저
    except OSError:
        raise RuntimeError(
            "Spacy 모델이 설치되지 않았습니다. 다음 명령어를 실행하세요:\n"
            "!python -m spacy download en_core_web_sm\n"
            "!python -m spacy download de_core_news_sm"
        )
    return spacy_en, spacy_de

# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


'''
- 필드(field) 라이브러리 이용해 데이터셋에 대한 구체적인 전처리 내용 명시
- Seq2Seq 모델과 다르게 batch_first 속성의 값을 True로 설정
- 소스(SRC): 독일어 -> 목표(TRG): 영어
'''

def load_dataset(batch_size=128, device=None):
    """Load Multi30k dataset with tokenized fields and iterators."""
    global spacy_en, spacy_de
    spacy_en, spacy_de = load_tokenizers()

    SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)

    # 데이터셋 로드
    ## Multi30k: 대표적인 영어-독어 번역 데이터셋
    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    print(f"학습 데이터셋 크기: {len(train_dataset.examples)}개")
    print(f"검증 데이터셋 크기: {len(valid_dataset.examples)}개")
    print(f"테스트 데이터셋 크기: {len(test_dataset.examples)}개")

    # 어휘 사전 생성
    ## 필드(field) 객체의 build_vocab 메서드를 이용해 영어와 독어 단어 사전 생성
    ## -> 최소 2번 이상 등장한 단어만을 선택
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    # 배치 로더 생성
    ## 한 문장에 포함된 단어가 순서대로 나열된 상태로 네트워크에 입력되어야 함
    ## -> 때문에 하나의 배치에 포함된 문장들의 단어의 수가 유사하도록 만들면 좋음
    ## -> BucketIterator 사용 (배치 크기(batch size): 128)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size=batch_size,
        device=device
    )

    return SRC, TRG, train_iterator, valid_iterator, test_iterator

def get_device():
    """Return available device (cuda or cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    device = get_device()
    SRC, TRG, train_iterator, valid_iterator, test_iterator = load_dataset(device=device)
    print(f"사용할 디바이스: {device}")