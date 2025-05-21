from conf import *
from util import get_tokenizers, build_vocabs, get_text_transform, get_collate_fn
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader

# 1. 토크나이저 준비
tokenize_en, tokenize_de = get_tokenizers()

# 2. 데이터셋 불러오기 (영어-독일어)
train_data = list(Multi30k(split='train', language_pair=('en', 'de')))
valid_data = list(Multi30k(split='valid', language_pair=('en', 'de')))
test_data  = list(Multi30k(split='test',  language_pair=('en', 'de')))

# 3. vocab 생성 (train_data만 사용)
vocab_en, vocab_de = build_vocabs(train_data, tokenize_en, tokenize_de)

# 4. 텍스트 변환 함수
text_transform_en = get_text_transform(vocab_en, tokenize_en)
text_transform_de = get_text_transform(vocab_de, tokenize_de)

# 5. collate_fn 준비
PAD_IDX_EN = vocab_en['<pad>']
PAD_IDX_DE = vocab_de['<pad>']
collate_fn = get_collate_fn(text_transform_en, text_transform_de, PAD_IDX_EN, PAD_IDX_DE)

# 6. DataLoader 생성
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 7. 인덱스, vocab 사이즈 등
src_pad_idx = PAD_IDX_EN
trg_pad_idx = PAD_IDX_DE
src_sos_idx = vocab_en['<sos>']
trg_sos_idx = vocab_de['<sos>']
enc_voc_size = len(vocab_en)
dec_voc_size = len(vocab_de)

# 8. 외부에서 import할 수 있게 변수/객체 export
__all__ = [
    "train_loader", "valid_loader", "test_loader",
    "src_pad_idx", "trg_pad_idx", "src_sos_idx", "trg_sos_idx",
    "enc_voc_size", "dec_voc_size",
    "vocab_en", "vocab_de"
]
