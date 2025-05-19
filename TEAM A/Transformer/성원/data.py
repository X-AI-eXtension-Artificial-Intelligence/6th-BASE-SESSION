""" 
데이터셋 로드 → 토큰화 → Vocab 생성 → Iterator로 배치 준비" 전체 과정을 자동화한 코드
"""


from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('en', 'de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

src_pad_idx = loader.vocab_src['<pad>']
trg_pad_idx = loader.vocab_trg['<pad>']
trg_sos_idx = loader.vocab_trg['<sos>']


enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
