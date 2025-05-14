"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import MyDataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = MyDataLoader(#ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train)    #어휘크기 구하기


# 패딩 인덱스 정의 (수동으로)
src_pad_idx = loader.src_pad_idx
trg_pad_idx = loader.trg_pad_idx
trg_sos_idx = loader.trg_sos_idx

# 패딩 인덱스를 활용해 encoder와 decoder의 vocabulary size를 설정
enc_voc_size = len(loader.source_vocab)
dec_voc_size = len(loader.target_vocab)


train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

# 최신 torchtext에서는 vocab을 자동으로 처리하므로 이 부분을 수동으로 처리할 필요 없음
# loader.source[target].vocab은 더 이상 사용되지 않음
# src_pad_idx = loader.source.vocab.stoi['<pad>']
# trg_pad_idx = loader.target.vocab.stoi['<pad>']
# trg_sos_idx = loader.target.vocab.stoi['<sos>']

# enc_voc_size = len(loader.source.vocab)
# dec_voc_size = len(loader.target.vocab)
