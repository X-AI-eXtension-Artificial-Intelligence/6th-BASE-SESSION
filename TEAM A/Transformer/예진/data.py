from data_loader import DataLoaderWrapper

loader = DataLoaderWrapper(max_len=128)
train_dataset = loader.make_dataset(split='train')
valid_dataset = loader.make_dataset(split='validation')

train_iter = loader.make_iter(train_dataset, batch_size=32)
valid_iter = loader.make_iter(valid_dataset, batch_size=32)

# 특수 토큰 인덱스
src_pad_idx = loader.token2id['<pad>']
trg_pad_idx = loader.token2id['<pad>']
trg_sos_idx = loader.token2id['<sos>']

enc_voc_size = len(loader.token2id)
dec_voc_size = len(loader.token2id)
