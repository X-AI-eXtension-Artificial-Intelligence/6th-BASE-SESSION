from torch.utils.data import DataLoader

from .config import *
from .data_preprocessing import load_data, data_preprocessing
from .data_loader import TranslationDataset

train_data, valid_data, test_data = load_data()
src_sentences_trn, tgt_sentences_trn = data_preprocessing(train_data)
dataset = TranslationDataset(src_sentences_trn, tgt_sentences_trn, tokenizer_name="bert-base-multilingual-cased", max_length=max_len)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

src_sentences_val, tgt_sentences_val = data_preprocessing(valid_data)
dataset_val = TranslationDataset(src_sentences_val, tgt_sentences_val, tokenizer_name="bert-base-multilingual-cased", max_length=max_len)
valid_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)


src_pad_idx = dataset.tokenizer.convert_tokens_to_ids('[PAD]')
trg_pad_idx = dataset.tokenizer.convert_tokens_to_ids('[PAD]')
trg_sos_idx = dataset.tokenizer.convert_tokens_to_ids('[SOS]')

enc_voc_size = dataset.tokenizer.vocab_size
dec_voc_size = dataset.tokenizer.vocab_size
