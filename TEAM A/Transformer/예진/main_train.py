
# main_train.py
# 학습 전체 파이프라인 통합

import math, time, torch, os, pickle
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model import Transformer

# -----------------------------
# Tokenizer
# -----------------------------
import sentencepiece as spm
class Tokenizer:
    def __init__(self, model_path="spm.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, add_special_tokens=True):
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def pad_id(self):
        return self.sp.pad_id()

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

# ... (중략) 전체 코드는 canvas에서 사용 가능
