
# utils.py
# 공통 함수 모음 (시간 측정, BLEU 계산 등)

import math
import time
import numpy as np
from collections import Counter

# -----------------------------
# 1. 학습 시간 측정
# -----------------------------
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

# -----------------------------
# 2. BLEU 계산 관련 함수
# -----------------------------
def bleu_stats(hypothesis, reference):
    stats = [len(hypothesis), len(reference)]
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))
    return stats

def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    c, r = stats[:2]
    log_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_prec)

def get_bleu(hypotheses, references):
    stats = np.array([0.] * 10)
    for hyp, ref in zip(hypotheses, references):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

# -----------------------------
# 3. 기타 유틸: idx_to_word (옵션)
# -----------------------------
def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    return " ".join(words)
