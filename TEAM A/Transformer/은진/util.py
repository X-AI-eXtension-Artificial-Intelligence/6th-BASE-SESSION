import math
from collections import Counter
import numpy as np
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 1. BLEU 점수 계산 함수

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, references):
    stats = np.array([0.] * 10)
    for hyp, ref in zip(hypotheses, references):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

def get_bleu_nltk(hypotheses, references):
    chencherry = SmoothingFunction()
    scores = []
    for hyp, ref in zip(hypotheses, references):
        # reference는 list of list로 넘겨야 함
        scores.append(sentence_bleu([ref], hyp, smoothing_function=chencherry.method4))
    return 100 * sum(scores) / len(scores)

# 2. 인덱스 → 단어 변환 함수

def idx_to_word(x, vocab):
    # vocab: torchtext.vocab.Vocab 객체
    words = []
    for i in x:
        word = vocab.lookup_token(i)
        if '<' not in word:
            words.append(word)
    return " ".join(words)

# 3. epoch 시간 측정 함수

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 4. 토크나이저 함수 (spacy 기반)

def get_tokenizers():
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    return tokenize_de, tokenize_en

# 5. vocab 생성 함수

def build_vocabs(train_iter, tokenize_de, tokenize_en, specials=['<unk>', '<pad>', '<sos>', '<eos>']):
    def yield_tokens(data_iter, tokenizer, idx):
        for pair in data_iter:
            yield tokenizer(pair[idx])
    vocab_de = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenize_de, 0),
        specials=specials,
        special_first=True
    )
    vocab_en = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenize_en, 1),
        specials=specials,
        special_first=True
    )
    vocab_de.set_default_index(vocab_de['<unk>'])
    vocab_en.set_default_index(vocab_en['<unk>'])
    return vocab_de, vocab_en

# 6. 텍스트 → 인덱스 변환 함수

def get_text_transform(vocab, tokenizer):
    def text_transform(text):
        return [vocab['<sos>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<eos>']]
    return text_transform

# 7. collate_fn (DataLoader용 배치 패딩 함수)

def get_collate_fn(text_transform_src, text_transform_tgt, pad_idx_src, pad_idx_tgt):
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(torch.tensor(text_transform_src(src_sample), dtype=torch.long))
            tgt_batch.append(torch.tensor(text_transform_tgt(tgt_sample), dtype=torch.long))
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx_src)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx_tgt)
        return src_batch, tgt_batch
    return collate_fn
