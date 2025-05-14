import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference): # hypothesis와 reference의 길이, 1~4-gram의 일치 개수와 전체 개수 등 BLEU 계산에 필요한 통계를 리스트로 반환
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis)) # 가설 문장 길이
    stats.append(len(reference)) # 참조 문장 길이
    for n in range(1, 5): # 문장 n-gram 생성 및 카운팅
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0])) # 공통 n-gram 개수 계산
        stats.append(max([len(hypothesis) + 1 - n, 0])) # 가설 문장의 전체 n-gram 개수 계산
    return stats


def bleu(stats): # 통계값을 받아, n-gram precision과 brevity penalty를 적용해 BLEU 점수를 계산
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0: # 0값이 있는 경우 BLEU=0 반환
        return 0
    (c, r) = stats[:2] # 길이 정보 추출
    log_bleu_prec = sum( # n-gram 정밀도 계산
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    # 최종 BLEU 점수 계산 (길이 페널티 사용)
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference): # 여러 쌍의 hypothesis와 reference에 대해 bleu_stats를 누적합산하여 전체 BLEU 점수를 계산하고, 0~100 점수로 반환
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(x, vocab): # 인덱스 시퀀스를 받아, vocab에서 실제 단어로 변환하고 특수 토큰을 제외해 문자열로 반환
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words
