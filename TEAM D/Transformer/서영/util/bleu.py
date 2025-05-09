"""
@author : Hyunwoong
@when : 2019-12-22
@homepage : https://github.com/gusdnd852
"""
import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))  # 예측 문장 길이 (c)
    stats.append(len(reference))   # 참조 문장 길이 (r)

    for n in range(1, 5):  # 1~4-gram까지 순차적으로 검사
        # 예측 문장에서 n-gram 추출
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        # 정답 문장에서 n-gram 추출
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        # n-gram precision용: 교집합 개수 (중복 포함)
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))

        # 예측된 n-gram 총 개수
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    return stats  # 총 길이 10인 list 반환: [c, r, 맞은 1g, 총 1g, 맞은 2g, 총 2g, ..., 맞은 4g, 총 4g]


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    # 통계 중 0이 하나라도 있으면 BLEU 스코어는 0
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0

    (c, r) = stats[:2]  # 예측 길이 c, 참조 길이 r

    # 각 n-gram precision의 로그 평균 계산 (4-gram까지)
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.

    # brevity penalty + 평균 precision
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    # BLEU 통계를 위한 누적 벡터 초기화
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    for hyp, ref in zip(hypotheses, reference):
        # 각 문장 쌍에 대해 통계 계산 후 누적
        stats += np.array(bleu_stats(hyp, ref))

    return 100 * bleu(stats)  # BLEU 점수를 % 단위로 반환


def idx_to_word(x, vocab):
    """토큰 인덱스 시퀀스를 단어 시퀀스로 변환 (특수 토큰 제외)"""
    words = []
    for i in x:
        word = vocab.itos[i]  # 인덱스를 문자열로 변환
        if '<' not in word:   # <pad>, <sos>, <eos> 등 특수 토큰 제외
            words.append(word)
    words = " ".join(words)  # 공백 기준 문장 생성
    return words
