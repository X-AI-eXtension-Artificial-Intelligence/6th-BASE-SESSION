"""
@author : Hyunwoong
@when : 2019-12-22
@homepage : https://github.com/gusdnd852
"""

import math
from collections import Counter
import numpy as np


def bleu_stats(hypothesis, reference):
    """
    BLEU 점수 계산을 위한 통계 정보 추출

    hypothesis: 예측 문장 (토큰 리스트)
    reference: 정답 문장 (토큰 리스트)

    반환: 다음 10개의 요소로 구성된 리스트
        [0] 예측 문장 길이 (c)
        [1] 참조 문장 길이 (r)
        [2] 1-gram 매치 개수
        [3] 1-gram 전체 개수
        [4] 2-gram 매치 개수
        [5] 2-gram 전체 개수
        [6] 3-gram 매치 개수
        [7] 3-gram 전체 개수
        [8] 4-gram 매치 개수
        [9] 4-gram 전체 개수
    """

    stats = []
    stats.append(len(hypothesis))   # 예측 문장 길이 c
    stats.append(len(reference))    # 참조 문장 길이 r

    for n in range(1, 5):  # 1~4-gram
        # 예측 문장의 n-gram 추출 (튜플 형태)
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )

        # 정답 문장의 n-gram 추출
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        # 겹치는 n-gram의 개수 (교집합 합)
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))

        # 예측 문장의 n-gram 개수
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    return stats


def bleu(stats):
    """
    n-gram 통계를 기반으로 BLEU 점수 계산
    stats: bleu_stats() 함수로부터 얻은 통계 리스트
    반환: BLEU score (0~1 사이 값)
    """

    # 통계 값 중 하나라도 0이면 BLEU 점수는 0
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0

    # c: 예측 문장 길이, r: 참조 문장 길이
    (c, r) = stats[:2]

    # n-gram precision의 로그 평균
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.

    # brevity penalty 적용 + exp 취해 최종 BLEU 계산
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """
    전체 dev 세트(여러 문장)에 대한 BLEU 점수 계산 함수

    hypotheses: 예측 문장들의 리스트 
    reference: 정답 문장들의 리스트

    반환: 100점 만점 기준의 BLEU 점수
    """

    # 통계 누적을 위한 numpy 배열 (길이 10)
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # 각 문장 쌍에 대해 통계 누적
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))

    # BLEU 계산 후 100점 기준으로 변환
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    """
    숫자 인덱스 리스트를 단어로 변환해 문자열로 합치는 함수

    x: 정수 인덱스 리스트
    vocab: vocab 객체 (vocab.itos는 index-to-string 매핑 리스트)

    반환: "<"로 시작하는 특수 토큰 제외한 단어 문자열 (띄어쓰기 포함)
    """

    words = []

    for i in x:
        word = vocab.itos[i]

        # 특수 토큰(<pad>, <eos> 등)은 제외
        if '<' not in word:
            words.append(word)

    return " ".join(words)
