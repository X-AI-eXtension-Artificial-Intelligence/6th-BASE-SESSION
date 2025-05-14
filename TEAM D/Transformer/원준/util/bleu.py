
import math                          # 로그, 지수 계산용
from collections import Counter      # n-gram 빈도 계산용
import numpy as np                   # 배열 기반 수치 연산용

# 단일 예측/정답 쌍에서 BLEU 계산에 필요한 통계 추출
def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))   # c: 예측 문장 길이  candidate length 
    stats.append(len(reference))    # r: 정답 문장 길이  reference length 

    # 1~4-gram 각각에 대해 precision 통계 계산
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))  # 일치하는 n-gram 수
        stats.append(max([len(hypothesis) + 1 - n, 0]))              # 전체 예측 n-gram 수
    return stats  # 총 10개 항목 반환: [c, r, p1_n, p1_d, ..., p4_n, p4_d]

# 주어진 통계 정보를 바탕으로 최종 BLEU 점수 계산
def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0  # 0이 하나라도 있으면 로그 연산 불가 → BLEU 0

    c, r = stats[:2]
    log_bleu_prec = sum([
        math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])
    ]) / 4.  # 1~4-gram 평균 로그 precision

    bp = min([0, 1 - float(r) / c])  # brevity penalty
    return math.exp(bp + log_bleu_prec)

# 전체 예측/정답 리스트에 대해 평균 BLEU 계산 (0~100)
def get_bleu(hypotheses, reference):
    stats = np.array([0.] * 10)  # 통계 누적 배열 초기화
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

# 인덱스 시퀀스를 단어 시퀀스로 변환 (<pad>, <sos> 등 특수 토큰 제거)
def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]  # 인덱스를 단어로 변환
        if '<' not in word:   # 특수 토큰 제외
            words.append(word)
    return " ".join(words)    # 단어들을 공백 기준으로 이어 문자열 반환
