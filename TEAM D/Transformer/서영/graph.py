"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib
import re                       # 문자열 정규 표현식 처리용


def read(name):
    # 텍스트 파일을 읽고 숫자 리스트로 변환하는 함수

    f = open(name, 'r')         # 파일 열기
    file = f.read()             # 전체 파일 내용 읽기

    # 파일에서 [ ] 괄호 제거
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)

    f.close()

    # 콤마(,) 기준으로 나눈 뒤 float로 변환하여 리스트로 반환
    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    # 학습 결과 시각화를 수행하는 함수
    # mode='loss' 또는 'bleu' 선택 가능

    if mode == 'loss':
        train = read('./result/train_loss.txt')     # 학습 손실 읽기
        test = read('./result/test_loss.txt')       # 검증 손실 읽기
        plt.plot(train, 'r', label='train')         # 빨간색으로 학습 손실 그래프
        plt.plot(test, 'b', label='validation')     # 파란색으로 검증 손실 그래프
        plt.legend(loc='lower left')                # 범례 위치 설정

    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')            # BLEU 점수 읽기
        plt.plot(bleu, 'b', label='bleu score')     # 파란색 BLEU 그래프
        plt.legend(loc='lower right')               # 범례 위치 설정

    # 공통 시각화 설정
    plt.xlabel('epoch')                             # x축: epoch
    plt.ylabel(mode)                                # y축: 선택된 지표
    plt.title('training result')                    # 그래프 제목
    plt.grid(True, which='both', axis='both')       # 격자선 표시
    plt.show()                                       # 그래프 출력


if __name__ == '__main__':
    draw(mode='loss')  # 손실 그래프 출력
    draw(mode='bleu')  # BLEU 그래프 출력
