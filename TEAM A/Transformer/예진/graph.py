"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt 
import re 


def read(name):
    """
    주어진 파일에서 숫자 데이터를 읽어 리스트로 반환하는 함수

    - name: 파일 경로 문자열
    - 반환값: float형 숫자 리스트

    파일 형식은 '[1.0, 2.3, 3.1]'과 같은 형태라고 가정.
    괄호를 제거하고, 쉼표로 나눠서 float형으로 변환.
    """
    f = open(name, 'r')           # 파일 열기
    file = f.read()               

    file = re.sub('\\[', '', file)  # 왼쪽 대괄호 제거
    file = re.sub('\\]', '', file)  # 오른쪽 대괄호 제거
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]  # 쉼표 기준 분할 후 float로 변환


def draw(mode):
    """
    손실 또는 BLEU 점수를 시각화하여 출력하는 함수

    - mode: 'loss' or 'bleu'
      - 'loss'면 train_loss.txt와 test_loss.txt를 읽어 시각화
      - 'bleu'면 bleu.txt를 읽어 시각화
    """

    if mode == 'loss':
        train = read('./result/train_loss.txt')  # 학습 손실 불러오기
        test = read('./result/test_loss.txt')    # 검증 손실 불러오기
        plt.plot(train, 'r', label='train')      # 빨간색 선: 학습 손실
        plt.plot(test, 'b', label='validation')  # 파란색 선: 검증 손실
        plt.legend(loc='lower left')             # 범례 왼쪽 아래 배치

    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')         # BLEU 점수 불러오기
        plt.plot(bleu, 'b', label='bleu score')  # 파란색 선: BLEU 점수
        plt.legend(loc='lower right')            # 범례 오른쪽 아래 배치

    # 그래프 출력
    plt.xlabel('epoch')      
    plt.ylabel(mode)             
    plt.title('training result') 
    plt.grid(True, which='both', axis='both')  
    plt.show()                  


if __name__ == '__main__':
    draw(mode='loss')  # 학습 손실 그래프 출력
    draw(mode='bleu')  # BLEU 점수 그래프 출력
