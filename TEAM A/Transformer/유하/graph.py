# graph.py : 학습 결과(손실값, BLEU 점수 등)를 그래프로 시각화하는 코드

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r') # 읽기 모드
    file = f.read() # 파일 내용 문자열로 읽어오기
    file = re.sub('\\[', '', file) # [ 모두 제거
    file = re.sub('\\]', '', file) # ] 모두 제거
    f.close() # 파일 닫음

    return [float(i) for idx, i in enumerate(file.split(','))] # 문자열 쉼표로 분리 후, 각 요소를 float 타입으로 변환하여 리스트 반환


def draw(mode): # 그래프 그리기 함수 정의
    if mode == 'loss': # 손실 그래프
        train = read('./result/train_loss.txt') # 학습 손실값이 저장된 파일 읽어 리스트로 만듦
        test = read('./result/test_loss.txt') # 검증 손실값이 저장된 파일 읽어 리스트로 만듦
        plt.plot(train, 'r', label='train') # 학습 손실값 빨간색 선으로 그리기
        plt.plot(test, 'b', label='validation') # 검증 손실값 파란색 선으로 그리기
        plt.legend(loc='lower left') # 범례


    elif mode == 'bleu': # BLEU 점수 그래프
        bleu = read('./result/bleu.txt') 
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch') # x축 이름
    plt.ylabel(mode) # y축 이름
    plt.title('training result') # 그래프 제목
    plt.grid(True, which='both', axis='both') # x축과 y축 모두에 그리드 표시
    plt.show() # 그래프 화면에 표시


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
