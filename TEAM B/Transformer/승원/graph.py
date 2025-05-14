"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):  # 함수: 파일 읽어서 숫자 리스트 반환
    f = open(name, 'r')  # 파일 열기
    file = f.read()  # 전체 내용 읽기
    file = re.sub('\[', '', file)  # '[' 제거
    file = re.sub('\]', '', file)  # ']' 제거
    f.close()  # 파일 닫기

    return [float(i) for idx, i in enumerate(file.split(','))]  # 쉼표로 분리해 float 리스트 생성


def draw(mode):  # 함수: 모드에 따라 그래프 그리기
    if mode == 'loss':  # 모드: 손실
        train = read('./result/train_loss.txt')  # 학습 손실 읽기
        test = read('./result/test_loss.txt')  # 검증 손실 읽기
        plt.plot(train, 'r', label='train')  # 학습 손실 빨간선
        plt.plot(test, 'b', label='validation')  # 검증 손실 파란선
        plt.legend(loc='lower left')  # 범례 위치: 왼쪽 하단


    elif mode == 'bleu':  # 모드: BLEU 점수
        bleu = read('./result/bleu.txt')  # BLEU 점수 읽기
        plt.plot(bleu, 'b', label='bleu score')  # BLEU 점수 파란선
        plt.legend(loc='lower right')  # 범례 위치: 오른쪽 하단

    plt.xlabel('epoch')  # x축 레이블 설정
    plt.ylabel(mode)  # y축 레이블 설정
    plt.title('training result')  # 그래프 제목 설정
    plt.grid(True, which='both', axis='both')  # 그리드 표시
    plt.show()  # 그래프 출력


if __name__ == '__main__':  # 스크립트 직접 실행 시
    draw(mode='loss')  # 손실 그래프 그리기
    draw(mode='bleu')  # BLEU 그래프 그리기