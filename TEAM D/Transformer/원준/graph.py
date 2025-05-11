import matplotlib.pyplot as plt
import re


# 특정 텍스트 파일을 열어 숫자 리스트로 반환하는 함수
def read(name):
    f = open(name, 'r')               # 파일을 읽기 모드로 염
    file = f.read()                   # 파일 전체 내용을 문자열로 읽음
    file = re.sub('\\[', '', file)    # 왼쪽 대괄호 '[' 제거
    file = re.sub('\\]', '', file)    # 오른쪽 대괄호 ']' 제거
    f.close()                         # 파일 닫기

    # ','로 나눈 뒤 각 요소를 float형으로 변환하여 리스트로 반환
    return [float(i) for idx, i in enumerate(file.split(','))]


# 'loss' 또는 'bleu' 모드에 따라 학습 결과를 시각화하는 함수
def draw(mode):
    if mode == 'loss':
        train = read('./result/train_loss.txt')    # 학습 손실 값 로딩
        test = read('./result/test_loss.txt')      # 검증 손실 값 로딩
        plt.plot(train, 'r', label='train')        # 빨간색 선으로 학습 손실 그래프
        plt.plot(test, 'b', label='validation')    # 파란색 선으로 검증 손실 그래프
        plt.legend(loc='lower left')               # 범례 위치 지정

    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')           # BLEU 점수 로딩
        plt.plot(bleu, 'b', label='bleu score')    # 파란색 선으로 BLEU 그래프
        plt.legend(loc='lower right')              # 범례 위치 지정

    plt.xlabel('epoch')                            # x축 레이블: 에폭
    plt.ylabel(mode)                               # y축 레이블: loss 또는 bleu
    plt.title('training result')                   # 그래프 제목
    plt.grid(True, which='both', axis='both')      # x, y축에 격자선 표시
    plt.show()                                     # 그래프 화면에 표시


# 스크립트가 단독 실행될 때만 실행됨
if __name__ == '__main__':
    draw(mode='loss')  # 학습 손실 그래프 출력
    draw(mode='bleu')  # BLEU 점수 그래프 출력
