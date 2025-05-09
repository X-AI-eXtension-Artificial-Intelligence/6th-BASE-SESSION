"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""


def epoch_time(start_time, end_time):
    # 전체 학습 epoch 또는 step에 걸린 총 시간 계산 함수

    elapsed_time = end_time - start_time  # 경과 시간 (초 단위)
    
    elapsed_mins = int(elapsed_time / 60)  # 분 단위로 변환 (정수형)
    
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 나머지 초 계산
    
    return elapsed_mins, elapsed_secs  # (분, 초) 형태로 반환
