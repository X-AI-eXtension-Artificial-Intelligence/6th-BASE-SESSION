

# 학습 에폭 시간 계산 함수
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time         # 총 경과 시간(초 단위)
    elapsed_mins = int(elapsed_time / 60)        # 경과 시간 중 '분' 단위 추출 (정수)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 나머지 '초' 단위 계산
    return elapsed_mins, elapsed_secs            # (분, 초) 형태로 반환
