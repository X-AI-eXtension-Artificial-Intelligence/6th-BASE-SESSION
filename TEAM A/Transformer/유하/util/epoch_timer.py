def epoch_time(start_time, end_time): # 시작/종료 시간을 받아, 경과 시간을 (분, 초) 단위로 변환하여 반환
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
