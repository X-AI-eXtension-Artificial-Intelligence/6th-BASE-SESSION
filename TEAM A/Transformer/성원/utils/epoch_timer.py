""" 
한 epoch(에폭)이 걸린 시간을 계산하고,
그걸 '분'과 '초'로 나눠서 보기 쉽게 반환하는 도우미 함수
""" 



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs





