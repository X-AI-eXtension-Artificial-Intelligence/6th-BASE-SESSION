사용한 성능 지표
1. IoU
2. Dice coefficient

사용한 데이터셋
Kaggle Data Science Bowl 2018 
다양한 세포/핵 이미지와 라벨이 포함된 대규모 데이터셋임
다양한 현미경 조건

1. stage1_train : 학습용 이미지 + 마스크
2. stage1_train_label.cav : 학습용 이미지의 라벻을 csv로 정리한 거
3. stage1_test : 테스트용 이미지(라벨 x)
4. stage1_solution.csv : 대회 종료 후 공개되는 테스트셋 정답 마스크 (RLE 형태)

- utils 에서는 RLE 파일을 인코딩,디코딩하는 함수가 있음
- data셋의 프레임별로 사이즈가 달라서...리사이즈하는 코드를 넣었음
- 모델 코드는 안 바꿈
- 평가 방식은 IoU, Dice
- 성능이 안 나와서... 고민중

60 에폭 돌린 결과...
평균 Dice: 0.2416...
평균 IoU: 0.1537...
