https://www.youtube.com/watch?v=rBb597ct_FQ 유튜브 참고
https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet/?tab=readme-ov-file 깃허브 참고

결과물은 result파일 내에 저장

[원본]
![input_0000](https://github.com/user-attachments/assets/49a0d2f5-b0bd-4af4-ae40-66c4db8eb5e9)


[결과]
![output_0000](https://github.com/user-attachments/assets/386c0751-8e33-4b22-868e-6a86402daf34)


# 구조 변경
## 기존 성능(loss 값은 변동 가능성 존재)
- eval.py 실행 후 loss: 0.6925

## 1. 학습률, 학습 횟수 변경
- epoch 10 실행 - loss: 0.7117
- Lr 0.0005 실행 - loss: 0.7031
- Lr 0.005 실행 - loss: 0.6775

## 2. 층 추가(사이즈를 2048까지 확장)
- loss: 0.6885

