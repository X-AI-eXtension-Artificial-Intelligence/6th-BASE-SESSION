import argparse

def get_hParams():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Train model을 위한 배치 크기"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,  # float 타입으로 변경
        default=1e-4,
        help="Train model을 위한 학습률"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=100,
        help="Train model을 위한 에폭 수"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_vanilla",
        help="학습된 모델을 저장할 모델 이름. '.pth'와 같은 확장자 제외하고 모델 이름만 입력"
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default=None,
        help="학습된 모델이 저장된 실제 이름 (ex. unet_vanilla_epoch100). '.pth'와 같은 확장자 제외하고 모델 이름만 입력"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_hParams()  # 함수 호출 시 괄호 추가
    lr = args.learning_rate
    print(lr)
