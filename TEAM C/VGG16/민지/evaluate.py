from torch.utils.data import DataLoader

from dataset import data_transform
from asymmetric_vgg16 import VGG16
import torch

def evaluate_model(setting_config: dict):
    device = setting_config['device']
    batch_size = setting_config['batch_size']
    model_path = setting_config['model_path']
    # data
    test_set = data_transform(train=False)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # model load
    model = VGG16(base_dim=64, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # eval
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad(): # 모델 평가 단계이므로 가중치 업뎃할 필요 없으므로 메모리 사용량과 연산 시간을 줄이기 위해 사용
        for i, [image, label] in enumerate(test_loader):
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x) # output : 모델이 예측한 클래스별 확률
            _, output_index = torch.max(output, 1) # output 텐서의 가장 큰 값과 해당 인덱스 반환

            total += label.size(0)
            correct += (output_index == y).sum().float()

    print("### Accuracy of Test Data: {}%".format(100 * correct / total))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model_path = "./model/asymmetric_vgg16_epoch100.pth" 

    setting_config = {
        "batch_size": 100,
        "device": device,
        "model_path": model_path}

    evaluate_model(setting_config)