from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import data_transform
from vgg16 import VGG16
import torch
import torch.nn as nn

def train_model(setting_config: dict):
    # setting
    batch_size = setting_config['batch_size']
    learning_rate = setting_config['learning_rate']
    num_epoch = setting_config['num_epoch']
    device = setting_config['device']

    # data
    train_set = data_transform(train=True)
    test_set = data_transform(train=False)

    # DataLoader : 미니배치(batch) 단위로 데이터를 제공
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)


    # Train
    model = VGG16(base_dim=64).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_arr = []

    for i in tqdm(range(num_epoch), total=num_epoch, desc='training...'):
        for j,[image,label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

        if i%10 == 0:
            print(f'epcoh {i} loss : ', loss)
            loss_arr.append(loss.cpu().detach().numpy())

    # 학습 완료된 모델 저장
    torch.save(model.state_dict(), setting_config['save_model_path'])

if __name__ == '__main__':
    setting_config = {
        "batch_size": 4,
        "learning_rate": 0.001,
        "num_epoch": 100,
        "device": torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
        "save_model_path": "./model/vgg16_epoch100.pth" # "./model/vgg16_10.pth"
    }
    train_model(setting_config)
