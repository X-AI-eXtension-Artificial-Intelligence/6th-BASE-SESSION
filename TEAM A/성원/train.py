
import vgg
import dataset


import torch
import torch.nn as nn

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from tqdm import trange

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA 사용 가능 여부 확인하여 device 설정

torch.manual_seed(777)  # 랜덤 시드 고정 (재현 가능성 확보)
if device =='cuda':
    torch.cuda.manual_seed_all(777)


# 배치 사이즈, 학습률, 에포크 지정
batch_size = 100
learning_rate = 0.00005
num_epoch = 100


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11  # conv가 8개 생성되고 아까 FC가 3개 있었으니 vgg11 
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 10 + 3 = vgg 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #13 + 3 = vgg 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 16 +3 =vgg 19
    'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M']
}


# DataLoader로 train, test set 준비, 순서 섞기

train_loader = torch.utils.data.DataLoader(  # 대용량 데이터를 효율적으로 로드하기 위해 미니배치 단위로 데이터를 불러오는 클래스
                                          dataset.data_transform(train=True),  # 데이터로더 클래스에 넣을 데이터(훈련 데이터) 
                                          batch_size=batch_size,  # 512장씩 미니배치로 설정
                                          shuffle=True,  # epoch마다 데이터 순서를 섞음
                                          num_workers=0)  # 멀티스레딩 비활성화 -> 하나의 프로세스(CPU)로 데이터를 불러옴



vgg19= vgg.VGG(vgg.make_layers(cfg['E']), 10, True).to(device)  # E 경우의 vgg19 구현, 분류클래스 10개, 가중치 초기화 실행  



criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(vgg19.parameters(), lr = 0.005, momentum=0.9)

lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 학습률을 조금 씩 줄이기 위한 코드 
# 5회마다 학습률에 0.9씩 곱해주세요 




print(len(train_loader))
epochs = 50

# for epoch in range(epochs):  # loop over the dataset multiple times
#     running_loss = 0.0
#     lr_sche.step()
    
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = vgg19(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 30 == 29:    # print every 30 mini-batches
#             # loss_tracker(loss_plt, torch.Tensor([running_loss/30]), torch.Tensor([i + epoch*len(trainloader) ]))
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 30))
#             running_loss = 0.0
        

# print('Finished Training')
# --------------------------------------------------------------------
loss_arr = [] #loss 담아줄 array 생성
for i in trange(num_epoch):  # 50 epoch 학습
    for j, [image,label] in enumerate(train_loader):  # image랑 label 불러오기. j 는 없어도 되지만 추후 배치번호를 추적할 일이 생김을 대비

        #GPU에 이미지랑 Label 얹기
        inputs = image.to(device)
        labels = label.to(device)

        optimizer.zero_grad()  # 이전 gradient 초기화
        output = vgg19.forward(inputs)  # 순전파
        loss = criterion(output, labels)  # 손실함수 값 
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 갱신 

        # 10번째 배치마다 loss 출력 후 array에 저장
        if i % 10 == 0:
          print(loss)
          loss_arr.append(loss.cpu().detach().numpy())

# loss curve 그리기
plt.plot(loss_arr)
# loss curve 그래프 이미지 저장
plt.savefig('CIFAR10_VGG19_Loss_curve.png')
plt.show()

# model 가중치 저장
torch.save(vgg19.state_dict(), "VGG19_model.pth") 


''' 가중치만 저장
torch.save(vgg19.state_dict(), "VGG19_model.pth") '''

''' 가중치 불러오기 
vgg19 = models.vgg19()
vgg19.load_state_dict(torch.load("VGG19_model.pth"))
vgg19.eval()  '''


''' 모델 전체 저장
torch.save(vgg19, "VGG19_model_full.pth")  '''

''' 모델 전체 불러오기 
vgg19 = torch.load("VGG19_model_full.pth")
vgg19.eval()  # 추가 훈련 없이 테스트하려면 평가모드로 변경이 필수 '''






































