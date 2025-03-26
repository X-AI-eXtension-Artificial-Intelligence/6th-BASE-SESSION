
import torch 
import vgg  # 모델 호출
import dataset  # 데이터셋 호출 


batch_size = 100  # 배치사이즈 100 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #CUDA GPU 활용


test_loader = torch.utils.data.DataLoader(  # 대용량 데이터를 효율적으로 로드하기 위해 미니배치 단위로 데이터를 불러오는 클래스
                                          dataset.data_transform(train=False),  # 데이터로더 클래스에 넣을 데이터(테스트 데이터) 
                                          batch_size=batch_size,  # 100장씩 미니배치로 설정
                                          shuffle=True,  # epoch마다 데이터 순서를 섞음
                                          num_workers=0)  # 멀티스레딩 비활성화 -> 하나의 프로세스(CPU)로 데이터를 불러옴

# 모델 불러오기 
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11  # conv가 8개 생성되고 아까 FC가 3개 있었으니 vgg11 
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 10 + 3 = vgg 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #13 + 3 = vgg 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 16 +3 =vgg 19
    'custom1' : [64,64,64,'M',128,128,128,'M',256,256,256,'M', 512, 512, 512, 'M', 1020, 1024, 1024, 'M']
}
# vgg19 = vgg.VGG(vgg.make_layers(cfg['E']), 10, True).to(device)  # 모델 구조 잡고
# vgg19.load_state_dict(torch.load("VGG19_model.pth"))  # 학습된 가중치 불러옴 
# transformed_model = vgg.VGG(vgg.make_layers(cfg['custom1'], batch_norm=True), 10, True).to(device)
# transformed_model.load_state_dict(torch.load("transformed_model.pth"))

transformed_without_BN_model = vgg.VGG(vgg.make_layers(cfg['custom1'], batch_norm=False), 10, True).to(device)
transformed_without_BN_model.load_state_dict(torch.load("transformed_without_BN_model.pth"))

# vgg19.load_state_dict(torch.load("./VGG19_model.pth", map_location=torch.device('cpu'))) # GPU로 학습한 가중치를 CPU환경에서 불러올 수 있게 







correct = 0
total = 0

transformed_model.eval()  # 드롭아웃, BN 비활성화. make_layers에서 BN을 True 해줬는가? 
with torch.no_grad():  #  gradient 계산 비활성화. 당연히 역전파도 비활성화 
    for data in test_loader:
        images, labels = data  # 데이터를 이미지와 라벨로 분리 
        images = images.to(device)
        labels = labels.to(device)
        # outputs = vgg19(images)  # 모델에 이미지 통과. 결과는 배치x클래스 수 모양의 중첩 리스트. 
        outputs = transformed_model(images)
        _, predicted = torch.max(outputs.data, 1)  # 확률이 가장 높은 1개 predicted에 저장 
        
        total += labels.size(0)  # 전체 샘플 수 증가
        
        correct += (predicted == labels).sum().item()  # 맞힌 개수 더해줌 



print(total)
print(correct)
print("Accuracy of Test Data: {}%".format(100*correct/total))  # 정확도 출력