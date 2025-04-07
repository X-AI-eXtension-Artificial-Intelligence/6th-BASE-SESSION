import os
import numpy as np
import torch
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import torchvision.transforms as transforms


class DatasetForSeg(torch.utils.data.Dataset):

    # torch.utils.data.Dataset 이라는 파이토치 base class를 상속받아
    # 그 method인 __len__(), __getitem__()을 오버라이딩 해줘서
    # 사용자 정의 Dataset class를 선언한다

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        # 문자열 검사해서 'label'이 있으면 True
        # 문자열 검사해서 'input'이 있으면 True
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    '''
    canny 알고리즘 : 엣지를 검출하는 알고리즘
    입력은 반드시 그레이 스케일(단일 채널)의 8bit 정수형 이미지여야 함 => 즉 (H, W)여야 함
    (H,W)에 대해 canny 엣지 검출 수행할 건데, 요구하는 타입인 uint8로 변환하고, 각각 어느정도로 만들지 100, 200 파라미터 줌

    '''

    # 여기가 데이터 load하는 파트
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        input_canny = cv2.Canny(inputs.astype(np.uint8), 400, 500)
        
        # normalize: 이미지는 0~255 값을 가지고 있어 이를 0~1사이로 scaling
        label = label / 255.0
        inputs = inputs / 255.0
        input_canny = input_canny /255.0

        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)
        input_canny = input_canny.astype(np.float32)

        # 인풋 데이터 차원이 2이면, 채널 축을 추가해줘야한다.
        # 파이토치 인풋은 (batch, 채널, 행, 열)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]  # (1,512,512)
        if input_canny.ndim == 2:
            input_canny = input_canny[:, :, np.newaxis] 

        data = {'input': inputs, 'input_canny': input_canny, 'label': label}
       
        if self.transform:
            transformed_input = self.transform(inputs)
            transformed_label = self.transform(label)
            transformed_canny = self.transform(input_canny)
            data = {'input': transformed_input, 'input_canny': transformed_canny, 'label': transformed_label}
        # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행
        return data

    def show_image(self):
        print("### Number of classes:", self.__len__())
        print("### Number of samples:", self.__len__())

        random_index = np.random.randint(0, self.__len__())
        input_tensor, canny_tensor, label_tensor = self.__getitem__(random_index)['input'], self.__getitem__(random_index)['input_canny'], self.__getitem__(random_index)['label']
        print("### Shape of each image:", input_tensor.shape)
        # PyTorch Tensor → NumPy 변환 (1, H, W) → (H, W)
        input_numpy = input_tensor.squeeze().numpy()
        label_numpy = label_tensor.squeeze().numpy()
        canny_numpy = canny_tensor.squeeze().numpy()

        input_numpy = Image.fromarray((input_numpy * 255).astype(np.uint8))
        label_numpy = Image.fromarray((label_numpy * 255).astype(np.uint8))
        canny_numpy = Image.fromarray((canny_numpy * 255).astype(np.uint8))
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(input_numpy, cmap='gray')
        axes[0].set_title("Input")
        axes[1].imshow(label_numpy, cmap='gray')
        axes[1].set_title("Label")
        axes[2].imshow(canny_numpy, cmap='gray')
        axes[2].set_title("Canny Input")
        plt.suptitle("Data - Input / Label / Canny Input")  # 전체 제목 설정
        plt.show()
        plt.savefig('dataset_img.png', dpi=300, bbox_inches='tight')

def data_transform():
    # Tran5sform 정의
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Resize((572, 572)),
                                    transforms.Normalize(0.5, 0.5)])
    return transform

if __name__ == '__main__':
    transform = data_transform()
    dataset = DatasetForSeg(data_dir='./data/train/', transform=transform)
    dataset.show_image()