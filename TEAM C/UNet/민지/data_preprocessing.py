import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

'''
UNet 학습을 위한 EM Segmentation Challenge에 사용된 membrane 데이터셋, 세포 이미지
.tif 파일 내에 512x512 크기의 이미지 30개가 압축
'''
# data from https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet/blob/master/datasets

def data_preprocessing(data_dir):
    name_label = "train-labels.tif"
    name_input = "train-volume.tif"

    img_label = Image.open(os.path.join(data_dir, name_label))
    img_input = Image.open(os.path.join(data_dir, name_input))

    ## train/test/val 폴더 생성
    nframe = img_label.n_frames

    nframe_train = 24
    nframe_val = 3
    nframe_test = 3

    dir_save_train = os.path.join(data_dir, 'train')
    dir_save_val = os.path.join(data_dir, 'val')
    dir_save_test = os.path.join(data_dir, 'test')

    os.makedirs(dir_save_train, exist_ok=True)
    os.makedirs(dir_save_val, exist_ok=True)
    os.makedirs(dir_save_test, exist_ok=True)

    ## 전체 이미지 30개를 섞어줌
    id_frame = np.arange(nframe)
    np.random.shuffle(id_frame)

    ## 선택된 train 이미지를 npy 파일로 저장
    offset_nframe = 0

    for i in range(nframe_train):
        img_label.seek(id_frame[i + offset_nframe])
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

    ## 선택된 val 이미지를 npy 파일로 저장
    offset_nframe = nframe_train

    for i in range(nframe_val):
        img_label.seek(id_frame[i + offset_nframe])
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

    ## 선택된 test 이미지를 npy 파일로 저장
    offset_nframe = nframe_train + nframe_val

    for i in range(nframe_test):
        img_label.seek(id_frame[i + offset_nframe])
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

    print("\n### Data Preprocessed. ###\n")


if __name__ == '__main__':
    data_dir="./data/"
    data_preprocessing(data_dir)