import os
import numpy as np
from PIL import Image
import cv2

def create_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_npy(path, array):
    np.save(path, array)

def preprocess_and_save(img_input, img_label, save_dir, idx):
    input_arr = np.asarray(img_input)
    label_arr = np.asarray(img_label)
    canny_arr = cv2.Canny(input_arr.astype(np.uint8), 100, 200)

    save_npy(os.path.join(save_dir, f'input_{idx:03d}.npy'), input_arr)
    save_npy(os.path.join(save_dir, f'label_{idx:03d}.npy'), label_arr)
    save_npy(os.path.join(save_dir, f'input_canny_{idx:03d}.npy'), canny_arr)

def data_preprocessing(data_dir):
    name_label = "train-labels.tif"
    name_input = "train-volume.tif"

    img_label = Image.open(os.path.join(data_dir, name_label))
    img_input = Image.open(os.path.join(data_dir, name_input))

    nframe = img_label.n_frames
    nframe_train, nframe_val, nframe_test = 24, 3, 3

    dir_train = os.path.join(data_dir, 'train')
    dir_val = os.path.join(data_dir, 'val')
    dir_test = os.path.join(data_dir, 'test')
    create_dirs(dir_train, dir_val, dir_test)

    id_frame = np.arange(nframe)
    np.random.shuffle(id_frame)

    offset = 0
    for i in range(nframe_train):
        img_label.seek(id_frame[i + offset])
        img_input.seek(id_frame[i + offset])
        preprocess_and_save(img_input, img_label, dir_train, i)

    offset += nframe_train
    for i in range(nframe_val):
        img_label.seek(id_frame[i + offset])
        img_input.seek(id_frame[i + offset])
        preprocess_and_save(img_input, img_label, dir_val, i)

    offset += nframe_val
    for i in range(nframe_test):
        img_label.seek(id_frame[i + offset])
        img_input.seek(id_frame[i + offset])
        preprocess_and_save(img_input, img_label, dir_test, i)

    print("\n Data 생성 완료! \nSaved to:")
    print(f" - {dir_train}\n - {dir_val}\n - {dir_test}\n")

if __name__ == '__main__':
    data_dir = "./data/"
    data_preprocessing(data_dir)
